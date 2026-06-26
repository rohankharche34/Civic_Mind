import os
import numpy as np
import pandas as pd
import networkx as nx
import faiss
import onnxruntime
from tokenizers import Tokenizer
from collections import defaultdict
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")


def _hour_to_peak(hour: int) -> bool:
    return hour in [8, 9, 10, 17, 18, 19, 20]


def _forecast_to_weather(forecast: str) -> str:
    mapping = {
        "clear": "Clear",
        "cloudy": "Cloudy",
        "rain_light": "Light Rain",
        "rain_6h": "Heavy Rain",
        "storm": "Heavy Rain",
    }
    return mapping.get(forecast, "Clear")


def _conditions_to_bool(conditions: list, target: str) -> bool:
    return any(target in c for c in conditions)


_CONDITION_MAP = {
    "heavy_rain": "Heavy Rain",
    "rain": "Heavy Rain",
    "road_construction": "Road Construction",
    "construction": "Road Construction",
    "festival": "Festival Crowd",
    "crowd": "Festival Crowd",
    "peak_traffic": "High Traffic Volume",
    "high_traffic": "High Traffic Volume",
}


def _normalize_condition(cond: str) -> str:
    return _CONDITION_MAP.get(cond.replace(" ", "_").lower(), cond.title())


class _ONNXEmbedder:
    def __init__(self, model_path: str):
        self.tokenizer = Tokenizer.from_file(
            os.path.join(model_path, "tokenizer.json")
        )
        self.tokenizer.enable_padding(
            pad_id=0, pad_token="[PAD]", length=128
        )
        self.tokenizer.enable_truncation(max_length=128)
        self.session = onnxruntime.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = [self.tokenizer.encode(t) for t in texts]
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attn_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        tok_type = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": tok_type,
        })
        emb = outputs[0]
        attn = attn_mask[:, :, np.newaxis].astype(np.float32)
        emb = np.sum(emb * attn, axis=1) / np.maximum(np.sum(attn, axis=1), 1e-9)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype(np.float32)


class CivicMindInference:

    def __init__(self, k_matches: int = 20):
        self.k_matches = k_matches
        self.dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        self._model = None
        self._index = None
        self._df = None
        self._graph = None
        self._graph_cache = None

        self._build_graph_cache()
        self.episode_hits = defaultdict(int)
        self.global_counter = 0
        self.importance = defaultdict(float)
        self.pruned_episodes = set()

        state_path = os.path.join(self.dataset_dir, "learning_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
                self.episode_hits.update(state["episode_hits"])
                self.global_counter = state["global_counter"]
                self.importance.update(state.get("importance", {}))
                self.pruned_episodes = set(state.get("pruned_episodes", []))

        self.alpha = 0.01

    @property
    def model(self):
        if self._model is None:
            onnx_path = os.path.join(self.dataset_dir, "onnx-model")
            if not os.path.exists(os.path.join(onnx_path, "model.onnx")):
                raise RuntimeError(
                    f"ONNX model not found at {onnx_path}. "
                    "Run 'python scripts/export_onnx_model.py' first."
                )
            self._model = _ONNXEmbedder(onnx_path)
        return self._model

    @property
    def index(self):
        if self._index is None:
            self._index = faiss.read_index(
                os.path.join(self.dataset_dir, "civicmind_memory.index")
            )
        return self._index

    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_pickle(
                os.path.join(self.dataset_dir, "episode_lookup.pkl")
            )
        return self._df

    @property
    def graph(self):
        if self._graph is None:
            self._graph = nx.read_graphml(
                os.path.join(self.dataset_dir, "civicmind_graph.graphml")
            )
            self._build_graph_cache()
        return self._graph

    def _build_graph_cache(self):
        self._concept_out = defaultdict(list)
        self._preceded_out = defaultdict(list)
        if self._graph is None:
            return
        for u, v, d in self._graph.edges(data=True):
            etype = d.get("edge_type")
            if etype == "CONCEPT":
                self._concept_out[u].append((v, d.get("weight", 1)))
            elif etype == "PRECEDED":
                self._preceded_out[u].append(
                    (v, d.get("time_gap_hours", 0))
                )

    def _build_query_text(
        self,
        weather: str,
        area_type: str,
        construction_active: bool,
        festival_active: bool,
        peak_hour: bool,
        event_name: str = "",
    ) -> str:
        parts = [
            f"Weather: {weather}",
            f"Area Type: {area_type}",
            f"Construction Active: {construction_active}",
            f"Festival Active: {festival_active}",
            f"Peak Hour: {peak_hour}",
        ]
        if event_name:
            parts.append(f"Event: {event_name}")
        return "\n".join(parts) + "\n"

    def predict(self, situation: dict) -> dict:
        self.global_counter += 1
        _ = self.graph  # ensure graph+cache are loaded before CONCEPT/preceded lookups
        ts = pd.Timestamp(situation["timestamp"])
        weather = _forecast_to_weather(
            situation.get("weather_forecast", "clear")
        )
        area_type = situation.get("ward", "Residential")
        peak_hour = _hour_to_peak(ts.hour)
        conditions = situation.get("active_conditions", [])
        construction_active = _conditions_to_bool(
            conditions, "construction"
        )
        festival_active = situation.get("event_name") is not None

        query_text = self._build_query_text(
            weather, area_type, construction_active, festival_active, peak_hour,
            event_name=situation.get("event_name", ""),
        )
        query_emb = self.model.encode([query_text]).astype("float32")
        distances, indices = self.index.search(query_emb, self.k_matches)

        similarities = 1.0 / (1.0 + distances[0])

        matched = []
        for rank, idx in enumerate(indices[0]):
            row = self.df.iloc[idx]
            matched.append({
                "rank": rank,
                "episode_id": row["episode_id"],
                "similarity": float(similarities[rank]),
                "event_type": row["event_type"],
                "severity": row["severity"],
                "weather": row["weather"],
                "area_type": row["area_type"],
                "intervention": row["intervention"],
                "outcome_score": float(row["outcome_score"]),
                "citizens_affected": int(row["citizens_affected"]),
                "avg_delay_minutes": int(row["avg_delay_minutes"]),
                "construction_active": bool(row["construction_active"]),
                "festival_active": bool(row["festival_active"]),
                "peak_hour": bool(row["peak_hour"]),
            })

        matched = sorted(matched, key=lambda x: (x["similarity"], self.episode_hits.get(x["episode_id"], 0)), reverse=True)
        matched = [m for m in matched if m["episode_id"] not in self.pruned_episodes]

        event_scores = defaultdict(float)
        event_match_counts = defaultdict(int)
        total_sim = sum(similarities)

        for m in matched:
            sim = m["similarity"]
            event_scores[m["event_type"]] += sim
            event_match_counts[m["event_type"]] += 1
            self.episode_hits[m["episode_id"]] += 1
            self.importance[m["episode_id"]] += self.alpha * self.episode_hits[m["episode_id"]]

        risk_scores_raw = {}
        for evt, score in event_scores.items():
            risk_scores_raw[evt] = round(score / total_sim, 4)

        transition_risks = defaultdict(float)
        for m in matched:
            ep = m["episode_id"]
            sim = m["similarity"]
            for target, gap in self._preceded_out.get(ep, []):
                t_type = self.graph.nodes[target].get("event_type")
                if t_type:
                    transition_risks[t_type] += sim * max(0, 1 - gap / 6)

        total_transition_sim = sum(
            sim for m in matched
            for _, _ in self._preceded_out.get(m["episode_id"], [])
        ) or 1
        for evt in transition_risks:
            transition_risks[evt] = round(
                transition_risks[evt] / total_transition_sim, 4
            )

        concept_risks = defaultdict(float)
        for cond in conditions:
            node = _normalize_condition(cond)
            for target, weight in self._concept_out.get(node, []):
                concept_risks[target] += weight
        total_concept_weight = sum(concept_risks.values()) or 1
        for evt in concept_risks:
            concept_risks[evt] = round(
                concept_risks[evt] / total_concept_weight, 4
            )

        all_events = set(risk_scores_raw) | set(transition_risks) | set(concept_risks)
        risk_scores = []
        for evt in all_events:
            r = risk_scores_raw.get(evt, 0)
            t = transition_risks.get(evt, 0)
            c = concept_risks.get(evt, 0)
            combined = round(0.5 * r + 0.3 * t + 0.2 * c, 4)
            risk_scores.append({
                "event": evt,
                "retrieval_score": r,
                "transition_score": t,
                "concept_score": c,
                "combined_score": combined,
            })
        risk_scores.sort(key=lambda x: x["combined_score"], reverse=True)

        seen_actions = set()
        actions = []
        for m in matched:
            if m["intervention"] == "No Action":
                continue
            key = (m["intervention"], m["event_type"])
            if key in seen_actions:
                continue
            seen_actions.add(key)
            score = round(m["similarity"] * m["outcome_score"], 4)
            actions.append({
                "action": m["intervention"],
                "target_event": m["event_type"],
                "confidence": score,
                "source_episode": m["episode_id"],
                "similarity": m["similarity"],
                "outcome_score": m["outcome_score"],
            })
        actions.sort(key=lambda x: x["confidence"], reverse=True)

        trace = []
        for cond in conditions:
            node = _normalize_condition(cond)
            total_concept_out = sum(
                w for _, w in self._concept_out.get(node, [])
            ) or 1
            for target, weight in self._concept_out.get(node, []):
                trace.append({
                    "from": node,
                    "to": target,
                    "type": "CONCEPT",
                    "confidence": round(weight / total_concept_out, 3),
                })

        for m in matched[:5]:
            ep = m["episode_id"]
            for target, gap in self._preceded_out.get(ep, [])[:2]:
                t_type = self.graph.nodes[target].get("event_type")
                if t_type:
                    trace.append({
                        "from": f"{m['event_type']} ({ep})",
                        "to": t_type,
                        "type": "PRECEDED",
                        "time_gap_hours": round(gap, 2),
                        "confidence": round(m["similarity"], 3),
                    })

        trace = trace[:10]
        
        state = {
                "episode_hits": dict(self.episode_hits),
                "global_counter": self.global_counter,
                "importance": dict(self.importance),
                "pruned_episodes": list(self.pruned_episodes),
        }
        with open(os.path.join(self.dataset_dir, "learning_state.json"), "w") as f:
            json.dump(state, f)

        if self.global_counter % 100 == 0:
            for ep_id in self.episode_hits:
                if self.episode_hits[ep_id] == 0:
                    self.pruned_episodes.add(ep_id)

        return {
            "situation": situation,
            "query_text": query_text,
            "risk_scores": risk_scores[:5],
            "actions": actions[:5],
            "trace": trace,
            "top_matches": [
                {
                    "episode_id": m["episode_id"],
                    "event_type": m["event_type"],
                    "similarity": m["similarity"],
                    "intervention": m["intervention"],
                }
                for m in matched[:5]
            ],
        }


if __name__ == "__main__":
    engine = CivicMindInference(k_matches=20)

    sample_situation = {
        "timestamp": "2026-06-17T07:42:00",
        "ward": "Transit Hub",
        "active_conditions": ["heavy_rain", "road_construction"],
        "weather_forecast": "rain_6h",
        "event_name": "City Music Festival",
        "expected_crowd": 12000,
    }

    result = engine.predict(sample_situation)

    print("=" * 60)
    print("  CIVICMIND INFERENCE RESULT")
    print("=" * 60)
    print(f"\nQuery: {result['query_text'].strip()}")
    print(f"\n--- Top Matches ---")
    for m in result["top_matches"]:
        print(f"  {m['episode_id']:>8} | {m['event_type']:<20} | "
              f"sim={m['similarity']:.3f} | {m['intervention']}")

    print(f"\n--- Risk Scores ---")
    for rs in result["risk_scores"]:
        print(f"  {rs['event']:<22} combined={rs['combined_score']:.3f}  "
              f"(retrieval={rs['retrieval_score']:.3f} "
              f"transition={rs['transition_score']:.3f} "
              f"concept={rs['concept_score']:.3f})")

    print(f"\n--- Recommended Actions ---")
    for a in result["actions"]:
        print(f"  [{a['confidence']:.3f}] {a['action']:<30} "
              f"← {a['source_episode']} ({a['target_event']})")

    print(f"\n--- Reasoning Trace ---")
    for t in result["trace"]:
        if t["type"] == "CONCEPT":
            print(f"  {t['from']} ──({t['confidence']})──▶ {t['to']}")
        else:
            print(f"  {t['from']} ──({t['confidence']}, {t['time_gap_hours']}h)──▶ {t['to']}")
