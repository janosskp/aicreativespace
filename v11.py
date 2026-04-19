import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import json

# =========================================================
# 🧠 ESRS KNOWLEDGE BASE
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "E1-5",
        "description": "electricity energy kwh power consumption grid facility usage"
    },
    "ghg_scope1": {
        "esrs": "E1-6",
        "description": "direct emissions fuel combustion diesel gas onsite boilers vehicles"
    },
    "ghg_scope2": {
        "esrs": "E1-6",
        "description": "indirect emissions purchased electricity grid upstream energy"
    },
    "water_usage": {
        "esrs": "E3-4",
        "description": "water consumption liters withdrawal cooling facility usage"
    },
    "waste": {
        "esrs": "E5-3",
        "description": "waste disposal landfill recycling production waste materials"
    },
    "fuel_consumption": {
        "esrs": "E1-5",
        "description": "gas fuel combustion heating fossil fuels consumption"
    }
}

# =========================================================
# 🧠 AUDIT STRUCTURES
# =========================================================
@dataclass
class DecisionNode:
    column: str
    rule_result: str
    semantic_result: str
    rule_score: float
    semantic_scores: dict
    final_decision: str
    decision_mode: str
    conflict: bool
    esrs: str
    confidence: float
    reasoning_chain: list


# =========================================================
# 🧠 ENGINE
# =========================================================
class ESRSAuditEngineV11:

    def __init__(self, threshold=0.45):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

        self.embeddings = {
            k: self.model.encode(v["description"])
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata = {"timestamp", "date", "site", "location", "id"}

    # =====================================================
    # 🧠 METADATA FILTER
    # =====================================================
    def is_metadata(self, col):
        return col.lower() in self.metadata

    # =====================================================
    # 🧠 RULE ENGINE
    # =====================================================
    def rule_engine(self, col):
        c = col.lower()

        if "kwh" in c or "energy" == c:
            return "energy_consumption", "energy_rule", 0.95

        if "scope1" in c:
            return "ghg_scope1", "scope1_rule", 0.95

        if "scope2" in c:
            return "ghg_scope2", "scope2_rule", 0.9

        if "co2" in c:
            return "ghg_scope2", "co2_default_rule", 0.75

        if "water" in c:
            return "water_usage", "water_rule", 0.9

        if "waste" in c:
            return "waste", "waste_rule", 0.9

        if "gas" in c or "fuel" in c:
            return "fuel_consumption", "fuel_rule", 0.85

        return None, None, 0.0

    # =====================================================
    # 🧠 SEMANTIC ENGINE
    # =====================================================
    def semantic_engine(self, col):
        emb = self.model.encode(col.lower())

        scores = {}
        for k, v in self.embeddings.items():
            scores[k] = float(cosine_similarity([emb], [v])[0][0])

        best = max(scores, key=scores.get)
        return best, scores, scores[best]

    # =====================================================
    # 🧠 DECISION GRAPH ENGINE
    # =====================================================
    def decide(self, col):

        # =========================
        # 1. METADATA
        # =========================
        if self.is_metadata(col):
            return DecisionNode(
                column=col,
                rule_result="metadata_filter",
                semantic_result="none",
                rule_score=1.0,
                semantic_scores={},
                final_decision="metadata",
                decision_mode="excluded",
                conflict=False,
                esrs=None,
                confidence=0.0,
                reasoning_chain=["Filtered as metadata field"]
            ).__dict__

        # =========================
        # 2. RULE + SEMANTIC
        # =========================
        rule_concept, rule_name, rule_score = self.rule_engine(col)
        sem_best, sem_scores, sem_conf = self.semantic_engine(col)

        reasoning = []

        if rule_concept:
            reasoning.append(f"Rule triggered: {rule_name}")
        else:
            reasoning.append("No rule triggered")

        reasoning.append(f"Semantic best match: {sem_best} ({sem_conf:.3f})")

        # =========================
        # 3. CONFLICT DETECTION
        # =========================
        conflict = False
        final = None
        mode = None
        confidence = 0.0

        if rule_concept:

            if rule_concept != sem_best:
                conflict = True
                reasoning.append("Conflict: rule != semantic")

            final = rule_concept
            mode = "rule_dominant"
            confidence = min(0.95, 0.7 + rule_score * 0.25 + sem_conf * 0.1)

        else:

            if sem_conf >= self.threshold:
                final = sem_best
                mode = "semantic_dominant"
                confidence = sem_conf
            else:
                final = "unknown"
                mode = "low_confidence"
                confidence = sem_conf

        esrs = ESG_CONCEPTS.get(final, {}).get("esrs") if final in ESG_CONCEPTS else "UNKNOWN"

        # =========================
        # 4. AUDIT NODE
        # =========================
        return DecisionNode(
            column=col,
            rule_result=rule_concept,
            semantic_result=sem_best,
            rule_score=rule_score,
            semantic_scores=sem_scores,
            final_decision=final,
            decision_mode=mode,
            conflict=conflict,
            esrs=esrs,
            confidence=confidence,
            reasoning_chain=reasoning
        ).__dict__

    # =====================================================
    # 🧠 BATCH PROCESSING
    # =====================================================
    def process(self, df):
        return [self.decide(c) for c in df.columns]


# =========================================================
# 🚀 RUN
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    engine = ESRSAuditEngineV11()
    results = engine.process(df)

    results_df = pd.DataFrame(results)

    results_df.to_csv("esrs_audit_v11.csv", index=False)

    print("\n=== AUDIT GRADE ESRS OUTPUT (v11) ===\n")
    print(results_df)