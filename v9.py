import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 🧠 1. ESG KNOWLEDGE BASE (unchanged from v8 core)
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "E1-5",
        "description": "electricity energy kwh mwh mj power consumption usage grid facility"
    },
    "ghg_scope1": {
        "esrs": "E1-6",
        "description": "direct emissions fuel combustion diesel gas onsite boilers vehicles"
    },
    "ghg_scope2": {
        "esrs": "E1-6",
        "description": "indirect emissions purchased electricity grid energy upstream"
    },
    "water_usage": {
        "esrs": "E3-4",
        "description": "water liters m3 consumption withdrawal cooling facility"
    },
    "waste": {
        "esrs": "E5-3",
        "description": "waste kg tons disposal recycling landfill production waste"
    },
    "fuel_consumption": {
        "esrs": "E1-5",
        "description": "gas diesel fuel consumption combustion energy fuel usage"
    }
}


# =========================================================
# 🧠 2. UNIT NORMALIZATION ENGINE (NEW - PROBLEM 3)
# =========================================================
class UnitNormalizer:

    def normalize_value(self, column_name: str, value: float):

        col = column_name.lower()

        # energy units
        if "mwh" in col:
            return value * 1000, "kwh_normalized"

        if "mj" in col:
            return value * 0.277, "kwh_normalized"

        # CO2 normalization
        if "tco2" in col:
            return value * 1000, "kgco2_normalized"

        return value, "no_conversion"


# =========================================================
# 🧠 3. CONFLICT RESOLVER (NEW - PROBLEM 1)
# =========================================================
class ConflictResolver:

    def resolve(self, rule_match, semantic_match, rule_conf, semantic_conf):

        # Rule strong dominance ONLY if high confidence
        if rule_match and rule_conf > 0.75:
            return rule_match, "rule_dominant"

        # Semantic stronger or equal
        if semantic_conf >= rule_conf:
            return semantic_match, "semantic_dominant"

        # hybrid fallback
        return rule_match or semantic_match, "hybrid_resolution"


# =========================================================
# 🧠 4. UNCERTAINTY ENGINE (NEW - PROBLEM 4)
# =========================================================
class UncertaintyEngine:

    def calibrate(self, score):

        # soft calibration (simple but effective)
        return float(1 / (1 + np.exp(-10 * (score - 0.5))))


# =========================================================
# 🧠 5. MAIN INTELLIGENCE ENGINE (v9)
# =========================================================
class ESGContextIntelligenceV9:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            k: self.model.encode(v["description"])
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata_fields = {"timestamp", "date", "location", "site", "id"}

        # NEW layers
        self.normalizer = UnitNormalizer()
        self.resolver = ConflictResolver()
        self.uncertainty = UncertaintyEngine()

        # CONTEXT MEMORY (PROBLEM 5)
        self.memory = {}

    # =====================================================
    # METADATA FILTER
    # =====================================================
    def is_metadata(self, column):
        return column.lower() in self.metadata_fields

    # =====================================================
    # RULE ENGINE (enhanced v8 logic)
    # =====================================================
    def rule_engine(self, column):

        col = column.lower()

        if "scope1" in col:
            return "ghg_scope1", 0.9, "hard_rule_scope1"

        if "scope2" in col:
            return "ghg_scope2", 0.9, "hard_rule_scope2"

        if "kwh" in col or "energy" in col:
            return "energy_consumption", 0.8, "energy_unit_rule"

        if "water" in col:
            return "water_usage", 0.8, "water_keyword_rule"

        if "waste" in col:
            return "waste", 0.8, "waste_keyword_rule"

        if "gas" in col or "fuel" in col:
            return "fuel_consumption", 0.8, "fuel_rule"

        return None, 0.0, None

    # =====================================================
    # SEMANTIC ENGINE
    # =====================================================
    def semantic_engine(self, column):

        col_emb = self.model.encode(column.lower())

        best, best_score = None, -1
        scores = {}

        for concept, data in ESG_CONCEPTS.items():
            emb = self.embeddings[concept]

            score = cosine_similarity([col_emb], [emb])[0][0]
            scores[concept] = float(score)

            if score > best_score:
                best_score = score
                best = concept

        return best, best_score, scores

    # =====================================================
    # MAIN MAP FUNCTION
    # =====================================================
    def map_column(self, column):

        # 1. metadata filter
        if self.is_metadata(column):
            return {
                "column": column,
                "concept": "metadata",
                "esrs": None,
                "confidence": 0.0,
                "status": "excluded_metadata",
                "reason": "Metadata field filtered",
                "scores": {}
            }

        # 2. rule engine
        rule_match, rule_conf, rule_name = self.rule_engine(column)

        # 3. semantic engine
        sem_match, sem_conf, scores = self.semantic_engine(column)

        # 4. conflict resolution (NEW)
        final_match, decision_path = self.resolver.resolve(
            rule_match,
            sem_match,
            rule_conf,
            sem_conf
        )

        # 5. confidence calibration (NEW)
        confidence = self.uncertainty.calibrate(max(rule_conf, sem_conf))

        # 6. ESRS mapping
        esrs = ESG_CONCEPTS.get(final_match, {}).get("esrs", "UNKNOWN")

        # 7. reasoning
        reason = {
            "decision": final_match,
            "rule": rule_name,
            "semantic_best": sem_match,
            "decision_mode": decision_path
        }

        return {
            "column": column,
            "concept": final_match,
            "esrs": esrs,
            "confidence": confidence,
            "status": "mapped" if confidence > 0.4 else "low_confidence",
            "reason": str(reason),
            "scores": scores,
            "method": decision_path
        }

    # =====================================================
    # BATCH PROCESSING
    # =====================================================
    def process_dataframe(self, df):

        results = []

        for col in df.columns:
            results.append(self.map_column(col))

        return results


# =========================================================
# 🚀 RUN EXAMPLE (same test data compatible)
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    engine = ESGContextIntelligenceV9()

    results = engine.process_dataframe(df)

    results_df = pd.DataFrame(results)

    results_df.to_csv("esg_mapping_v9.csv", index=False)

    print("\n=== ESG MAPPING V9 ===\n")
    print(results_df)