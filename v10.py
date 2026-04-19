import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 🧠 1. ESG KNOWLEDGE BASE (UNCHANGED CORE)
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
# 🧠 2. UNIT NORMALIZATION (FIX #3 ENHANCED)
# =========================================================
class UnitNormalizer:

    def normalize_value(self, column, value):

        col = column.lower()

        # energy
        if "mwh" in col:
            return value * 1000

        if "mj" in col:
            return value * 0.277

        # CO2
        if "tco2" in col:
            return value * 1000

        # gas heuristics (optional future)
        if "m3" in col:
            return value

        return value


# =========================================================
# 🧠 3. UNCERTAINTY CALIBRATION (FIX #4 REALISTIC)
# =========================================================
class ConfidenceCalibrator:

    def calibrate(self, rule_conf, semantic_conf):

        # FIX: no fake 0.95 inflation anymore

        base = max(rule_conf, semantic_conf)

        # nonlinear compression (reduces overconfidence)
        calibrated = np.tanh(2 * (base - 0.5)) * 0.5 + 0.5

        # ensure realistic bounds
        return float(np.clip(calibrated, 0.05, 0.95))


# =========================================================
# 🧠 4. CONFLICT RESOLVER (FIX #1 IMPROVED)
# =========================================================
class ConflictResolver:

    def resolve(self, rule_match, sem_match, rule_conf, sem_conf):

        # CASE 1: strong rule + weak semantic
        if rule_match and rule_conf > sem_conf + 0.2:
            return rule_match, "rule_dominant"

        # CASE 2: strong semantic override
        if sem_conf > rule_conf + 0.15:
            return sem_match, "semantic_dominant"

        # CASE 3: conflict detected
        if rule_match != sem_match:
            return sem_match or rule_match, "conflict_resolved"

        return rule_match or sem_match, "aligned"


# =========================================================
# 🧠 5. MAIN ENGINE v10
# =========================================================
class ESGContextIntelligenceV10:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            k: self.model.encode(v["description"])
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata_fields = {"timestamp", "date", "location", "site", "id"}

        # NEW LAYERS
        self.normalizer = UnitNormalizer()
        self.calibrator = ConfidenceCalibrator()
        self.resolver = ConflictResolver()

        # CONTEXT MEMORY (FIX #5)
        self.memory = {
            "columns_seen": {},
            "site_context": None
        }

    # =====================================================
    # METADATA FILTER
    # =====================================================
    def is_metadata(self, col):
        return col.lower() in self.metadata_fields

    # =====================================================
    # RULE ENGINE (v9 preserved)
    # =====================================================
    def rule_engine(self, column):

        col = column.lower()

        if "scope1" in col:
            return "ghg_scope1", 0.85

        if "scope2" in col:
            return "ghg_scope2", 0.85

        if "kwh" in col or "energy" in col:
            return "energy_consumption", 0.8

        if "water" in col:
            return "water_usage", 0.8

        if "waste" in col:
            return "waste", 0.8

        if "gas" in col or "fuel" in col:
            return "fuel_consumption", 0.8

        return None, 0.0

    # =====================================================
    # SEMANTIC ENGINE (v9 preserved)
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
    # FIX #5 CONTEXT MEMORY UPDATE
    # =====================================================
    def update_memory(self, column, concept):

        self.memory["columns_seen"][column] = concept

    # =====================================================
    # MAIN MAPPING LOGIC (v9 + FIXES)
    # =====================================================
    def map_column(self, column):

        # -------------------------
        # 1. Metadata filter
        # -------------------------
        if self.is_metadata(column):
            return {
                "column": column,
                "concept": "metadata",
                "esrs": None,
                "confidence": 0.0,
                "status": "excluded_metadata",
                "reason": "metadata detected",
                "scores": {},
                "method": "metadata_filter"
            }

        # -------------------------
        # 2. Rule + Semantic
        # -------------------------
        rule_match, rule_conf = self.rule_engine(column)
        sem_match, sem_conf, scores = self.semantic_engine(column)

        # -------------------------
        # 3. Conflict Resolver (FIX #1)
        # -------------------------
        final_match, decision_mode = self.resolver.resolve(
            rule_match,
            sem_match,
            rule_conf,
            sem_conf
        )

        # -------------------------
        # 4. Confidence Calibration (FIX #4)
        # -------------------------
        confidence = self.calibrator.calibrate(rule_conf, sem_conf)

        # -------------------------
        # 5. ESRS Mapping (FIX #2 ambiguity ready)
        # -------------------------
        esrs = ESG_CONCEPTS.get(final_match, {}).get("esrs", "UNKNOWN")

        # -------------------------
        # 6. Memory Update (FIX #5)
        # -------------------------
        self.update_memory(column, final_match)

        # -------------------------
        # 7. Explainability
        # -------------------------
        reason = {
            "decision": final_match,
            "rule": rule_match,
            "semantic_best": sem_match,
            "mode": decision_mode
        }

        return {
            "column": column,
            "concept": final_match,
            "esrs": esrs,
            "confidence": confidence,
            "status": "mapped" if confidence > 0.4 else "low_confidence",
            "reason": str(reason),
            "scores": scores,
            "method": decision_mode
        }

    # =====================================================
    # BATCH PROCESSING
    # =====================================================
    def process_dataframe(self, df):

        return [self.map_column(c) for c in df.columns]


# =========================================================
# 🚀 RUN
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    engine = ESGContextIntelligenceV10()

    results = engine.process_dataframe(df)

    out = pd.DataFrame(results)

    out.to_csv("esg_mapping_v10.csv", index=False)

    print("\n=== ESG MAPPING V10 ===\n")
    print(out)