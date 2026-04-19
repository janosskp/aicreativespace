import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 🧠 1. ESRS KNOWLEDGE GRAPH (V8 enhanced granularity)
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "E1-5",
        "description": "electricity energy kwh power consumption usage grid facility"
    },
    "ghg_scope1": {
        "esrs": "E1-6",
        "description": "direct emissions fuel combustion diesel gas onsite boilers vehicles"
    },
    "ghg_scope2": {
        "esrs": "E1-6",
        "description": "indirect emissions purchased electricity grid energy upstream emissions"
    },
    "water_usage": {
        "esrs": "E3-4",
        "description": "water consumption liters usage cooling withdrawal facility water"
    },
    "waste": {
        "esrs": "E5-3",
        "description": "waste disposal landfill recycling kg production waste materials"
    },
    "fuel_consumption": {
        "esrs": "E1-5",
        "description": "gas fuel diesel natural gas consumption heating combustion fuel usage"
    }
}


# =========================================================
# 🧠 2. ENGINE V8
# =========================================================
class ESGContextIntelligenceV8:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            k: self.model.encode(v["description"])
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata_fields = {"timestamp", "date", "site", "location", "id"}

        self.threshold_low = 0.40
        self.threshold_high = 0.70


    # =====================================================
    # 🧠 3. METADATA FILTER (unchanged V7)
    # =====================================================
    def is_metadata(self, col):
        return any(m in col.lower() for m in self.metadata_fields)


    # =====================================================
    # 🧠 4. RULE ENGINE (unchanged V7 core)
    # =====================================================
    def rule_engine(self, col):

        c = col.lower()

        if "scope1" in c:
            return "ghg_scope1", 0.85, "hard_rule_scope1"

        if "scope2" in c:
            return "ghg_scope2", 0.85, "hard_rule_scope2"

        if "co2" in c and "scope1" not in c and "scope2" not in c:
            return "ghg_scope2", 0.65, "co2_default_scope2_rule"

        if "kwh" in c or "energy" in c:
            return "energy_consumption", 0.65, "energy_unit_rule"

        if "water" in c:
            return "water_usage", 0.65, "water_keyword_rule"

        if "waste" in c:
            return "waste", 0.65, "waste_keyword_rule"

        if "gas" in c or "fuel" in c:
            return "fuel_consumption", 0.60, "fuel_rule"

        return None, 0.0, None


    # =====================================================
    # 🧠 5. SEMANTIC ENGINE
    # =====================================================
    def semantic_match(self, col):

        emb = self.model.encode(col.lower())

        scores = {
            k: float(cosine_similarity([emb], [v])[0][0])
            for k, v in self.embeddings.items()
        }

        best = max(scores.items(), key=lambda x: x[1])

        return best[0], best[1], scores


    # =====================================================
    # 🧠 6. NARRATIVE EXPLAINABILITY (FIXED PROBLEM 1)
    # =====================================================
    def explain_narrative(self, col, concept, rule, score):

        if rule:
            return (
                f"The column '{col}' was classified as '{concept}' "
                f"because a deterministic rule ('{rule}') was triggered, "
                f"which takes priority over semantic similarity."
            )

        return (
            f"The column '{col}' was classified as '{concept}' "
            f"based on semantic similarity with a score of {round(score,3)} "
            f"against ESG ontology definitions."
        )


    # =====================================================
    # 🧠 7. BLIND SPOT DETECTION (FIXED PROBLEM 3)
    # =====================================================
    def detect_non_esg_signal(self, col):

        non_esg_keywords = [
            "production", "output", "revenue", "sales", "units"
        ]

        if any(k in col.lower() for k in non_esg_keywords):
            return True

        return False


    # =====================================================
    # 🧠 8. MAIN MAPPING LOGIC
    # =====================================================
    def map_column(self, col):

        # STEP 1: metadata
        if self.is_metadata(col):
            return {
                "column": col,
                "concept": "metadata",
                "esrs": None,
                "confidence": 0.0,
                "status": "excluded_metadata",
                "reason": "Metadata field filtered",
                "scores": {}
            }

        # STEP 2: rule engine
        rule_concept, rule_conf, rule_name = self.rule_engine(col)

        # STEP 3: semantic engine
        sem_concept, sem_score, scores = self.semantic_match(col)

        # STEP 4: hybrid decision
        if rule_conf > sem_score:
            final = rule_concept
            confidence = rule_conf
            method = f"rule_override:{rule_name}"
        else:
            final = sem_concept
            confidence = sem_score
            method = "semantic_primary"

        # STEP 5: low confidence
        if confidence < self.threshold_low:
            return {
                "column": col,
                "concept": "unknown",
                "esrs": "UNKNOWN",
                "confidence": confidence,
                "status": "low_confidence",
                "reason": "No strong match",
                "scores": scores
            }

        # STEP 6: ESRS mapping (FIXED PROBLEM 2)
        esrs = ESG_CONCEPTS.get(final, {}).get("esrs", "UNKNOWN")

        # STEP 7: blind spot detection (FIXED PROBLEM 3)
        if self.detect_non_esg_signal(col):
            method += "|non_esg_signal_detected"

        # STEP 8: narrative explanation (FIXED PROBLEM 1)
        reason = self.explain_narrative(col, final, rule_name, sem_score)

        return {
            "column": col,
            "concept": final,
            "esrs": esrs,
            "confidence": float(confidence),
            "status": "mapped",
            "reason": reason,
            "scores": scores,
            "method": method
        }


    # =====================================================
    # 🧠 9. BATCH PROCESSING
    # =====================================================
    def process_dataframe(self, df):
        return [self.map_column(c) for c in df.columns]


# =========================================================
# 🚀 10. RUN
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    engine = ESGContextIntelligenceV8()

    results = engine.process_dataframe(df)

    results_df = pd.DataFrame(results)

    if "scores" in results_df.columns:
        results_df["scores"] = results_df["scores"].apply(lambda x: str(x))

    results_df.to_csv("esg_mapping_v8.csv", index=False)

    print("\n=== V8 RESULTS ===\n")
    print(results_df)