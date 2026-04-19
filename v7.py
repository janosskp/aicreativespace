import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast


# =========================================================
# 🧠 1. ESG ONTOLOGY (expanded ESRS coverage)
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "E1",
        "description": "electricity energy kwh power consumption usage grid facility"
    },
    "ghg_scope1": {
        "esrs": "E1",
        "description": "direct emissions fuel combustion diesel gas onsite boilers vehicles"
    },
    "ghg_scope2": {
        "esrs": "E1",
        "description": "indirect emissions purchased electricity grid energy upstream emissions"
    },
    "water_usage": {
        "esrs": "E3",
        "description": "water consumption liters usage cooling withdrawal facility water"
    },
    "waste": {
        "esrs": "E5",
        "description": "waste disposal landfill recycling kg production waste materials"
    },
    "fuel_consumption": {
        "esrs": "E1",
        "description": "gas fuel diesel natural gas consumption heating combustion fuel usage"
    }
}


# =========================================================
# 🧠 2. INTELLIGENCE ENGINE V7
# =========================================================
class ESGContextIntelligenceV7:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # embeddings cache
        self.embeddings = {
            k: self.model.encode(v["description"])
            for k, v in ESG_CONCEPTS.items()
        }

        # metadata detection
        self.metadata_fields = {
            "timestamp", "date", "time", "site", "location", "id"
        }

        # confidence thresholds (calibrated)
        self.threshold_low = 0.40
        self.threshold_high = 0.70


    # =====================================================
    # 🧠 3. METADATA FILTER
    # =====================================================
    def is_metadata(self, col: str):
        return any(m in col.lower() for m in self.metadata_fields)


    # =====================================================
    # 🧠 4. RULE ENGINE (V6 preserved)
    # =====================================================
    def rule_engine(self, col: str):

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
    # 🧠 5. SEMANTIC SCORING
    # =====================================================
    def semantic_match(self, col: str):

        emb = self.model.encode(col.lower())

        scores = {}

        for concept, vec in self.embeddings.items():
            scores[concept] = float(cosine_similarity([emb], [vec])[0][0])

        best = max(scores.items(), key=lambda x: x[1])

        return best[0], best[1], scores


    # =====================================================
    # 🧠 6. EXPLAINABILITY ENGINE (V7 upgraded)
    # =====================================================
    def explain(self, col, concept, rule, semantic_score, scores):

        explanation = {
            "column": col,
            "decision_path": {
                "rule_trigger": rule if rule else "none",
                "semantic_best_match": concept,
                "semantic_score": round(semantic_score, 3)
            },
            "top_alternatives": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3],
            "decision": f"{concept} selected based on hybrid rule + semantic scoring"
        }

        return explanation


    # =====================================================
    # 🧠 7. MAIN MAPPING LOGIC
    # =====================================================
    def map_column(self, col: str):

        # STEP 1: metadata filter
        if self.is_metadata(col):
            return {
                "column": col,
                "concept": "metadata",
                "esrs": None,
                "confidence": 0.0,
                "status": "excluded_metadata",
                "reason": "metadata detected",
                "scores": {}
            }

        # STEP 2: rule engine (V6 preserved)
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

        # STEP 5: low confidence handling
        if confidence < self.threshold_low:
            return {
                "column": col,
                "concept": "unknown",
                "esrs": "UNKNOWN",
                "confidence": confidence,
                "status": "low_confidence",
                "reason": "no strong match",
                "scores": scores
            }

        # STEP 6: ESRS mapping
        esrs = ESG_CONCEPTS.get(final, {}).get("esrs", "UNKNOWN")

        # STEP 7: explainability
        explanation = self.explain(col, final, rule_name, sem_score, scores)

        return {
            "column": col,
            "concept": final,
            "esrs": esrs,
            "confidence": float(confidence),
            "status": "mapped",
            "reason": explanation,
            "method": method,
            "scores": scores
        }


    # =====================================================
    # 🧠 8. MISSING DISCLOSURE CHECK (NEW V7)
    # =====================================================
    def missing_esrs_check(self, mapped_results):

        required = {"E1", "E3", "E5"}

        present = {r.get("esrs") for r in mapped_results if r.get("esrs")}

        missing = required - present

        return {
            "present_esrs": list(present),
            "missing_esrs": list(missing),
            "completeness_score": len(present) / len(required)
        }


    # =====================================================
    # 🧠 9. BATCH PROCESSING
    # =====================================================
    def process_dataframe(self, df):

        results = []

        for col in df.columns:
            results.append(self.map_column(col))

        return results


# =========================================================
# 🚀 10. RUN WITH CSV
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    print("\n📊 INPUT DATA")
    print(df.head())

    engine = ESGContextIntelligenceV7()

    results = engine.process_dataframe(df)

    results_df = pd.DataFrame(results)

    if "scores" in results_df.columns:
        results_df["scores"] = results_df["scores"].apply(lambda x: str(x))

    results_df.to_csv("esg_mapping_v7.csv", index=False)

    print("\n=== ESG MAPPING V7 ===\n")
    print(results_df)

    # 🧠 ESRS coverage check
    coverage = engine.missing_esrs_check(results)

    print("\n=== ESRS COVERAGE ===\n")
    print(coverage)