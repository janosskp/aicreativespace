import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 🧠 1. ESG KNOWLEDGE BASE (enhanced with context)
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "ESRS E1-5",
        "description": "electricity energy kwh power consumption usage grid facility"
    },

    "ghg_scope1": {
        "esrs": "ESRS E1-6",
        "description": "direct emissions fuel combustion diesel gas company vehicles boilers onsite combustion"
    },

    "ghg_scope2": {
        "esrs": "ESRS E1-6",
        "description": "indirect emissions purchased electricity grid energy consumption upstream energy"
    },

    "water_usage": {
        "esrs": "ESRS E3-4",
        "description": "water consumption liters usage cooling facility water withdrawal"
    },

    "waste": {
        "esrs": "ESRS E5-3",
        "description": "waste disposal landfill recycling kg production waste materials"
    }
}


# =========================================================
# 🧠 2. INTELLIGENCE ENGINE
# =========================================================
class ESGContextIntelligenceV2:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # precompute embeddings
        self.embeddings = {
            k: self.model.encode(v["description"])
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata_fields = {"timestamp", "date", "location", "site", "id"}

    # =====================================================
    # 🧠 3. METADATA FILTER (KEY IMPROVEMENT)
    # =====================================================
    def is_metadata(self, column_name: str):

        col = column_name.lower()

        return any(meta in col for meta in self.metadata_fields)

    # =====================================================
    # 🧠 4. SCOPE DETECTOR (IMPORTANT UPGRADE)
    # =====================================================
    def detect_scope_hint(self, column_name: str):

        col = column_name.lower()

        if "scope1" in col or "fuel" in col or "combustion" in col:
            return "ghg_scope1"

        if "scope2" in col or "electricity" in col or "grid" in col:
            return "ghg_scope2"

        return None

    # =====================================================
    # 🧠 5. MAIN MAPPING LOGIC
    # =====================================================
    def map_column(self, column_name: str):

        # ❌ STEP 1: FILTER METADATA
        if self.is_metadata(column_name):
            return {
                "column": column_name,
                "concept": "metadata",
                "esrs": None,
                "confidence": 0.0,
                "status": "excluded_metadata",
                "reason": "Detected as metadata field"
            }

        col_embedding = self.model.encode(column_name.lower())

        best_match = None
        best_score = -1
        scores = {}

        # =====================================================
        # STEP 2: SEMANTIC MATCHING
        # =====================================================
        for concept, data in ESG_CONCEPTS.items():

            emb = self.embeddings[concept]

            score = cosine_similarity(
                [col_embedding],
                [emb]
            )[0][0]

            scores[concept] = float(score)

            if score > best_score:
                best_score = score
                best_match = concept

        confidence = float(best_score)

        # =====================================================
        # STEP 3: SCOPE OVERRIDE LOGIC (CRITICAL UPGRADE)
        # =====================================================
        scope_hint = self.detect_scope_hint(column_name)

        if scope_hint:
            best_match = scope_hint
            confidence = max(confidence, 0.75)  # boost confidence

        # =====================================================
        # STEP 4: LOW CONFIDENCE HANDLING
        # =====================================================
        if confidence < 0.45:
            return {
                "column": column_name,
                "concept": "unknown",
                "esrs": "UNKNOWN",
                "confidence": confidence,
                "status": "low_confidence",
                "reason": "No strong semantic match",
                "scores": scores
            }

        # =====================================================
        # STEP 5: FINAL OUTPUT
        # =====================================================
        return {
            "column": column_name,
            "concept": best_match,
            "esrs": ESG_CONCEPTS[best_match]["esrs"],
            "confidence": confidence,
            "status": "mapped",
            "reason": self.explain(column_name, best_match, scores),
            "scores": scores
        }

    # =====================================================
    # 🧠 6. EXPLAINABILITY LAYER (VERY IMPORTANT)
    # =====================================================
    def explain(self, column, concept, scores):

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        explanation = (
            f"Column '{column}' was mapped to '{concept}' "
            f"because it had highest semantic similarity "
            f"({top[0][1]:.3f}) compared to other ESG categories."
        )

        return explanation

    # =====================================================
    # 🧠 7. BATCH PROCESSING
    # =====================================================
    def process_dataframe(self, df):

        results = []

        for col in df.columns:
            results.append(self.map_column(col))

        return results


# =========================================================
# 🚀 8. RUN EXAMPLE
# =========================================================
if __name__ == "__main__":

    df = pd.DataFrame({
        "timestamp": [],
        "location": [],
        "energy_kwh": [],
        "scope1_emissions_kgco2e": [],
        "water_liters": [],
        "waste_kg": []
    })

    engine = ESGContextIntelligenceV2()

    results = engine.process_dataframe(df)

    for r in results:
        print(r)