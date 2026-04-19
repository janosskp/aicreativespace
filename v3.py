import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 🧠 1. ESG KNOWLEDGE BASE
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "ESRS E1-5",
        "description": "electricity energy kwh power consumption usage grid facility"
    },

    "ghg_scope1": {
        "esrs": "ESRS E1-6",
        "description": "direct emissions fuel combustion diesel gas company vehicles boilers onsite combustion co2 scope1"
    },

    "ghg_scope2": {
        "esrs": "ESRS E1-6",
        "description": "indirect emissions purchased electricity grid energy consumption upstream energy co2 scope2"
    },

    "water_usage": {
        "esrs": "ESRS E3-4",
        "description": "water consumption liters usage cooling facility water withdrawal water usage"
    },

    "waste": {
        "esrs": "ESRS E5-3",
        "description": "waste disposal landfill recycling kg production waste materials wastekg"
    }
}


# =========================================================
# 🧠 2. INTELLIGENCE ENGINE
# =========================================================
class ESGContextIntelligenceV3:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            k: self.model.encode(v["description"])
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata_fields = {"timestamp", "date", "location", "site", "id"}

    # =====================================================
    # 🧹 3. DATA CLEANING (CRITICAL FIX)
    # =====================================================
    def clean_columns(self, df):

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        return df

    # =====================================================
    # 🧠 4. METADATA FILTER
    # =====================================================
    def is_metadata(self, column_name: str):

        col = column_name.lower()
        return any(meta in col for meta in self.metadata_fields)

    # =====================================================
    # 🧠 5. SCOPE DETECTION
    # =====================================================
    def detect_scope_hint(self, column_name: str):

        col = column_name.lower()

        if "scope1" in col or "fuel" in col or "combustion" in col:
            return "ghg_scope1"

        if "scope2" in col or "electricity" in col or "grid" in col:
            return "ghg_scope2"

        return None

    # =====================================================
    # 🧠 6. MAIN MAPPING LOGIC
    # =====================================================
    def map_column(self, column_name: str):

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

        # scope override
        scope_hint = self.detect_scope_hint(column_name)

        if scope_hint:
            best_match = scope_hint
            confidence = max(confidence, 0.75)

        # low confidence
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

        return {
            "column": column_name,
            "concept": best_match,
            "esrs": ESG_CONCEPTS[best_match]["esrs"],
            "confidence": confidence,
            "status": "mapped",
            "reason": f"Mapped to {best_match} via semantic similarity",
            "scores": scores
        }

    # =====================================================
    # 🧠 7. BATCH PROCESSING
    # =====================================================
    def process_dataframe(self, df):

        results = []

        for col in df.columns:
            results.append(self.map_column(col))

        return results


# =========================================================
# 🚀 8. MAIN PIPELINE (FIXED + ROBUST)
# =========================================================
if __name__ == "__main__":

    # 📊 LOAD CSV
    df = pd.read_csv("testdata.csv")

    # 🧹 CLEAN DATA (IMPORTANT FIX)
    engine = ESGContextIntelligenceV3()
    df = engine.clean_columns(df)

    print("\n📊 CLEANED INPUT COLUMNS:")
    print(df.columns.tolist())

    print("\n📊 INPUT DATA SAMPLE:")
    print(df.head())

    # 🧠 RUN AI ENGINE
    results = engine.process_dataframe(df)

    # 📁 TO DATAFRAME
    results_df = pd.DataFrame(results)

    # optional cleanup
    if "scores" in results_df.columns:
        results_df = results_df.drop(columns=["scores"])

    # 💾 SAVE OUTPUT
    results_df.to_csv("esg_mapping_results.csv", index=False)

    # 🖨️ OUTPUT
    print("\n=== ESG MAPPING RESULTS ===\n")
    print(results_df)