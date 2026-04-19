import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 🧠 1. ESG KNOWLEDGE BASE (ENHANCED)
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "ESRS E1-5",
        "description": "electricity energy kwh power consumption usage grid facility elec energy"
    },

    "ghg_scope1": {
        "esrs": "ESRS E1-6",
        "description": "direct emissions fuel combustion diesel gas boilers onsite co2 scope1"
    },

    "ghg_scope2": {
        "esrs": "ESRS E1-6",
        "description": "indirect emissions purchased electricity grid energy co2 scope2 upstream"
    },

    "water_usage": {
        "esrs": "ESRS E3-4",
        "description": "water consumption liters usage cooling withdrawal water usage liters"
    },

    "waste": {
        "esrs": "ESRS E5-3",
        "description": "waste disposal landfill recycling kg waste production materials wastekg"
    }
}


# =========================================================
# 🧠 2. INTELLIGENCE ENGINE V4
# =========================================================
class ESGContextIntelligenceV4:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            k: self.model.encode(v["description"])
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata_fields = {"timestamp", "date", "location", "site"}

    # =====================================================
    # 🧹 CLEAN DATA
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
    # 🧠 METADATA FILTER
    # =====================================================
    def is_metadata(self, col):
        return any(m in col.lower() for m in self.metadata_fields)

    # =====================================================
    # ⚡ VALUE TYPE DETECTION (NEW IN V4)
    # =====================================================
    def detect_unit_hint(self, col):

        col = col.lower()

        if "kwh" in col or "elec" in col:
            return "energy"

        if "co2" in col or "emission" in col:
            return "emissions"

        if "water" in col:
            return "water"

        if "waste" in col or "kg" in col:
            return "waste"

        if "gas" in col:
            return "emissions"

        return None

    # =====================================================
    # 🧠 SCOPE LOGIC (IMPROVED)
    # =====================================================
    def detect_scope_hint(self, col):

        col = col.lower()

        if "scope1" in col:
            return "ghg_scope1"

        if "scope2" in col:
            return "ghg_scope2"

        # weak inference for emissions
        if "co2" in col and "electric" in col:
            return "ghg_scope2"

        if "co2" in col and "fuel" in col:
            return "ghg_scope1"

        return None

    # =====================================================
    # 🧠 MAIN MAPPING ENGINE
    # =====================================================
    def map_column(self, column_name):

        if self.is_metadata(column_name):
            return {
                "column": column_name,
                "concept": "metadata",
                "esrs": None,
                "confidence": 0.0,
                "status": "excluded_metadata",
                "reason": "Metadata field filtered"
            }

        col_embedding = self.model.encode(column_name)

        best_match = None
        best_score = -1
        scores = {}

        for concept, data in ESG_CONCEPTS.items():

            score = cosine_similarity(
                [col_embedding],
                [self.embeddings[concept]]
            )[0][0]

            scores[concept] = float(score)

            if score > best_score:
                best_score = score
                best_match = concept

        confidence = float(best_score)

        # =====================================================
        # 🧠 CONTEXT BOOST (NEW)
        # =====================================================
        unit_hint = self.detect_unit_hint(column_name)
        scope_hint = self.detect_scope_hint(column_name)

        if scope_hint:
            best_match = scope_hint
            confidence = max(confidence, 0.8)

        # =====================================================
        # ⚠️ AMBIGUITY DETECTION (NEW)
        # =====================================================
        ambiguity_flags = []

        if "co2" in column_name.lower() and not scope_hint:
            ambiguity_flags.append("CO2 without scope (uncertain attribution)")

        if unit_hint == "energy" and best_match != "energy_consumption":
            ambiguity_flags.append("Energy unit mismatch risk")

        # =====================================================
        # ❌ LOW CONFIDENCE
        # =====================================================
        if confidence < 0.45:
            return {
                "column": column_name,
                "concept": "unknown",
                "esrs": "UNKNOWN",
                "confidence": confidence,
                "status": "low_confidence",
                "reason": "No strong semantic match",
                "flags": ambiguity_flags,
                "scores": scores
            }

        # =====================================================
        # ✅ FINAL OUTPUT
        # =====================================================
        return {
            "column": column_name,
            "concept": best_match,
            "esrs": ESG_CONCEPTS[best_match]["esrs"],
            "confidence": confidence,
            "status": "mapped",
            "reason": f"Mapped via semantic + context + unit inference",
            "flags": ambiguity_flags,
            "scores": scores
        }

    # =====================================================
    # 🧠 BATCH PROCESSING
    # =====================================================
    def process_dataframe(self, df):

        results = []

        for col in df.columns:
            results.append(self.map_column(col))

        return results

    # =====================================================
    # 📊 COMPLIANCE CHECK (NEW IN V4)
    # =====================================================
    def compliance_check(self, results):

        required = {"energy_consumption", "ghg_scope1", "ghg_scope2", "water_usage", "waste"}

        mapped = {r["concept"] for r in results if r["concept"] in required}

        missing = required - mapped

        return {
            "compliance_score": len(mapped) / len(required),
            "missing_fields": list(missing)
        }


# =========================================================
# 🚀 3. RUN WITH YOUR TEST DATA
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    engine = ESGContextIntelligenceV4()

    df = engine.clean_columns(df)

    print("\n📊 CLEANED DATA COLUMNS:")
    print(df.columns.tolist())

    results = engine.process_dataframe(df)

    results_df = pd.DataFrame(results)

    compliance = engine.compliance_check(results)

    print("\n=== ESG MAPPING V4 RESULTS ===\n")
    print(results_df)

    print("\n=== COMPLIANCE CHECK ===\n")
    print(compliance)

    results_df.to_csv("esg_v4_results.csv", index=False)