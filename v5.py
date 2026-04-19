import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


# =========================================================
# 🧠 1. ESRS KNOWLEDGE GRAPH (V5 CORE)
# =========================================================
ESRS_GRAPH = {
    "E1": {
        "name": "Climate Change",
        "required": ["energy_consumption", "ghg_scope1", "ghg_scope2"]
    },
    "E3": {
        "name": "Water",
        "required": ["water_usage"]
    },
    "E5": {
        "name": "Resource & Waste",
        "required": ["waste"]
    }
}


# =========================================================
# 🧠 2. ESG ONTOLOGY (IMPROVED)
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": "electricity energy kwh power consumption usage grid",
    "ghg_scope1": "direct emissions fuel diesel gas combustion onsite boilers scope1",
    "ghg_scope2": "indirect emissions electricity grid purchased energy scope2",
    "water_usage": "water liters consumption withdrawal cooling facility",
    "waste": "waste kg disposal recycling landfill material waste"
}


# =========================================================
# 🧠 3. HYBRID INTELLIGENCE ENGINE
# =========================================================
class ESGIntelligenceV5Hybrid:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            k: self.model.encode(v)
            for k, v in ESG_CONCEPTS.items()
        }

        # V4 FEATURE: metadata detection
        self.metadata_keywords = [
            "timestamp", "date", "time", "site", "location", "id"
        ]

    # =====================================================
    # 🧹 CLEAN COLUMN (V4 + V5)
    # =====================================================
    def clean(self, col):
        col = col.lower().strip()
        col = re.sub(r"\s+", "_", col)
        return col

    # =====================================================
    # 🧠 METADATA FILTER (V4)
    # =====================================================
    def is_metadata(self, col):
        return any(m in col for m in self.metadata_keywords)

    # =====================================================
    # 🧠 UNIT / NOISE DETECTION (V4 RESTORED)
    # =====================================================
    def detect_noise(self, col):

        noise_patterns = [
            r"invalid",
            r"unknown",
            r"na",
            r"nan"
        ]

        for p in noise_patterns:
            if re.search(p, col):
                return True

        return False

    # =====================================================
    # 🧠 SCOPE HARD RULES (CRITICAL FIX)
    # =====================================================
    def scope_override(self, col, concept):

        if "scope1" in col:
            return "ghg_scope1"

        if "scope2" in col:
            return "ghg_scope2"

        if "co2" in col and "scope1" not in col and "scope2" not in col:
            return "ghg_scope2"  # conservative default

        return concept

    # =====================================================
    # 🧠 SEMANTIC MAPPING
    # =====================================================
    def map(self, col):

        col_clean = self.clean(col)

        if self.is_metadata(col_clean):
            return {
                "column": col,
                "concept": "metadata",
                "esrs": None,
                "confidence": 0.0,
                "status": "excluded_metadata",
                "reason": "metadata field detected"
            }

        if self.detect_noise(col_clean):
            return {
                "column": col,
                "concept": "unknown",
                "esrs": "UNKNOWN",
                "confidence": 0.0,
                "status": "invalid_data",
                "reason": "noise / invalid value detected"
            }

        col_emb = self.model.encode(col_clean)

        best, best_score = None, -1
        scores = {}

        for concept, emb in self.embeddings.items():

            score = cosine_similarity([col_emb], [emb])[0][0]
            scores[concept] = float(score)

            if score > best_score:
                best = concept
                best_score = score

        confidence = float(best_score)

        # =====================================================
        # V4 + V5 HYBRID OVERRIDE LOGIC
        # =====================================================
        best = self.scope_override(col_clean, best)

        # =====================================================
        # LOW CONFIDENCE HANDLING
        # =====================================================
        if confidence < 0.40:
            return {
                "column": col,
                "concept": "unknown",
                "esrs": "UNKNOWN",
                "confidence": confidence,
                "status": "low_confidence",
                "reason": "no strong semantic match",
                "scores": scores
            }

        return {
            "column": col,
            "concept": best,
            "esrs": self.assign_esrs(best),
            "confidence": confidence,
            "status": "mapped",
            "reason": self.explain(col, best),
            "scores": scores
        }

    # =====================================================
    # 🧠 ESRS ASSIGNMENT (V5)
    # =====================================================
    def assign_esrs(self, concept):

        for esrs, data in ESRS_GRAPH.items():
            if concept in data["required"]:
                return esrs

        return "UNKNOWN"

    # =====================================================
    # 🧠 EXPLAINABILITY (V4 + V5)
    # =====================================================
    def explain(self, col, concept):

        return f"Column '{col}' mapped to '{concept}' using semantic similarity + ESRS constraints"

    # =====================================================
    # 🧠 COMPLIANCE CHECK (V5 CORE)
    # =====================================================
    def compliance(self, results):

        detected = {r["concept"] for r in results}

        report = {}

        for esrs, data in ESRS_GRAPH.items():

            missing = set(data["required"]) - detected

            report[esrs] = {
                "name": data["name"],
                "missing": list(missing),
                "coverage": 1 - len(missing) / len(data["required"]),
                "status": "complete" if not missing else "incomplete"
            }

        return report

    # =====================================================
    # 🚀 RUN PIPELINE
    # =====================================================
    def run(self, df):

        results = []

        for col in df.columns:
            results.append(self.map(col))

        return results, self.compliance(results)


# =========================================================
# 🚀 EXECUTION
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    engine = ESGIntelligenceV5Hybrid()

    results, compliance = engine.run(df)

    results_df = pd.DataFrame(results)

    print("\n=== ESG INTELLIGENCE V5 HYBRID ===\n")
    print(results_df)

    print("\n=== ESRS COMPLIANCE REPORT ===\n")
    for k, v in compliance.items():
        print(k, v)

    results_df.to_csv("esg_v5_hybrid_results.csv", index=False)