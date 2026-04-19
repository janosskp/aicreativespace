import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


# =========================================================
# 🧠 1. ESRS GRAPH (V5 preserved)
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
        "name": "Waste",
        "required": ["waste"]
    }
}


# =========================================================
# 🧠 2. EXPANDED ESG ONTOLOGY (NEW V6)
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": "electricity energy kwh power grid usage consumption",
    "ghg_scope1": "direct emissions fuel diesel gas combustion onsite boilers vehicles",
    "ghg_scope2": "indirect emissions electricity purchased grid energy scope2",
    "water_usage": "water liters consumption cooling withdrawal usage facility",
    "waste": "waste kg disposal landfill recycling material waste",
    "fuel_consumption": "gas diesel fuel combustion heating fossil fuel usage"
}


# =========================================================
# 🧠 3. V6 ENGINE
# =========================================================
class ESRSComplianceEngineV6:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            k: self.model.encode(v)
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata_keys = ["timestamp", "date", "site", "location", "id"]

    # =====================================================
    # 🧹 CLEANING (V5 preserved)
    # =====================================================
    def clean(self, col):
        col = col.lower().strip()
        col = re.sub(r"\s+", "_", col)
        return col

    # =====================================================
    # 🧠 METADATA FILTER
    # =====================================================
    def is_metadata(self, col):
        return any(k in col for k in self.metadata_keys)

    # =====================================================
    # 🧠 NOISE DETECTION (V4 restored)
    # =====================================================
    def is_noise(self, col):
        return bool(re.search(r"invalid|unknown|nan|na", col))

    # =====================================================
    # 🧠 HARD RULE ENGINE (NEW V6 CORE)
    # =====================================================
    def apply_hard_rules(self, col, concept):

        # 🔴 RULE 1: Scope1 lock
        if "scope1" in col:
            return "ghg_scope1", "hard_rule_scope1"

        # 🔴 RULE 2: Scope2 lock
        if "scope2" in col:
            return "ghg_scope2", "hard_rule_scope2"

        # 🔴 RULE 3: CO2 fallback logic
        if "co2" in col and "scope1" not in col and "scope2" not in col:
            return "ghg_scope2", "co2_default_scope2_rule"

        return concept, "semantic"

    # =====================================================
    # 🧠 MAPPING ENGINE
    # =====================================================
    def map_column(self, col):

        col_clean = self.clean(col)

        # metadata
        if self.is_metadata(col_clean):
            return self.build_output(col, "metadata", None, 0.0, "excluded_metadata", "metadata detected", {})

        # noise
        if self.is_noise(col_clean):
            return self.build_output(col, "unknown", "UNKNOWN", 0.0, "invalid", "noise detected", {})

        # embedding
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

        # HARD RULES APPLY
        best, rule_reason = self.apply_hard_rules(col_clean, best)

        # confidence calibration (V6 FIX)
        confidence = self.calibrate_confidence(confidence, rule_reason)

        # low confidence handling
        if confidence < 0.40:
            return self.build_output(col, "unknown", "UNKNOWN", confidence, "low_confidence", "no strong match", scores)

        esrs = self.assign_esrs(best)

        reason = f"Mapped via {rule_reason} + semantic similarity"

        return self.build_output(col, best, esrs, confidence, "mapped", reason, scores)

    # =====================================================
    # 🧠 CONFIDENCE CALIBRATION (NEW V6)
    # =====================================================
    def calibrate_confidence(self, conf, rule_reason):

        if "hard_rule" in rule_reason:
            return max(conf, 0.85)  # rules are authoritative

        return conf

    # =====================================================
    # 🧠 ESRS ASSIGNMENT
    # =====================================================
    def assign_esrs(self, concept):

        for esrs, data in ESRS_GRAPH.items():
            if concept in data["required"]:
                return esrs

        return "UNKNOWN"

    # =====================================================
    # 🧠 AUDIT OUTPUT STRUCTURE
    # =====================================================
    def build_output(self, col, concept, esrs, confidence, status, reason, scores):

        return {
            "column": col,
            "concept": concept,
            "esrs": esrs,
            "confidence": confidence,
            "status": status,
            "reason": reason,
            "scores": scores
        }

    # =====================================================
    # 🧠 COMPLIANCE CHECK (V5 preserved)
    # =====================================================
    def compliance_report(self, results):

        detected = {r["concept"] for r in results}

        report = {}

        for esrs, data in ESRS_GRAPH.items():

            missing = set(data["required"]) - detected

            report[esrs] = {
                "coverage": 1 - len(missing) / len(data["required"]),
                "missing": list(missing),
                "status": "complete" if not missing else "incomplete"
            }

        return report

    # =====================================================
    # 🧠 AUDIT TRAIL (NEW V6)
    # =====================================================
    def audit_trail(self, results):

        return [
            {
                "column": r["column"],
                "decision": r["concept"],
                "confidence": r["confidence"],
                "esrs": r["esrs"],
                "status": r["status"]
            }
            for r in results
        ]

    # =====================================================
    # 🚀 PIPELINE
    # =====================================================
    def run(self, df):

        results = []

        for col in df.columns:
            results.append(self.map_column(col))

        return {
            "results": results,
            "compliance": self.compliance_report(results),
            "audit": self.audit_trail(results)
        }


# =========================================================
# 🚀 EXECUTION
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    engine = ESRSComplianceEngineV6()

    output = engine.run(df)

    results_df = pd.DataFrame(output["results"])

    print("\n=== V6 AUDIT-GRADE ESG ENGINE ===\n")
    print(results_df)

    print("\n=== ESRS COMPLIANCE ===\n")
    print(output["compliance"])

    print("\n=== AUDIT TRAIL ===\n")
    print(pd.DataFrame(output["audit"]))

    results_df.to_csv("esg_v6_results.csv", index=False)