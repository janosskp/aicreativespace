import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math


# =========================================================
# 🧠 ESG KNOWLEDGE BASE (unchanged but extended logic-wise)
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "E1-5",
        "keywords": ["energy", "kwh", "electric", "power"]
    },
    "ghg_scope1": {
        "esrs": "E1-6",
        "keywords": ["scope1", "fuel combustion", "direct emissions"]
    },
    "ghg_scope2": {
        "esrs": "E1-6",
        "keywords": ["scope2", "electricity indirect", "grid"]
    },
    "water_usage": {
        "esrs": "E3-4",
        "keywords": ["water", "liters", "h2o"]
    },
    "waste": {
        "esrs": "E5-3",
        "keywords": ["waste", "kg", "landfill"]
    },
    "fuel_consumption": {
        "esrs": "E1-5",
        "keywords": ["gas", "fuel", "diesel"]
    }
}


# =========================================================
# 🧠 ENGINE V12
# =========================================================
class ESGContextIntelligenceV12:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            k: self.model.encode(" ".join(v["keywords"]))
            for k, v in ESG_CONCEPTS.items()
        }

        self.metadata_fields = {"timestamp", "date", "site", "location", "id"}

    # =====================================================
    # 🧠 METADATA FILTER
    # =====================================================
    def is_metadata(self, col):
        return col.lower().strip() in self.metadata_fields

    # =====================================================
    # 🧠 CONTEXT BUILDER (NEW)
    # =====================================================
    def build_context(self, col):

        col_l = col.lower()

        context = col_l

        if "kwh" in col_l:
            context += " electricity energy usage"

        if "kg" in col_l:
            context += " weight mass waste"

        if "liter" in col_l:
            context += " volume water"

        if "co2" in col_l:
            context += " emissions carbon"

        return context

    # =====================================================
    # 🧠 RULE ENGINE (unchanged logic)
    # =====================================================
    def rule_engine(self, col):

        c = col.lower()

        if "kwh" in c or "energy" in c:
            return "energy_consumption"

        if "scope1" in c:
            return "ghg_scope1"

        if "scope2" in c:
            return "ghg_scope2"

        if "water" in c:
            return "water_usage"

        if "waste" in c:
            return "waste"

        if "gas" in c or "fuel" in c:
            return "fuel_consumption"

        if "co2" in c:
            return "ghg_scope2"

        return None

    # =====================================================
    # 🧠 SEMANTIC ENGINE (normalized)
    # =====================================================
    def semantic_engine(self, col):

        context = self.build_context(col)
        emb = self.model.encode(context)

        scores = {}

        for k, v in self.embeddings.items():
            scores[k] = float(cosine_similarity([emb], [v])[0][0])

        # softmax normalization
        exp = np.exp(list(scores.values()))
        probs = exp / np.sum(exp)

        norm_scores = dict(zip(scores.keys(), probs))

        best = max(norm_scores, key=norm_scores.get)

        return best, norm_scores[best], norm_scores

    # =====================================================
    # 🧠 CONFLICT RESOLVER (NEW)
    # =====================================================
    def resolve(self, rule, semantic, sem_score, scores):

        if rule and rule == semantic:
            return rule, "aligned"

        if rule and not semantic:
            return rule, "rule_dominant"

        if semantic and not rule:
            return semantic, "semantic_dominant"

        if rule and semantic and rule != semantic:

            rule_score = scores.get(rule, 0)
            semantic_score = scores.get(semantic, 0)

            if rule_score >= semantic_score:
                return rule, "rule_dominant"
            else:
                return semantic, "semantic_dominant"

        return semantic, "semantic_dominant"

    # =====================================================
    # 🧠 CONFIDENCE MODEL (NEW - REAL)
    # =====================================================
    def confidence(self, scores, best):

        vals = list(scores.values())
        vals_sorted = sorted(vals, reverse=True)

        top1 = vals_sorted[0]
        top2 = vals_sorted[1] if len(vals_sorted) > 1 else 0.0

        margin = top1 - top2
        entropy = -sum([p * math.log(p + 1e-9) for p in vals])

        return float((top1 * 0.6) + (margin * 0.3) + (1 - entropy) * 0.1)

    # =====================================================
    # 🧠 MAIN MAP
    # =====================================================
    def map_column(self, col):

        if self.is_metadata(col):
            return {
                "column": col,
                "final_decision": "metadata",
                "decision_mode": "excluded",
                "confidence": 0.0,
                "esrs": "",
                "reasoning_chain": ["Filtered as metadata"]
            }

        rule = self.rule_engine(col)
        semantic, sem_score, scores = self.semantic_engine(col)

        final, mode = self.resolve(rule, semantic, sem_score, scores)

        conf = self.confidence(scores, final)

        return {
            "column": col,
            "rule_result": rule,
            "semantic_result": semantic,
            "final_decision": final,
            "decision_mode": mode,
            "conflict": rule != semantic if rule else False,
            "esrs": ESG_CONCEPTS.get(final, {}).get("esrs", "UNKNOWN"),
            "confidence": conf,
            "reasoning_chain": [
                f"Rule: {rule}",
                f"Semantic: {semantic}",
                f"Final selected: {final}",
                f"Confidence computed via margin + entropy"
            ],
            "scores": scores
        }

    # =====================================================
    # 🧠 BATCH
    # =====================================================
    def process(self, df):
        return [self.map_column(c) for c in df.columns]


# =========================================================
# 🚀 RUN
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("testdata.csv")

    engine = ESGContextIntelligenceV12()
    results = engine.process(df)

    out = pd.DataFrame(results)

    out["scores"] = out["scores"].apply(str)
    out["reasoning_chain"] = out["reasoning_chain"].apply(str)

    out.to_csv("v12_results.csv", index=False)

    print(out)