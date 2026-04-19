import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import os

# =========================================================
# 🧠 ESG KNOWLEDGE BASE
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "E1-5",
        "keywords": [
            "total energy consumption", "purchased electricity", "renewable energy", 
            "kwh kilowatt hours", "mwh megawatt hours", "energy intensity", 
            "power grid usage", "thermal energy", "heating cooling consumption"
        ]
    },
    "ghg_scope1": {
        "esrs": "E1-6",
        "keywords": [
            "direct ghg emissions", "stationary combustion", "mobile combustion", 
            "fugitive emissions", "tco2e scope 1", "refrigerant leakages", 
            "company owned vehicle emissions", "onsite fuel burning"
        ]
    },
    "ghg_scope2": {
        "esrs": "E1-6",
        "keywords": [
            "indirect ghg emissions", "location-based emissions", "market-based emissions", 
            "purchased steam", "district heating", "purchased cooling", 
            "electricity indirect tco2e", "grid carbon intensity"
        ]
    },
    "water_usage": {
        "esrs": "E3-4",
        "keywords": [
            "total water withdrawal", "water consumption liters", "m3 cubic meters", 
            "groundwater abstraction", "municipal water supply", "waste water discharge", 
            "water intensity", "freshwater usage", "h2o volume"
        ]
    },
    "waste": {
        "esrs": "E5-3",
        "keywords": [
            "total waste generated", "hazardous waste kg", "non-hazardous waste", 
            "waste diverted from landfill", "recycling rate", "incineration", 
            "waste recovery operations", "tailings and overburden", "circular economy waste"
        ]
    },
    "fuel_consumption": {
        "esrs": "E1-5",
        "keywords": [
            "natural gas consumption", "diesel fuel usage", "heating oil", 
            "heavy fuel oil hfo", "liquefied petroleum gas lpg", "aviation fuel", 
            "gasoline petrol consumption", "fossil fuel energy", "combustion energy"
        ]
    }
}

# =========================================================
# 🧠 ENGINE V13 (Optimierte V12 Architektur)
# =========================================================
class ESGContextIntelligenceV13:
    def __init__(self, temperature=0.05, min_sim_threshold=0.3):
        # Initialisierung wie V12
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.temp = temperature 
        self.threshold = min_sim_threshold
        
        self.embeddings = {
            k: self.model.encode(" ".join(v["keywords"]))
            for k, v in ESG_CONCEPTS.items()
        }
        self.metadata_fields = {"timestamp", "date", "site", "location", "id"}

    # 🧠 METADATA FILTER
    def is_metadata(self, col):
        return col.lower().strip() in self.metadata_fields

    # 🧠 CONTEXT BUILDER
    def build_context(self, col):
        col_l = col.lower()
        context = col_l
        if "kwh" in col_l: context += " electricity energy usage"
        if "kg" in col_l: context += " weight mass waste"
        if "liter" in col_l: context += " volume water"
        if "co2" in col_l: context += " emissions carbon"
        return context

    # 🧠 RULE ENGINE
    def rule_engine(self, col):
        c = col.lower()
        if "kwh" in c or "energy" in c: return "energy_consumption"
        if "scope1" in c: return "ghg_scope1"
        if "scope2" in c: return "ghg_scope2"
        if "water" in c: return "water_usage"
        if "waste" in c: return "waste"
        if "gas" in c or "fuel" in c: return "fuel_consumption"
        if "co2" in c: return "ghg_scope2"
        return None

    # 🧠 SEMANTIC ENGINE (Normalized with Temperature)
    def semantic_engine(self, col):
        context = self.build_context(col)
        emb = self.model.encode(context)
        
        raw_scores = {}
        for k, v in self.embeddings.items():
            raw_scores[k] = float(cosine_similarity([emb], [v])[0][0])
        
        # Check threshold to avoid misclassifying non-ESG data
        max_raw = max(raw_scores.values())
        if max_raw < self.threshold:
            return None, 0.0, raw_scores

        # Temperature-scaled Softmax
        logits = np.array(list(raw_scores.values()))
        exp = np.exp(logits / self.temp)
        probs = exp / np.sum(exp)
        
        norm_scores = dict(zip(raw_scores.keys(), probs))
        best = max(norm_scores, key=norm_scores.get)
        return best, norm_scores[best], norm_scores

    # 🧠 CONFLICT RESOLVER
    def resolve(self, rule, semantic, scores):
        if rule and rule == semantic:
            return rule, "aligned"
        if rule and not semantic:
            return rule, "rule_dominant"
        if semantic and not rule:
            return semantic, "semantic_dominant"
        if rule and semantic and rule != semantic:
            rule_score = scores.get(rule, 0)
            semantic_score = scores.get(semantic, 0)
            # Falls die Regel semantisch halbwegs plausibel ist, bleibt sie dominant
            if rule_score + 0.1 >= semantic_score: 
                return rule, "rule_dominant"
            else:
                return semantic, "semantic_dominant"
        return semantic, "semantic_dominant"

    # 🧠 CONFIDENCE MODEL
    def confidence_model(self, scores, best):
        if not scores or best is None: return 0.0
        vals = sorted(list(scores.values()), reverse=True)
        top1 = vals[0]
        top2 = vals[1] if len(vals) > 1 else 0.0
        margin = top1 - top2
        return float(top1 * 0.8 + margin * 0.2)

    # 🧠 MAIN MAP
    def map_column(self, col):
        if self.is_metadata(col):
            return {
                "column": col, "final_decision": "metadata", 
                "decision_mode": "excluded", "confidence": 1.0, "esrs": "",
                "reasoning_chain": ["Filtered as metadata"]
            }

        rule = self.rule_engine(col)
        semantic, sem_prob, scores = self.semantic_engine(col)
        final, mode = self.resolve(rule, semantic, scores)
        
        if final is None:
            return {
                "column": col, "final_decision": "unknown", "decision_mode": "failed", 
                "confidence": 0.0, "esrs": "N/A", "reasoning_chain": ["No match found"]
            }

        conf = self.confidence_model(scores, final)

        return {
            "column": col,
            "rule_result": rule,
            "semantic_result": semantic,
            "final_decision": final,
            "decision_mode": mode,
            "conflict": rule != semantic if (rule and semantic) else False,
            "esrs": ESG_CONCEPTS.get(final, {}).get("esrs", "UNKNOWN"),
            "confidence": round(conf, 4),
            "reasoning_chain": [f"Rule: {rule}", f"Sem: {semantic}", f"Mode: {mode}"],
            "scores": scores
        }

    # 🧠 BATCH
    def process(self, df):
        return [self.map_column(c) for c in df.columns]

# =========================================================
# 🚀 RUN (VS Code Integration)
# =========================================================
if __name__ == "__main__":
    input_file = "2testdata.csv"
    output_file = "results.csv"

    if os.path.exists(input_file):
        print(f"🔄 Lade {input_file}...")
        df = pd.read_csv(input_file)
        
        engine = ESGContextIntelligenceV13()
        results = engine.process(df)
        
        out_df = pd.DataFrame(results)
        
        # Formatierung für CSV (ähnlich wie V12)
        out_df["scores"] = out_df["scores"].apply(lambda x: str(x) if x is not None else "")
        out_df["reasoning_chain"] = out_df["reasoning_chain"].apply(str)
        
        out_df.to_csv(output_file, index=False)
        print(f"✅ Mapping abgeschlossen. Ergebnisse in '{output_file}' gespeichert.")
        print(out_df[["column", "final_decision", "confidence", "esrs"]])
    else:
        print(f"❌ Fehler: '{input_file}' nicht gefunden. Bitte erstelle die Datei im selben Ordner.")