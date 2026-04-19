import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# =========================================================
# 🧠 KNOWLEDGE BASE V15
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "E1-5",
        "keywords": [
            "electricity mwh", "renewable energy share", "total energy consumption", 
            "kwh usage", "power grid", "solar panels", "wind power", "energy efficiency",
            "purchased electricity", "renewable electricity"
        ]
    },
    "ghg_scope1": {
        "esrs": "E1-6",
        "keywords": [
            "direct emissions", "scope 1 tco2e", "stationary combustion", 
            "fleet emissions", "refrigerant leak", "company cars", "transport fleet",
            "fugitive emissions", "co2 direct", "owned vehicles"
        ]
    },
    "ghg_scope2": {
        "esrs": "E1-6",
        "keywords": [
            "indirect emissions", "scope 2 tco2e", "market-based emissions", 
            "location-based emissions", "purchased heat steam", "purchased cooling",
            "indirect ghg", "electricity emissions"
        ]
    },
    "water_usage": {
        "esrs": "E3-4",
        "keywords": [
            "water withdrawal", "freshwater m3", "water consumption liters", 
            "groundwater abstraction", "cooling water discharge", "effluent water",
            "water intake", "municipal water", "wastewater discharge"
        ]
    },
    "waste": {
        "esrs": "E5-3",
        "keywords": [
            "hazardous waste kg", "non-hazardous trash", "waste landfill", 
            "recycled plastic weight", "waste incineration", "scrap metal",
            "compostable waste", "waste disposal", "reused materials"
        ]
    },
    "fuel_consumption": {
        "esrs": "E1-5",
        "keywords": [
            "fuel oil liters", "natural gas consumption", "diesel usage", 
            "heating oil", "petrol gasoline", "natural gas heating", "biomass fuel",
            "propane", "heavy fuel oil", "combustion fuels"
        ]
    }
}

class ESGContextIntelligenceV15:
    def __init__(self, temperature=0.05, min_sim_threshold=0.32):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.temp = temperature
        self.threshold = min_sim_threshold
        self.embeddings = {k: self.model.encode(" ".join(v["keywords"])) for k, v in ESG_CONCEPTS.items()}
        self.metadata_fields = {"timestamp", "date", "site_id", "location_country", "id", "site"}

    def is_metadata(self, col):
        return col.lower().strip() in self.metadata_fields

    def build_context(self, col):
        c = col.lower()
        context = c
        if "kwh" in c or "mwh" in c: context += " energy power"
        if "co2" in c or "emissions" in c: context += " ghg carbon"
        if "waste" in c or "trash" in c: context += " refuse disposal"
        if "fuel" in c or "oil" in c or "gas" in c: context += " combustion burning"
        return context

    def rule_engine(self, col):
        c = col.lower()
        if "fuel" in c or "oil" in c or "gas" in c: return "fuel_consumption"
        if "mwh" in c or "kwh" in c or "energy" in c: return "energy_consumption"
        if "scope 1" in c or "scope1" in c: return "ghg_scope1"
        if "scope 2" in c or "scope2" in c: return "ghg_scope2"
        if "water" in c: return "water_usage"
        if "waste" in c or "trash" in c: return "waste"
        if "emissions" in c: return "ghg_scope1"
        return None

    def semantic_engine(self, col):
        context = self.build_context(col)
        emb = self.model.encode(context)
        raw_scores = {k: float(cosine_similarity([emb], [v])[0][0]) for k, v in self.embeddings.items()}
        
        max_raw = max(raw_scores.values())
        if max_raw < self.threshold: return None, 0.0, raw_scores

        logits = np.array(list(raw_scores.values()))
        exp = np.exp(logits / self.temp)
        probs = exp / np.sum(exp)
        prob_dict = dict(zip(raw_scores.keys(), probs))
        best_key = max(raw_scores, key=raw_scores.get)
        return best_key, prob_dict[best_key], prob_dict

    def resolve(self, rule, semantic, scores):
        if rule == semantic: return rule, "aligned"
        if rule and not semantic: return rule, "rule_dominant"
        if semantic and not rule: return semantic, "semantic_dominant"
        
        if rule and semantic and rule != semantic:
            rule_prob = scores.get(rule, 0) + 0.25
            sem_prob = scores.get(semantic, 0)
            if rule_prob >= sem_prob:
                return rule, "rule_priority"
            else:
                return semantic, "semantic_override"
        
        return semantic, "semantic_dominant"

    def confidence_model(self, scores, best):
        if not scores or best is None or best == "unknown": return 0.0
        vals = sorted(list(scores.values()), reverse=True)
        top1 = vals[0]
        top2 = vals[1] if len(vals) > 1 else 0.0
        return float(top1 * 0.8 + (top1 - top2) * 0.2)

    def map_column(self, col):
        res = {
            "column": col, "final_decision": "unknown", "decision_mode": "failed", 
            "confidence": 0.0, "esrs": "N/A", "reasoning_chain": [], 
            "rule_result": None, "semantic_result": None, "conflict": False, "scores": None
        }

        if self.is_metadata(col):
            res.update({"final_decision": "metadata", "decision_mode": "excluded", "confidence": 1.0, "esrs": "", "reasoning_chain": ["Filtered as metadata"]})
            return res
        
        rule = self.rule_engine(col)
        semantic, sem_prob, scores = self.semantic_engine(col)
        final, mode = self.resolve(rule, semantic, scores)
        
        if final is None:
            res["reasoning_chain"] = ["No match found"]
            return res

        conf = round(self.confidence_model(scores, final), 4)
        
        return {
            "column": col,
            "final_decision": final,
            "decision_mode": mode,
            "confidence": conf,
            "esrs": ESG_CONCEPTS.get(final, {}).get("esrs", "UNKNOWN"),
            "reasoning_chain": [f"Rule: {rule}", f"Sem: {semantic}", f"Mode: {mode}"],
            "rule_result": rule,
            "semantic_result": semantic,
            "conflict": rule != semantic if (rule and semantic) else False,
            "scores": scores,
            "needs_review": conf < 0.60 or mode in ["semantic_override", "rule_priority"]
        }

    def process(self, df):
        return [self.map_column(c) for c in df.columns]

if __name__ == "__main__":
    input_file = "2testdata.csv"
    output_file = "results_v15_final.csv"

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        engine = ESGContextIntelligenceV15()
        mapping_results = engine.process(df)
        out_df = pd.DataFrame(mapping_results)
        
        # Formatierung für CSV Export (wie früher)
        out_df['scores'] = out_df['scores'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else "nan")
        out_df['reasoning_chain'] = out_df['reasoning_chain'].apply(lambda x: str(x))

        out_df.to_csv(output_file, index=False)
        print(f"✅ V15 Processing abgeschlossen. Ergebnisse in {output_file}")
    else:
        print(f"❌ Datei {input_file} nicht gefunden.")