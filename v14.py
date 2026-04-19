import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# =========================================================
# 🧠 KNOWLEDGE BASE V15
# =========================================================
ESG_CONCEPTS = {
    "energy_consumption": {
        "esrs": "E1-5",
        "keywords": ["electricity mwh", "renewable energy share", "total energy consumption", "kwh usage", "power grid"]
    },
    "ghg_scope1": {
        "esrs": "E1-6",
        "keywords": ["direct emissions", "scope 1 tco2e", "stationary combustion", "fleet emissions", "refrigerant leak"]
    },
    "ghg_scope2": {
        "esrs": "E1-6",
        "keywords": ["indirect emissions", "scope 2 tco2e", "market-based emissions", "purchased heat steam"]
    },
    "water_usage": {
        "esrs": "E3-4",
        "keywords": ["water withdrawal", "freshwater m3", "water consumption liters", "groundwater abstraction"]
    },
    "waste": {
        "esrs": "E5-3",
        "keywords": ["hazardous waste kg", "non-hazardous trash", "waste landfill", "recycled plastic weight"]
    },
    "fuel_consumption": {
        "esrs": "E1-5",
        "keywords": ["fuel oil liters", "natural gas consumption", "diesel usage", "heating oil", "petrol gasoline"]
    }
}

# =========================================================
# 🧠 ENGINE V15 (Fix: Ensure 'process' is present)
# =========================================================
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
        
        # Sicherstellen, dass die Keys und Probs gemappt werden
        prob_dict = dict(zip(raw_scores.keys(), probs))
        best_key = max(prob_dict, key=prob_dict.get)
        return best_key, prob_dict[best_key], prob_dict

    def resolve(self, rule, semantic, scores):
        if rule == semantic: return rule, "aligned"
        if rule and not semantic: return rule, "rule_dominant"
        if semantic and not rule: return semantic, "semantic_dominant"
        if rule and semantic and rule != semantic:
            rule_prob = scores.get(rule, 0) + 0.25
            sem_prob = scores.get(semantic, 0)
            return (rule, "rule_priority") if rule_prob >= sem_prob else (semantic, "semantic_override")
        return semantic, "semantic_dominant"

    def confidence_model(self, scores, best):
        if not scores or best is None or best == "unknown": return 0.0
        vals = sorted(list(scores.values()), reverse=True)
        top1 = vals[0]
        top2 = vals[1] if len(vals) > 1 else 0.0
        return float(top1 * 0.8 + (top1 - top2) * 0.2)

    def map_column(self, col):
        if self.is_metadata(col):
            return {"column": col, "final_decision": "metadata", "decision_mode": "excluded", "confidence": 1.0, "esrs": "", "needs_review": False}
        
        rule = self.rule_engine(col)
        semantic, sem_prob, scores = self.semantic_engine(col)
        final, mode = self.resolve(rule, semantic, scores)
        
        if final is None or (mode == "semantic_dominant" and sem_prob < 0.4):
            return {"column": col, "final_decision": "unknown", "decision_mode": "failed", "confidence": 0.0, "esrs": "N/A", "needs_review": True}

        conf = round(self.confidence_model(scores, final), 4)
        needs_review = conf < 0.60 or mode in ["semantic_override", "rule_priority"]

        return {
            "column": col,
            "final_decision": final,
            "decision_mode": mode,
            "confidence": conf,
            "esrs": ESG_CONCEPTS.get(final, {}).get("esrs", "UNKNOWN"),
            "conflict": rule != semantic if (rule and semantic) else False,
            "needs_review": needs_review
        }

    # --- DIE METHODE DIE GEFEHLT HAT ---
    def process(self, df):
        return [self.map_column(c) for c in df.columns]

# =========================================================
# 📊 POST-PROCESSING
# =========================================================
def run_post_processing(results_df):
    print("\n" + "="*50)
    print("📢 POST-PROCESSING ANALYSE")
    print("="*50)
    
    total = len(results_df)
    metadata = len(results_df[results_df['final_decision'] == 'metadata'])
    mapped = len(results_df[(results_df['final_decision'] != 'metadata') & (results_df['final_decision'] != 'unknown')])
    failed = len(results_df[results_df['final_decision'] == 'unknown'])
    
    print(f"Gesamtanzahl Spalten: {total}")
    print(f"Erfolgreich gemappt:  {mapped}")
    print(f"Metadaten (Ignoriert): {metadata}")
    print(f"Nicht identifiziert:  {failed}")
    
    review_needed = results_df[results_df['needs_review'] == True]
    if not review_needed.empty:
        print("\n⚠️ REVIEW EMPFOHLEN FÜR:")
        for _, row in review_needed.iterrows():
            print(f" - [{row['column']}] -> Ziel: {row['final_decision']} (Conf: {row['confidence']})")
    
    print("="*50 + "\n")

# =========================================================
# 🚀 HAUPTPROGRAMM
# =========================================================
if __name__ == "__main__":
    input_file = "2testdata.csv"
    output_file = "results_v15_final.csv"

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        # Wichtig: Name der Klasse muss exakt stimmen
        engine = ESGContextIntelligenceV15()
        
        mapping_results = engine.process(df)
        out_df = pd.DataFrame(mapping_results)
        
        out_df.to_csv(output_file, index=False)
        print(f"✅ CSV gespeichert unter: {output_file}")
        
        run_post_processing(out_df)
    else:
        print(f"❌ Datei '{input_file}' fehlt.")