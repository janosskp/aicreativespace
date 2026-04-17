import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AIIntelligenceLayer:

    def __init__(self):

        # 🔥 Lightweight but strong model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # 🧠 ESG Knowledge Base (semantic descriptions)
        self.esg_concepts = {
            "energy_consumption": "electricity energy kwh power consumption usage grid",
            "ghg_scope1": "direct emissions fuel combustion diesel gas company vehicles",
            "ghg_scope2": "indirect emissions electricity purchased energy grid usage",
            "water_usage": "water consumption liters usage facility cooling",
            "waste": "waste disposal landfill recycling kg production waste"
        }

        # 🧠 ESRS mapping layer
        self.esrs_mapping = {
            "energy_consumption": "ESRS E1-5",
            "ghg_scope1": "ESRS E1-6",
            "ghg_scope2": "ESRS E1-6",
            "water_usage": "ESRS E3-4",
            "waste": "ESRS E5-3"
        }

        # Precompute embeddings
        self.concept_embeddings = {
            k: self.model.encode(v)
            for k, v in self.esg_concepts.items()
        }

    # =====================================================
    # 🧠 CORE FUNCTION: semantic mapping
    # =====================================================
    def map_column(self, column_name: str):

        col_embedding = self.model.encode(column_name.lower())

        best_match = None
        best_score = -1

        scores = {}

        # compare against all ESG concepts
        for concept, emb in self.concept_embeddings.items():
            score = cosine_similarity(
                [col_embedding],
                [emb]
            )[0][0]

            scores[concept] = float(score)

            if score > best_score:
                best_score = score
                best_match = concept

        # 🎯 confidence logic
        confidence = best_score

        # decision logic (important for production)
        if confidence < 0.45:
            return {
                "column": column_name,
                "concept": "unknown",
                "esrs": "UNKNOWN",
                "confidence": confidence,
                "status": "low_confidence",
                "scores": scores
            }

        concept = best_match

        return {
            "column": column_name,
            "concept": concept,
            "esrs": self.esrs_mapping.get(concept, "UNKNOWN"),
            "confidence": confidence,
            "status": "mapped",
            "scores": scores
        }

    # =====================================================
    # 🧠 batch processing
    # =====================================================
    def process_dataframe(self, df):

        results = []

        for col in df.columns:
            results.append(self.map_column(col))

        return results