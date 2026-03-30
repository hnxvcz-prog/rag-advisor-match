from typing import List, Tuple
from ..models.schemas import AdvisorDocument, ParsedUserNeeds, RecommendationResult

class Matcher:
    def __init__(self, indexer):
        self.indexer = indexer
        # Weights
        self.SEMANTIC_WEIGHT = 0.6
        self.STRUCTURED_WEIGHT = 0.4
        
    def _calculate_structured_score(self, profile, requirements: ParsedUserNeeds) -> float:
        """Calculates logic exact match bonus based on overlaps. Max score 1.0."""
        matched = 0
        total_conditions = 0
        
        if requirements.expertise_needed:
            total_conditions += 1
            if any(req.lower() in [e.lower() for e in profile.expertise] for req in requirements.expertise_needed):
                matched += 1
                
        if requirements.target_clients_needed:
            total_conditions += 1
            if any(req.lower() in [c.lower() for c in profile.target_clients] for req in requirements.target_clients_needed):
                matched += 1
                
        if requirements.communication_preference:
            total_conditions += 1
            if requirements.communication_preference.lower() in profile.communication_style.lower():
                matched += 1
                
        if total_conditions == 0:
            return 1.0 # If user asks nothing structured, don't penalize structured score
            
        return matched / total_conditions

    def rank_advisors(self, raw_query: str, parsed_needs: ParsedUserNeeds, top_k: int = 3) -> List[Tuple[AdvisorDocument, float]]:
        # Phase 1: Semantic Search
        semantic_results = self.indexer.semantic_search(raw_query, top_k=10)
        
        ranked_docs = []
        for doc, sem_score in semantic_results:
            # Phase 2: Structured Scoring
            struct_score = self._calculate_structured_score(doc.profile, parsed_needs)
            
            # Phase 3: Blended Score
            # sem_score is cosine similarity roughly [0, 1]
            final_score = (sem_score * self.SEMANTIC_WEIGHT) + (struct_score * self.STRUCTURED_WEIGHT)
            ranked_docs.append((doc, final_score))
            
        # Sort descending by final score
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs[:top_k]
