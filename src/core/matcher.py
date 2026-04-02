from typing import List, Tuple
from ..models.schemas import AdvisorDocument, ParsedUserNeeds, RecommendationResult

class Matcher:
    def __init__(self, indexer):
        self.indexer = indexer
        # Weights (Base: Semantic 90%, Bonus: Structured 10%)
        self.SEMANTIC_WEIGHT = 0.9
        self.STRUCTURED_WEIGHT = 0.1
        
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
            return 0.0 # If user asks nothing structured, don't award any bonus
            
        return matched / total_conditions

    def rank_advisors(self, raw_query: str, parsed_needs: ParsedUserNeeds, top_k: int = 3) -> Tuple[List[Tuple[AdvisorDocument, float]], List[Tuple[AdvisorDocument, float]]]:
        # Phase 0: Hard Filter (Branch)
        all_docs = self.indexer.documents
        filtered_docs = all_docs
        if parsed_needs.branch_needed and parsed_needs.branch_needed != "未提供":
            filtered_docs = [doc for doc in all_docs if doc.profile.branch == parsed_needs.branch_needed]
            
        # If no one matches the branch, we might return empty or continue with all to be safe? 
        # Requirement said "Hard Filter", so we should strictly respect it.
        if not filtered_docs:
            return [], []

        # Phase 1: Semantic Search (Retrieve candidates among filtered pool)
        # Note: Indexer currently searches globally. We need to filter the results.
        semantic_results = self.indexer.semantic_search(raw_query, top_k=20) # Get more to allow filtering
        
        # Filter semantic results by branch
        if parsed_needs.branch_needed and parsed_needs.branch_needed != "未提供":
            semantic_results = [(doc, score) for doc, score in semantic_results if doc.profile.branch == parsed_needs.branch_needed]
        
        # Take Top 10 from filtered semantic results
        semantic_results = semantic_results[:10]
        
        ranked_docs = []
        for doc, sem_score in semantic_results:
            # Phase 2: Structured Scoring
            struct_score = self._calculate_structured_score(doc.profile, parsed_needs)
            
            # Phase 3: Blended Score
            # sem_score is cosine similarity. We treat it as the base.
            # structured score (max 1.0) is multiplied by 0.1 as an additive bonus.
            final_score = sem_score + (struct_score * self.STRUCTURED_WEIGHT)
            ranked_docs.append((doc, final_score))
            
        # Sort descending by final score
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return semantic_results, ranked_docs[:top_k]
