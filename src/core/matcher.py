from typing import List, Tuple
from ..models.schemas import AdvisorDocument, ParsedUserNeeds, RecommendationResult

class Matcher:
    def __init__(self, indexer):
        self.indexer = indexer
        # Weights (50/50 split as requested)
        self.BIO_WEIGHT = 0.5
        self.TAGS_WEIGHT = 0.5
        

    def rank_advisors(self, raw_query: str, parsed_needs: ParsedUserNeeds, top_k: int = 3) -> Tuple[List[Tuple[AdvisorDocument, float]], List[Tuple[AdvisorDocument, float]], List[Tuple[AdvisorDocument, float]]]:
        # Phase 0: Retrieval from both indices
        # Build a descriptive query for the tags index to leverage client profile matching
        tags_query_parts = [raw_query]
        if parsed_needs.investment_experience:
            tags_query_parts.append(f"投資經驗:{parsed_needs.investment_experience}")
        if parsed_needs.asset_scale:
            tags_query_parts.append(f"資產規模:{parsed_needs.asset_scale}")
        if parsed_needs.products_touched:
            tags_query_parts.append(f"曾接觸商品:{', '.join(parsed_needs.products_touched)}")
        if parsed_needs.asset_allocation:
            tags_query_parts.append(f"目前的資產配置:{', '.join(parsed_needs.asset_allocation)}")
            
        tags_query = ", ".join(tags_query_parts)

        # We fetch a larger pool to ensure we can merge them and apply filters
        bio_results = self.indexer.semantic_search(raw_query, index_type="bio", top_k=50)
        tags_results = self.indexer.semantic_search(tags_query, index_type="tags", top_k=50)
        
        # Branch Filter (Hard Filter)
        branch = parsed_needs.branch_needed
        if branch and branch != "未提供":
            bio_results = [(doc, score) for doc, score in bio_results if doc.profile.branch == branch]
            tags_results = [(doc, score) for doc, score in tags_results if doc.profile.branch == branch]
            
        if not bio_results and not tags_results:
            return [], []

        # Phase 1: Score Merging
        # advisor_id -> [bio_score, tags_score]
        scores_map = {}
        docs_map = {}
        
        for doc, score in bio_results:
            aid = doc.profile.advisor_id
            scores_map[aid] = [score, 0.0]
            docs_map[aid] = doc
            
        for doc, score in tags_results:
            aid = doc.profile.advisor_id
            if aid in scores_map:
                scores_map[aid][1] = score
            else:
                scores_map[aid] = [0.0, score] # If not in bio top_k, bio score is 0
                docs_map[aid] = doc
                
        # Phase 2: Combined Scoring (50/50) * 100
        ranked_docs = []
        for aid, (s_bio, s_tags) in scores_map.items():
            # Average similarity * 100
            final_score = ((s_bio * self.BIO_WEIGHT) + (s_tags * self.TAGS_WEIGHT)) * 100
            ranked_docs.append((docs_map[aid], final_score))
            
        # Sort descending by final score
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return (bio_top5, tags_top5, final_results)
        return bio_results[:5], tags_results[:5], ranked_docs[:top_k]
