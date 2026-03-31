from typing import List
from copy import deepcopy
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..models.schemas import AdvisorDocument, ParsedUserNeeds, RecommendationResult

class RationaleGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.parser = JsonOutputParser()
        
    def generate_recommendation_reasoning(self, 
                                        raw_query: str, 
                                        parsed_needs: ParsedUserNeeds, 
                                        ranked_docs: List[tuple]) -> List[RecommendationResult]:
        
        prompt = PromptTemplate(
            template=(
                "You are a sophisticated 'Elite Financial Concierge' matching expert. Your goal is to explain why a specific advisor is a perfect match for the user's personality, needs, and lifestyle.\n\n"
                "I will provide:\n"
                "1. User's RAW QUERY\n"
                "2. Standardized requirements\n"
                "3. Advisor contexts (including a structured profile AND a full biography/self-introduction)\n\n"
                "YOUR TASK:\n"
                "For EACH advisor, write a personalized 'Matching Rationale' (2-3 sentences). "
                "GO BEYOND TAGS: Instead of just saying 'they have expertise X', focus on their 'Service Philosophy', 'Unique Personal Strengths', or 'Personal Interests' revealed in their BIOGRAPHY.\n"
                "Make the user feel like this advisor's personality and approach (e.g., 'careful', 'innovative', 'trust-based') resonate with their situation.\n\n"
                "CRITICAL: You MUST extract 1-2 EXACT verbatim quotes (citations) from the BIOGRAPHY to prove these unique personal traits.\n\n"
                "Output as a raw valid JSON list of dictionaries. Keys: 'advisor_id', 'match_reasoning', 'citations'.\n\n"
                "USER RAW QUERY:\n{raw_query}\n\n"
                "USER STANDARDIZED NEEDS:\n{parsed_needs}\n\n"
                "ADVISOR CONTEXTS:\n{contexts}\n"
            ),
            input_variables=["raw_query", "parsed_needs", "contexts"]
        )
        
        chain = prompt | self.llm | self.parser
        
        # Prepare context payload
        contexts_str = ""
        for doc, score in ranked_docs:
            contexts_str += f"=== ADVISOR ID: {doc.profile.advisor_id} (Score: {score:.2f}) ===\n"
            contexts_str += f"NAME: {doc.profile.name}\n"
            contexts_str += f"STRUCTURED: Exp: {doc.profile.expertise}, Clients: {doc.profile.target_clients}, Comm: {doc.profile.communication_style}\n"
            contexts_str += f"RAW TEXT: {doc.full_text[:1000]}... [TRUNCATED]\n\n"
            
        try:
            llm_results = chain.invoke({
                "raw_query": raw_query,
                "parsed_needs": parsed_needs.model_dump_json(),
                "contexts": contexts_str
            })
            
            # Reconstruct RecommendationResults mapping LLM JSON to original objects
            final_results = []
            for doc, score in ranked_docs:
                advisor_id = doc.profile.advisor_id
                # Find corresponding generated rationale
                gen_data = next((x for x in llm_results if x.get("advisor_id") == advisor_id), None)
                
                if gen_data:
                    res = RecommendationResult(
                        advisor=doc.profile,
                        match_score=score,
                        rationale=gen_data.get("match_reasoning", "Recommended by semantic matching."),
                        citations=gen_data.get("citations", [])
                    )
                    final_results.append(res)
            return final_results
        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback wrapper
            return [RecommendationResult(
                advisor=d.profile, 
                match_score=s, 
                rationale="Successfully matched by retrieval system, explanation generation failed.", 
                citations=[]) for d, s in ranked_docs]
