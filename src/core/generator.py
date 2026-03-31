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
                "你是一位資深的「頂級理財管家」匹配專家。你的目標是向使用者解釋，為什麼特定的理專是最符合其個性、需求與生活型態的選擇。\n\n"
                "請全程僅使用「繁體中文 (Traditional Chinese)」進行回答，這點非常重要。\n\n"
                "我將提供：\n"
                "1. 使用者的原始查詢 (RAW QUERY)\n"
                "2. 標準化需求 (Standardized requirements)\n"
                "3. 理專背景資料（包含結構化標籤與完整的「理專自傳」）\n\n"
                "你的任務：\n"
                "為每一位理專撰寫一段個人化的「推薦理由」(約 2-3 句話)。\n"
                "超越標籤：不要只說「他具備某某專業」，請專注於自傳中揭露的「服務理念」、「獨特個人優勢」或「個人興趣」。\n"
                "讓使用者感覺這位理專的性格與作風（例如：細心且穩健、具備創新精神、以信任為本）非常契合他的處境。\n\n"
                "關鍵要求：必須從「理專自傳」中提取 1-2 句原始引述 (citations) 來佐證這些獨特的個人特質。\n\n"
                "輸出格式：請輸出為 JSON 列表，每個物件包含鍵：'advisor_id', 'match_reasoning', 'citations'。\n\n"
                "使用者查詢內容：\n{raw_query}\n\n"
                "標準化需求總結：\n{parsed_needs}\n\n"
                "理專背景資料：\n{contexts}\n"
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
