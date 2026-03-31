from typing import List
from copy import deepcopy
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

from pydantic import BaseModel, Field
from ..models.schemas import AdvisorDocument, ParsedUserNeeds, RecommendationResult

class GenResult(BaseModel):
    advisor_id: str = Field(description="The exact ID of the advisor")
    match_reasoning: str = Field(description="Personalized rationale in Traditional Chinese")
    citations: List[str] = Field(description="List of verbatim quotes in Traditional Chinese")

class GenResponse(BaseModel):
    results: List[GenResult]

class RationaleGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.parser = PydanticOutputParser(pydantic_object=GenResponse)
        
    def generate_recommendation_reasoning(self, 
                                        raw_query: str, 
                                        parsed_needs: ParsedUserNeeds, 
                                        ranked_docs: List[tuple]) -> List[RecommendationResult]:
        
        prompt = PromptTemplate(
            template=(
                "你現在是使用者的「熟識好友」。你的任務是向使用者熱情地推薦一位你認識的理專朋友，解釋為什麼這位理專超適合他。\n\n"
                "【語氣要求】\n"
                "極度自然、日常口語化、充滿人情味（像是在跟好朋友喝咖啡時的聊天口吻），要帶入情感，稱呼理專時就像在熱心介紹自己的好夥伴或好朋友。千萬不要像死板的客服人員或高高在上的管家。\n\n"
                "請全程僅使用「繁體中文」進行回答，這點非常重要。\n\n"
                "我將提供：\n"
                "1. 使用者的原始查詢 (RAW QUERY)\n"
                "2. 標準化需求 (Standardized requirements)\n"
                "3. 理專背景資料（包含結構化標籤與完整的「理專自傳」）\n\n"
                "你的任務：\n"
                "為每一位理專撰寫一段個人化的「推薦對話」(約 3-4 句話)。\n"
                "超越標籤：不要冷冰冰地列點說「他具備某某專業」，請專注於自傳中揭露的「服務理念」、「獨特個人優勢」或「個人興趣」。\n"
                "讓使用者感覺你真的很懂這位理專的性格與作風（例如：細心且穩健、具備創新精神、以信任為本），並且告訴使用者這點有多契合他的處境。\n\n"
                "關鍵要求：必須從「完整自傳原文」中提取 1-2 句原始引述 (citations) 來佐證這些獨特的個人特質。\n\n"
                "【重要輸出格式】\n"
                "{format_instructions}\n"
                "請嚴格遵守這個 JSON 結構，並且保證所有的 values (推薦對話與引述) 都必須是「繁體中文」。\n\n"
                "使用者查詢內容：\n{raw_query}\n\n"
                "標準化需求總結：\n{parsed_needs}\n\n"
                "理專背景資料：\n{contexts}\n"
            ),
            input_variables=["raw_query", "parsed_needs", "contexts"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | self.parser
        
        contexts_str = ""
        for doc, score in ranked_docs:
            contexts_str += f"=== 理專檔案 ID: {doc.profile.advisor_id} (配對分數: {score:.2f}) ===\n"
            contexts_str += f"姓名: {doc.profile.name}\n"
            contexts_str += f"結構化特徵: 專長={doc.profile.expertise}, 目標客群={doc.profile.target_clients}, 溝通風格={doc.profile.communication_style}\n"
            contexts_str += f"完整自傳原文: {doc.full_text[:1000]}... [以下省略]\n\n"
            
        try:
            llm_results = chain.invoke({
                "raw_query": raw_query,
                "parsed_needs": parsed_needs.model_dump_json(),
                "contexts": contexts_str
            })
            
            # Reconstruct RecommendationResults mapping LLM Pydantic objects to original objects
            final_results = []
            for doc, score in ranked_docs:
                advisor_id = doc.profile.advisor_id
                # Find corresponding generated rationale
                gen_data = next((x for x in llm_results.results if x.advisor_id == advisor_id), None)
                
                if gen_data:
                    res = RecommendationResult(
                        advisor=doc.profile,
                        match_score=score,
                        rationale=gen_data.match_reasoning or "系統已透過語意搜尋成功找到此理專，但未能順利生成推薦摘要。",
                        citations=gen_data.citations or []
                    )
                    final_results.append(res)
            return final_results
        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback wrapper
            return [RecommendationResult(
                advisor=d.profile, 
                match_score=s, 
                rationale="系統已成功檢索到高分的理專檔案，但向語言模型請求推薦理由時發生錯誤或逾時。", 
                citations=[]) for d, s in ranked_docs]
