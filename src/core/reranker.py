from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from ..models.schemas import AdvisorDocument, ParsedUserNeeds

class RerankScore(BaseModel):
    bio_fit_score: float = Field(..., description="自傳文本與使用者軟性需求的契合度 (0-100分)")
    tag_fit_score: float = Field(..., description="專業標籤、投資經驗等硬性條件與使用者財務需求的契合度 (0-100分)")
    reasoning: str = Field(..., description="給出此評分的簡短中文理由")

class LLMReranker:
    def __init__(self):
        # We use a lower temperature for consistent scoring
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=RerankScore)
        self.prompt = PromptTemplate(
            template=(
                "你是一個極度嚴格且客觀的「財富管理顧問面試官」。你的任務是評估理財專員與使用者需求的契合度。\n\n"
                "【嚴格評分準則 - 必讀】\n"
                "1. 證據導向：僅能根據提供的文字證據評分。嚴禁憑空補腦或假設使用者未提及的需求。\n"
                "2. 區分「無衝突」與「強契合」：如果理專的資料與需求只是「沒有衝突」，基準分應為 60 分。只有在理專自傳或標籤中有「明確且具體」的證據證明能解決使用者痛點時，才能給予 85 分以上的高分。\n"
                "3. 懲罰模糊或空輸入：如果 raw_query 為 \"[無手寫描述]\" 或內容極其空洞，代表使用者未提供任何性格或服務期望。在此情況下，bio_fit_score (軟性契合度) 嚴禁超過 60 分，且必須在理由中註明「因缺乏使用者描述，無法評估契合度」。\n"
                "4. 拒絕推銷語氣：你的理由 (reasoning) 必須冷靜、批判且中立。請同時指出該理專可能「不足」或「僅是勉強符合」的地方。\n\n"
                "【維度定義】\n"
                "1. 軟性需求契合度 (bio_fit_score): 0-100分。根據『理專自傳』，評估其風格、特質是否「精準」對應使用者手寫描述的期望。\n"
                "2. 專業能力契合度 (tag_fit_score): 0-100分。評估標籤、專長、分行等硬性條件是否能直接滿足使用者選擇的財務目標。\n\n"
                "使用者原始自然語言查詢：\n{raw_query}\n\n"
                "系統解析後的使用者多維度需求：\n{parsed_needs}\n\n"
                "候選理專資料：\n"
                "【理專標籤】: 分行={branch}, 專長={expertise}, 熟悉客群={target_clients}, 溝通風格={communication_style}\n"
                "【理專全文自傳】: {full_text}\n\n"
                "請嚴格根據以下 JSON 格式回傳這兩個分數與理由：\n"
                "{format_instructions}"
            ),
            input_variables=["raw_query", "parsed_needs", "branch", "expertise", "target_clients", "communication_style", "full_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser

    def score_document(self, raw_query: str, parsed_needs: ParsedUserNeeds, doc: AdvisorDocument) -> RerankScore:
        try:
            result = self.chain.invoke({
                "raw_query": raw_query,
                "parsed_needs": parsed_needs.model_dump_json(),
                "branch": doc.profile.branch,
                "expertise": ", ".join(doc.profile.expertise),
                "target_clients": ", ".join(doc.profile.target_clients),
                "communication_style": doc.profile.communication_style,
                "full_text": doc.full_text[:1500]  # limit to avoid context max out
            })
            return result
        except Exception as e:
            print(f"Reranker Error: {e}")
            # Fallback score if parser fails
            return RerankScore(bio_fit_score=50.0, tag_fit_score=50.0, reasoning="LLM Reranker 發生錯誤，回傳基準分數。")
