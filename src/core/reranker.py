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
                "你是一個專業的財富管理 Reranker 評分系統。你的任務是評估一位「理財專員」與「使用者需求」的契合度。\n"
                "我們將評分拆分為兩個獨立維度，滿分皆為 100 分：\n"
                "1. 軟性需求契合度 (bio_fit_score): 根據『理專自傳』，評估其溝通風格、服務理念、背景特質是否符合使用者的『理想特質與服務經歷』期望。\n"
                "2. 專業能力契合度 (tag_fit_score): 根據『理專專業標籤』，評估其專長、目標客群、服務分行、擅長商品是否能直接解決使用者的具體財務痛點。\n\n"
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
