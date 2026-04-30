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
                "你是一個專業的理財顧問評分系統。請根據以下準則進行 JSON 評分：\n\n"
                "1. 若 raw_query 為 \"[空值]\"：代表缺乏主觀描述，【所有評分】上限為 70 分。\n"
                "2. 只有在使用者有手寫描述，且理專自傳內容「精準契合」該描述時，才允許給予 85 分以上。\n"
                "3. 僅標籤吻合但無手寫描述，基準分為 60-70 分。\n\n"
                "使用者查詢：{raw_query}\n"
                "需求解析：{parsed_needs}\n"
                "理專資料：{branch}, {expertise}, {target_clients}, {communication_style}\n"
                "理專自傳：{full_text}\n\n"
                "回傳 JSON (bio_fit_score, tag_fit_score, reasoning)：\n"
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
