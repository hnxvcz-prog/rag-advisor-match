from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from ..models.schemas import ParsedUserNeeds

class QueryParser:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=ParsedUserNeeds)
        
    def parse_query(self, raw_query: str) -> ParsedUserNeeds:
        prompt = PromptTemplate(
            template=(
                "你是一位智能理財顧問匹配系統的需求解析專家。\n"
                "請從以下使用者的口語化查詢中，提取出結構化的需求欄位。\n"
                "如果查詢太過模糊、曖昧，或未提及特定的客群/專業領域，請將欄位保留為空或 null。\n"
                "{format_instructions}\n\n"
                "查詢內容：\n{query}\n"
            ),
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | self.parser
        
        try:
            return chain.invoke({"query": raw_query})
        except Exception as e:
            print(f"Failed to parse query: {e}")
            return ParsedUserNeeds()
