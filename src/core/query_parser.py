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
                "You are an intelligent NLP query parser for a financial advisor matching system.\n"
                "Extract the structured user needs from the following query.\n"
                "If the query is too vague, ambiguous, or doesn't mention specific demographics/expertise, leave the fields empty or null.\n"
                "{format_instructions}\n\n"
                "QUERY:\n{query}\n"
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
