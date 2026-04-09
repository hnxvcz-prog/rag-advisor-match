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
                "【核心解析重點】：請特別判定此查詢中有關「人格特質、專業領域、投資特徵」等關鍵字。\n"
                "如果查詢太過模糊、曖昧，或未提及特定的客群/專業領域，請將專業領域、投資特徵等欄位保留為空或 null。\n\n"
                "特別注意：\n"
                "1. 若使用者提到了特定的地點或分行（如：新北、台北），請將其填入 branch_needed 欄位。\n"
                "2. 請根據對話內容，將客戶特徵對應到以下標準選項（若未提及則跳過）：\n"
                "   - investment_experience: 1年以下, 1~3年, 3~5年, 5年以上\n"
                "   - products_touched: 定存, 外匯, 基金, ETF, 股票, 債券, 結構型商品, 保險\n"
                "   - asset_allocation: 現金/存款, 基金/ETF, 股票, 債券/固定收益商品, 結構型商品, 保險, 尚未明確配置\n"
                "   - asset_scale: 300~1000萬, 1000~3000萬, 3000萬以上\n\n"
                "【重要判斷】：請判定此查詢「是否屬於尋找理財顧問、金融服務或投資規劃的範疇」。\n"
                "   - 視為相關 (is_relevant: True)：包含尋找理專、尋求理財諮詢、描述特定的投資需求、詢問銀行金融服務。\n"
                "   - 視為不相關 (is_relevant: False)：單純的打招呼（你好、哈囉）、日常閒聊（今天天氣如何）、非理專相關的請求（幫我寫程式、講個笑話、我想吃飯）。\n\n"
                "如果判定為不相關，請將 is_relevant 設為 False，"
                "並且在 guidance_message 中生成一段【繁體中文】友善的回應，告訴他這是理專媒合系統，並引導他提出與財務、投資或理想理專特質相關的需求。\n"
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
