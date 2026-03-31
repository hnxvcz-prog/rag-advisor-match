import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
import streamlit as st
import json

from src.core.document_parser import DocumentParser
from src.core.indexer import Indexer
from src.core.query_parser import QueryParser
from src.core.matcher import Matcher
from src.core.generator import RationaleGenerator
from src.models.schemas import AdvisorDocument

load_dotenv()

st.set_page_config(page_title="RAG Advisor Match", layout="wide")

@st.cache_resource
def load_system(version_tag: str):
    # Attempt to load processed JSONs if they exist, otherwise initialize
    processed_dir = Path("data/processed")
    docs = []
    
    # We add a print statement to see what's happening in logs
    if processed_dir.exists():
        files = list(processed_dir.glob("*.json"))
        # Filter out any lingering temp files or non-json
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    docs.append(AdvisorDocument(**data))
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    
    # If no processed data or we want to force re-parse
    if not docs:
        st.info("Parsing raw Word files...")
        parser = DocumentParser()
        docs = parser.process_all()
        if not docs:
            st.error("No documents found.")
            return None, None, None, 0
            
    indexer = Indexer()
    indexer.add_documents(docs)
    
    query_parser = QueryParser()
    matcher = Matcher(indexer)
    generator = RationaleGenerator()
    
    return query_parser, matcher, generator, len(docs)

st.title("🤝 RAG Advisor Matching MVP v1.4")
st.caption("更新：強化繁體中文輸出指令 | 強制快取刷新機制")

# We pass a version string to force-invalidate st.cache_resource if we update it
query_parser, matcher, generator, docs_count = load_system(version_tag="v1.4-chinese-fix")

if docs_count:
    st.success(f"System Ready. Indexed **{docs_count}** Advisor Documents.")
    
    query = st.text_input("描述您的理財需求 (Describe your financial needs):", placeholder="範例：我在尋找一位能為高資產客群進行退休規劃，且溝通風格溫和的理專")
    
    if st.button("搜尋 (Search)") and query:
        with st.spinner("Analyzing query and matching advisors..."):
            # 1. Parse Query
            parsed_needs = query_parser.parse_query(query)
            with st.expander("🔍 查詢解析結果 (Parsed Criteria)"):
                st.json(parsed_needs.model_dump())
                
            # 2. Match and Score
            ranked_results = matcher.rank_advisors(query, parsed_needs, top_k=3)
            
            # 3. Generate Rationales
            final_recommendations = generator.generate_recommendation_reasoning(query, parsed_needs, ranked_results)
            
            st.subheader("🏆 推薦名單 (Top Recommendations)")
            for i, rec in enumerate(final_recommendations):
                st.markdown("---")
                cols = st.columns([1, 2])
                
                with cols[0]:
                    st.metric(label=f"#{i+1} 綜合配對分數 (Match Score)", value=f"{rec.match_score:.2f}")
                    st.markdown(f"**姓名**: {rec.advisor.name}")
                    st.markdown(f"**專長 (Expertise)**: {', '.join(rec.advisor.expertise)}")
                    st.markdown(f"**客群 (Clients)**: {', '.join(rec.advisor.target_clients)}")
                    st.markdown(f"**風格 (Style)**: {rec.advisor.communication_style}")
                    
                with cols[1]:
                    st.markdown("**🧠 推薦理由 (Rationale):**")
                    st.write(rec.rationale)
                    
                    if rec.citations:
                        st.markdown("**📝 文本依據 (Citations from Source):**")
                        for cit in rec.citations:
                            st.info(f'"{cit}"')
