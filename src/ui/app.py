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

st.title("🤝 RAG 理專智能媒合系統 v1.9.1")
st.caption("更新：生成器解析模型 Hotfix")

# We pass a version string to force-invalidate st.cache_resource if we update it
query_parser, matcher, generator, docs_count = load_system(version_tag="v1.9.1-hotfix")

if docs_count:
    st.success(f"系統已準備就緒，共載入 **{docs_count}** 份理專檔案。")
    
    query = st.text_input("請描述您的理財需求：", placeholder="範例：我在尋找一位能為高資產客群進行退休規劃，且溝通風格溫和的理專")
    
    if st.button("開始搜尋") and query:
        with st.spinner("正在分析您的需求並配對合適的理財顧問..."):
            # 1. Parse Query
            parsed_needs = query_parser.parse_query(query)
            with st.expander("🔍 查詢解析結果"):
                st.json(parsed_needs.model_dump())
                
            # 2. Match and Score
            phase1_results, ranked_results = matcher.rank_advisors(query, parsed_needs, top_k=3)
            
            with st.expander("📊 第一階段：語意檢索初選名單 (Top 10)"):
                st.markdown("僅依賴「自傳全文」的語意相似度撈出 10 人候選名單：")
                for doc, sem_score in phase1_results:
                    st.write(f"- **{doc.profile.name}** (純語意關聯度分數: {sem_score:.4f})")
                    
            # 3. Generate Rationales
            final_recommendations = generator.generate_recommendation_reasoning(query, parsed_needs, ranked_results)
            
            st.subheader("🏆 最佳推薦名單")
            for i, rec in enumerate(final_recommendations):
                st.markdown("---")
                cols = st.columns([1, 2])
                
                with cols[0]:
                    st.metric(label=f"#{i+1} 綜合配對分數", value=f"{rec.match_score:.2f}")
                    st.markdown(f"**姓名**: {rec.advisor.name}")
                    st.markdown(f"**專長**: {', '.join(rec.advisor.expertise)}")
                    st.markdown(f"**熟悉客群**: {', '.join(rec.advisor.target_clients)}")
                    st.markdown(f"**溝通風格**: {rec.advisor.communication_style}")
                    
                with cols[1]:
                    st.markdown("**🧠 推薦理由:**")
                    st.write(rec.rationale)
                    
                    if rec.citations:
                        st.markdown("**📝 原文依據:**")
                        for cit in rec.citations:
                            st.info(f'"{cit}"')
