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
    
    # Extract unique categories for UI
    all_expertise = sorted(list(set(exp for doc in docs for exp in doc.profile.expertise if exp != "未提供")))
    all_clients = sorted(list(set(client for doc in docs for client in doc.profile.target_clients if client != "未提供")))
    all_styles = sorted(list(set(doc.profile.communication_style for doc in docs if doc.profile.communication_style != "未提供")))
    all_branches = sorted(list(set(doc.profile.branch for doc in docs if doc.profile.branch != "未提供")))
    
    return query_parser, matcher, generator, docs, all_expertise, all_clients, all_styles, all_branches

st.title("🤝 RAG 理專智能媒合系統 v2.0")
st.caption("更新：動態標籤篩選、分行硬過濾與語意混合檢索架構")

# We pass a version string to force-invalidate st.cache_resource if we update it
query_parser, matcher, generator, docs, all_expertise, all_clients, all_styles, all_branches = load_system(version_tag="v2.0-ui-tags")

# --- UI Sidebar for Tag Selection ---
st.sidebar.header("🔍 精確條件篩選")
st.sidebar.markdown("您可以手動指定特定條件，或留空讓 AI 全自動解析。")

selected_branch = st.sidebar.selectbox("🏠 經管分行 (硬過濾)", ["所有分行"] + all_branches)
selected_expertise = st.sidebar.multiselect("💡 專業領域", all_expertise)
selected_clients = st.sidebar.multiselect("👥 熟悉客群", all_clients)
selected_style = st.sidebar.selectbox("💬 溝通風格", ["不限"] + all_styles)

if docs:
    st.success(f"系統已準備就緒，共載入 **{len(docs)}** 份理專檔案。")
    
    query = st.text_input("請描述您的理財需求：", placeholder="範例：我在尋找一位能為高資產客群進行退休規劃，且溝通風格溫和的理專")
    
    if st.button("開始搜尋") and (query or selected_expertise or selected_clients or selected_branch != "所有分行"):
        # Handle empty query gracefully if tags are selected
        effective_query = query if query else "搜尋符合特定條件的理專"
        
        with st.spinner("正在分析您的需求並配對合適的理財顧問..."):
            # 1. Parse Query
            parsed_needs = query_parser.parse_query(effective_query)
            
            # --- UI Selection Merging Logic ---
            # 1.1 Override/Merge with UI Selections
            if selected_branch != "所有分行":
                parsed_needs.branch_needed = selected_branch
                
            if selected_expertise:
                # Union of LLM extracted and UI selected, de-duplicated
                parsed_needs.expertise_needed = list(set(parsed_needs.expertise_needed + selected_expertise))
                
            if selected_clients:
                parsed_needs.target_clients_needed = list(set(parsed_needs.target_clients_needed + selected_clients))
                
            if selected_style != "不限":
                parsed_needs.communication_preference = selected_style

            with st.expander("🔍 綜合需求解析結果"):
                st.json(parsed_needs.model_dump())
            
            # 1.5 Gatekeeper Check (Only if query was non-empty)
            if query and not getattr(parsed_needs, 'is_relevant', True):
                st.warning("⚠️ 系統偵測到無關查詢")
                st.info(getattr(parsed_needs, 'guidance_message', "您的查詢似乎與理財顧問無關。請輸入與財務管理、投資或您理想的理專特質相關的描述。"))
                st.stop()
                
            # 2. Match and Score
            phase1_results, ranked_results = matcher.rank_advisors(effective_query, parsed_needs, top_k=3)
            
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
