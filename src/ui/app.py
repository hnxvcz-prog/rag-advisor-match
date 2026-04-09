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
    # Dynamically resolve paths relative to this script
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    processed_dir = PROJECT_ROOT / "data" / "processed"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    
    docs = []
    
    # Attempt to load processed JSONs if they exist
    if processed_dir.exists():
        files = list(processed_dir.glob("*.json"))
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    docs.append(AdvisorDocument(**data))
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    
    # If no processed data found, fall back to parsing raw Word files
    if not docs:
        st.info("Parsing raw Word files...")
        if not raw_dir.exists():
            st.error(f"Raw data directory not found at {raw_dir}")
            return None, None, None, [], [], [], [], []
            
        parser = DocumentParser(raw_dir=str(raw_dir), processed_dir=str(processed_dir))
        docs = parser.process_all()
        if not docs:
            st.error("No documents found in raw data directory.")
            return None, None, None, [], [], [], [], []
            
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

st.title("🤝 RAG 理專智能媒合系統 v2.6")
st.caption("核心更新：雙階段 RAG (自傳+標籤) 50/50 權重融合、客戶特徵多維度匹配")

# We pass a version string to force-invalidate st.cache_resource if we update it
query_parser, matcher, generator, docs, all_expertise, all_clients, all_styles, all_branches = load_system(version_tag="v2.8.5-sync-fix")

# --- UI Sidebar for Tag Selection ---
st.sidebar.header("🔍 理專需求型態")
st.sidebar.markdown("清楚描述您的需求，或直接在下方選擇條件。")

selected_branch = st.sidebar.selectbox("🏠 所在分行 (硬過濾)", ["所有分行"] + all_branches)
selected_clients = st.sidebar.selectbox("👥 您偏好哪種服務類型的理專", ["不限", "資產配置型:擅長根據您的生命週期建立長期資產配置組合", "市場領航型:擅長解讀市場資訊和新聞", "研究專家型:邏輯嚴密，提供詳盡的數據及研究報告支撐", "私人管家型:細膩且主動，配息提醒、資產報告、VIP活動不遺漏", "專業經理型:擅長財富傳承及退休規畫安排，整合不同資源"])
selected_style = st.sidebar.selectbox("💬 您偏好哪種溝通類型的理專", ["不限"] + all_styles)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 財富管理需求")
selected_expertise = st.sidebar.multiselect("💡 您目前想了解的財管服務", all_expertise)
selected_exp = st.sidebar.selectbox("⏳ 您目前的投資經驗", ["不限", "1年以下", "1~3年", "3~5年", "5年以上"])
selected_products = st.sidebar.multiselect("📦 曾接觸商品", ["定存", "外匯", "基金", "ETF", "股票", "債券", "結構型商品", "保險"])
selected_alloc = st.sidebar.multiselect("📊 目前資產配置 (最重兩項)", ["現金/存款", "基金/ETF", "股票", "債券/固定收益商品", "結構型商品", "保險", "尚未明確配置"], max_selections=2)
selected_scale = st.sidebar.selectbox("💰 預計管理資產規模", ["不限", "300~1000萬", "1000~3000萬", "3000萬以上"])

if docs:
    st.success(f"系統已準備就緒，共載入 **{len(docs)}** 份理專檔案。")
    
    query = st.text_input("描述您滿意或不滿意的服務經歷：", placeholder="範例：我在尋找一位能為高資產客群進行退休規劃，且溝通風格溫和的理專")
    
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
                current_expertise = parsed_needs.expertise_needed or []
                parsed_needs.expertise_needed = list(set(current_expertise + selected_expertise))
                
            if selected_clients != "不限":
                current_clients = parsed_needs.target_clients_needed or []
                parsed_needs.target_clients_needed = list(set(current_clients + [selected_clients]))
                
            if selected_style != "不限":
                parsed_needs.communication_preference = selected_style
                
            if selected_exp != "不限":
                parsed_needs.investment_experience = selected_exp
                
            if selected_products:
                current_products = parsed_needs.products_touched or []
                parsed_needs.products_touched = list(set(current_products + selected_products))
                
            if selected_alloc:
                current_alloc = parsed_needs.asset_allocation or []
                parsed_needs.asset_allocation = list(set(current_alloc + selected_alloc))
                
            if selected_scale != "不限":
                parsed_needs.asset_scale = selected_scale

            with st.expander("🔍 綜合需求解析結果"):
                st.json(parsed_needs.model_dump())
            
            # 1.5 Gatekeeper Check (Only if query was non-empty)
            is_relevant = getattr(parsed_needs, 'is_relevant', True)
            if query and not is_relevant:
                st.warning("⚠️ 系統偵測到無關查詢")
                msg = getattr(parsed_needs, 'guidance_message', None)
                if not msg or len(msg.strip()) < 5:
                    msg = "您的查詢似乎與理財顧問或投資規劃無關。請試著描述您理想的理專特質（例如：溝通親切、專業穩健）或目前的財務需求（例如：退休規劃、資產傳承）。"
                st.info(f"💡 友善提醒：{msg}")
                st.stop()
                
            # 2. Match and Score
            bio_raw, tags_raw, ranked_results = matcher.rank_advisors(effective_query, parsed_needs, top_k=3)
            
            with st.expander("📊 第一階段：雙路徑語意檢索初步得分 (自傳 vs 標籤)"):
                st.markdown("系統同步檢索兩個向量空間，最終得分為兩者平均值（各佔 50%）：")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📄 自傳檢索路徑 (Bio)**")
                    for doc, sem_score in bio_raw:
                        st.write(f"- **{doc.profile.name}** ({sem_score:.4f})")
                with c2:
                    st.markdown("**🏷️ 標籤檢索路徑 (Tags)**")
                    for doc, sem_score in tags_raw:
                        st.write(f"- **{doc.profile.name}** ({sem_score:.4f})")
                    
            # 3. Generate Rationales
            final_recommendations = generator.generate_recommendation_reasoning(query, parsed_needs, ranked_results)
            
            st.subheader("🏆 最佳推薦名單")
            for i, rec in enumerate(final_recommendations):
                st.markdown("---")
                cols = st.columns([1, 2])
                
                with cols[0]:
                    st.metric(label=f"#{i+1} 綜合配對得分 (0-100)", value=f"{rec.match_score:.1f}")
                    st.markdown(f"**姓名**: {rec.advisor.name}")
                    st.markdown(f"**專長**: {', '.join(rec.advisor.expertise)}")
                    st.markdown(f"**熟悉客群**: {', '.join(rec.advisor.target_clients)}")
                    st.markdown(f"**溝通風格**: {rec.advisor.communication_style}")
                    
                with cols[1]:
                    st.markdown("**🧠 推薦理由:**")
                    st.write(rec.rationale)
                    
                    if rec.citations:
                        st.markdown("**📝 關鍵引用 (來自自傳):**")
                        for cit in rec.citations:
                            st.info(f'"{cit}"')
