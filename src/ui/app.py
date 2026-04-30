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

# --- UI Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #eee;
    }
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        margin-bottom: 30px;
        border-left: 5px solid #007bff;
    }
    .metric-box {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "input"
if "results_data" not in st.session_state:
    st.session_state.results_data = None

def switch_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# --- Load Data ---
query_parser, matcher, generator, docs, all_expertise, all_clients, all_styles, all_branches = load_system(version_tag="v2.8.5-sync-fix")

import base64

@st.cache_data
def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def icon_header(icon_name, title):
    # Dynamically resolve pngs path
    png_path = Path(__file__).parent.parent.parent / "pngs" / icon_name
    if png_path.exists():
        b64 = get_base64_image(png_path)
        return f'<div style="display: flex; align-items: center; margin-bottom: 10px;"><img src="data:image/png;base64,{b64}" width="28" style="margin-right: 10px;"><h3 style="margin: 0;">{title}</h3></div>'
    return f"<h3>{title}</h3>"

def icon_label(icon_name, title):
    # Dynamically resolve pngs path
    png_path = Path(__file__).parent.parent.parent / "pngs" / icon_name
    if png_path.exists():
        b64 = get_base64_image(png_path)
        return f'''
            <div style="display: flex; align-items: center; margin-bottom: 5px; margin-top: 12px;">
                <img src="data:image/png;base64,{b64}" width="20" style="margin-right: 8px;">
                <span style="font-weight: bold; font-size: 14px; color: #31333F;">{title}</span>
            </div>
        '''
    return f'<div style="font-weight: bold; font-size: 14px; margin-top: 10px;">{title}</div>'

# --- Main App Logic ---

if st.session_state.page == "input":
    st.title(" RAG 理專智能媒合系統")
    st.markdown("### 📋 填寫您的理財需求")
    st.caption("請填寫以下資訊，系統將透過雙階段 RAG 引擎為您精準配對最合適的理財顧問。")

    if docs:
        st.success(f"系統已準備就緒，共載入 **{len(docs)}** 份理專檔案。")

    # Group: All Preferences in 3 Columns
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(icon_header("home.png", "基本偏好"), unsafe_allow_html=True)
            
            st.markdown(icon_label("home.png", "所在分行 (硬過濾)"), unsafe_allow_html=True)
            sel_branch = st.selectbox("分行", ["所有分行"] + all_branches, label_visibility="collapsed")
            
            st.markdown(icon_label("handshake.png", "您偏好哪種服務類型的理專"), unsafe_allow_html=True)
            sel_clients = st.selectbox("服務類型", ["不限", "資產配置型:擅長根據您的生命週期建立長期資產配置組合", "市場領航型:擅長解讀市場資訊與新聞", "研究專家型:邏輯嚴密，提供詳盡的數據及研究報告支撐", "私人管家型:細膩且主動，配息提醒、資產報告、VIP活動不遺漏", "專業經理型:擅長財富傳傳承及退休規畫安排，整合不同資源"], label_visibility="collapsed")
            
            st.markdown(icon_label("comment.png", "您偏好哪種溝通類型的理專"), unsafe_allow_html=True)
            sel_style = st.selectbox("溝通風格", ["不限"] + all_styles, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(icon_header("user.png", "財富管理需求"), unsafe_allow_html=True)
            
            st.markdown(icon_label("dollar.png", "您目前想了解的財管服務"), unsafe_allow_html=True)
            sel_expertise = st.multiselect("財管服務", all_expertise, label_visibility="collapsed")
            
            st.markdown(icon_label("piggy-bank.png", "預計管理資產規模"), unsafe_allow_html=True)
            sel_scale = st.selectbox("資產規模", ["不限", "300~1000萬", "1000~3000萬", "3000萬以上"], label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(icon_header("chart-histogram.png", "投資背景"), unsafe_allow_html=True)
            
            st.markdown(icon_label("crown.png", "您目前的投資經驗"), unsafe_allow_html=True)
            sel_exp = st.selectbox("投資經驗", ["不限", "1年以下", "1~3年", "3~5年", "5年以上"], label_visibility="collapsed")
            
            st.markdown(icon_label("shopping-cart-add.png", "曾接觸商品"), unsafe_allow_html=True)
            sel_products = st.multiselect("接觸商品", ["定存", "外匯", "基金", "ETF", "股票", "債券", "結構型商品", "保險"], label_visibility="collapsed")
            
            st.markdown(icon_label("apps-add.png", "目前資產配置 (最重兩項)"), unsafe_allow_html=True)
            sel_alloc = st.multiselect("資產配置", ["現金/存款", "基金/ETF", "股票", "債券/固定收益商品", "結構型商品", "保險", "尚未明確配置"], max_selections=2, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

    # Open-ended questions
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(icon_header("pencil.png", "開放式需求描述"), unsafe_allow_html=True)
    
    st.markdown(icon_label("sparkles.png", "您心目中的理專，應該具備哪些特質或特長？"), unsafe_allow_html=True)
    q1 = st.text_area("特質特長", placeholder="範例：我在尋找一位能為高資產客群進行退休規劃，且溝通風格溫和的理專。", height=100, label_visibility="collapsed")
    
    st.markdown(icon_label("edit.png", "描述您滿意或不滿意的服務經歷："), unsafe_allow_html=True)
    q2 = st.text_area("服務經歷", placeholder="範例：之前的理專很少主動聯繫，常常錯過市場機會。", height=100, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Submit button
    if st.button("開始精準媒合"):
        combined_q = ""
        if q1: combined_q += f"理想特質：{q1}\n"
        if q2: combined_q += f"服務經歷：{q2}"
        combined_q = combined_q.strip()
        
        # Explicitly mark empty descriptions to avoid LLM over-inference
        effective_q = combined_q if combined_q else "[無手寫描述]"
        
        with st.status("正在進行深度媒合分析...", expanded=True) as status:
            st.write("1️⃣ 正在解析您的口語需求...")
            parsed = query_parser.parse_query(effective_q)
            
            # Merge UI Selections
            if sel_branch != "所有分行": parsed.branch_needed = sel_branch
            if sel_expertise: parsed.expertise_needed = list(set((parsed.expertise_needed or []) + sel_expertise))
            if sel_clients != "不限": parsed.target_clients_needed = list(set((parsed.target_clients_needed or []) + [sel_clients]))
            if sel_style != "不限": parsed.communication_preference = sel_style
            if sel_exp != "不限": parsed.investment_experience = sel_exp
            if sel_products: parsed.products_touched = list(set((parsed.products_touched or []) + sel_products))
            if sel_alloc: parsed.asset_allocation = list(set((parsed.asset_allocation or []) + sel_alloc))
            if sel_scale != "不限": parsed.asset_scale = sel_scale

            # Gatekeeper
            is_relevant = getattr(parsed, 'is_relevant', True)
            if combined_q and not is_relevant:
                status.update(label="❌ 偵測到無關查詢", state="error")
                st.error("⚠️ 系統偵測到無關查詢")
                msg = getattr(parsed, 'guidance_message', "您的查詢似乎與理財顧問無關，請重新描述。")
                st.info(f"💡 友善提醒：{msg}")
            else:
                st.write("2️⃣ 執行雙路徑向量檢索 (Bio & Tags)...")
                b_raw, t_raw, r_results = matcher.rank_advisors(effective_q, parsed, top_k=3)
                
                st.write("3️⃣ 正在生成個人化推薦理由...")
                f_recs = generator.generate_recommendation_reasoning(combined_q, parsed, r_results)
                
                # Save to session state
                st.session_state.results_data = {
                    "combined_query": combined_q,
                    "parsed_needs": parsed,
                    "bio_raw": b_raw,
                    "tags_raw": t_raw,
                    "ranked_results": r_results,
                    "final_recommendations": f_recs
                }
                status.update(label="✅ 媒合完成！", state="complete", expanded=False)
                switch_page("results")

elif st.session_state.page == "results":
    data = st.session_state.results_data
    if not data:
        switch_page("input")

    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ 返回修改"):
            switch_page("input")
    with col_title:
        st.markdown(icon_header("heart.png", "您的理專媒合結果"), unsafe_allow_html=True)

    # 1. Parsed Needs Summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(icon_header("search.png", "您的需求解析摘要"), unsafe_allow_html=True)
    pn = data["parsed_needs"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**📍 偏好分行**: {pn.branch_needed or '不限'}")
    c2.markdown(f"**⏳ 投資經驗**: {pn.investment_experience or '不限'}")
    c3.markdown(f"**💰 資產規模**: {pn.asset_scale or '不限'}")
    st.markdown(f"**💡 專業領域**: {', '.join(pn.expertise_needed) if pn.expertise_needed else '由 AI 自動匹配'}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Path Scores
    with st.expander("📊 雙路徑語意檢索初步得分詳情"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(icon_label("user.png", "自傳檢索路徑 (Bio)"), unsafe_allow_html=True)
            for doc, score in data["bio_raw"]:
                st.write(f"- {doc.profile.name}: `{score:.4f}`")
        with c2:
            st.markdown(icon_label("apps-add.png", "標籤檢索路徑 (Tags)"), unsafe_allow_html=True)
            for doc, score in data["tags_raw"]:
                st.write(f"- {doc.profile.name}: `{score:.4f}`")

    # 3. Final Recommendations
    st.markdown(icon_header("heart.png", "最佳推薦名單"), unsafe_allow_html=True)
    for i, rec in enumerate(data["final_recommendations"]):
        orig = next((x for x in data["ranked_results"] if x[0].profile.advisor_id == rec.advisor.advisor_id), None)
        faiss_s = orig[3] if (orig and len(orig) > 3) else 0.0
        rerank = orig[2] if (orig and len(orig) > 2) else None

        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        cols = st.columns([1, 3])
        
        with cols[0]:
            st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
            st.metric(label=f"#{i+1} 綜合契合度", value=f"{rec.match_score:.1f}")
            st.caption(f"FAISS 初篩: {faiss_s:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"**👤 {rec.advisor.name}**")
            st.markdown(f"**🏢 {rec.advisor.branch}**")
            st.caption(f"風格: {rec.advisor.communication_style}")
            
        with cols[1]:
            st.markdown(icon_label("sparkles.png", "推薦核心理由"), unsafe_allow_html=True)
            st.write(rec.rationale)
            
            if rerank:
                with st.expander("🔬 深度評分細節"):
                    st.write(f"專業契合: {rerank.tag_fit_score} | 軟性契合: {rerank.bio_fit_score}")
                    st.caption(f"*分析: {rerank.reasoning}*")

            if rec.citations:
                st.markdown(icon_label("pencil.png", "關鍵引用"), unsafe_allow_html=True)
                for cit in rec.citations:
                    st.markdown(f'<div style="font-style: italic; color: #666; font-size: 0.9em; margin-bottom: 5px;">"{cit}"</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
