from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from .agent import analyze_claim, create_client, generate_literature_review, read_api_key
    from .retrieval import (
        PaperRetriever,
        SearchHit,
        build_link_action,
        build_reference_list,
        load_papers,
    )
except ImportError:  # Streamlit runs app.py directly from the web directory.
    from agent import analyze_claim, create_client, generate_literature_review, read_api_key
    from retrieval import (
        PaperRetriever,
        SearchHit,
        build_link_action,
        build_reference_list,
        load_papers,
    )


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "papers.xlsx"

st.set_page_config(
    page_title="论点驱动的文献推荐智能体",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e4e9f2;
        padding: 0.8rem 1rem; border-radius: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_papers(DATA_PATH)


@st.cache_resource(show_spinner="正在建立本地检索索引……")
def get_retriever() -> PaperRetriever:
    return PaperRetriever(get_data())


def render_hit(hit: SearchHit, rank: int) -> None:
    with st.container(border=True):
        st.markdown(f"#### {rank}. {hit.title}")
        meta = " · ".join(part for part in [hit.authors, hit.journal, hit.year] if part)
        st.caption(meta)
        st.write(hit.abstract)
        left, middle, right = st.columns([2, 2, 1])
        left.caption(f"关键词：{hit.keywords}")
        middle.caption(f"本地检索分数：{hit.score:.3f}")
        action = build_link_action(hit.title, hit.link)
        right.link_button(action.label, action.url, width="stretch")
        if action.uses_title_search:
            right.caption("原链接为临时或异常地址")


def run_search(claim: str, top_k: int) -> list[SearchHit]:
    hits = get_retriever().search(claim, top_k=top_k)
    st.session_state["claim"] = claim
    st.session_state["hits"] = hits
    st.session_state.pop("analysis", None)
    st.session_state.pop("literature_review", None)
    return hits


data = get_data()
api_key = read_api_key(st.secrets)
client = create_client(api_key)

st.sidebar.title("📚 文献推荐智能体")
page = st.sidebar.radio("功能", ["论点检索", "数据概览", "系统说明"])
st.sidebar.divider()
st.sidebar.caption(f"本地论文库：{len(data):,} 篇")
if client is None:
    st.sidebar.info("当前为本地检索模式。配置服务器端密钥后可启用证据化智能分析。")
else:
    st.sidebar.success("智能分析已启用")

if page == "论点检索":
    st.title("论点驱动的文献推荐智能体")
    st.write("输入一个需要论证、反驳或补充证据的观点。系统先检索本地论文，再让大模型只依据命中的摘要进行分析。")

    claim = st.text_area(
        "你的论点或研究问题",
        value=st.session_state.get("claim", "生成式人工智能可能削弱新闻编辑的专业把关能力"),
        height=110,
        placeholder="例如：算法推荐会加剧新闻用户的信息茧房。",
    )
    top_k = st.slider("候选论文数量", min_value=5, max_value=15, value=8)
    search_col, analyze_col = st.columns(2)

    search_clicked = search_col.button("只检索论文", type="primary", width="stretch")
    analyze_clicked = analyze_col.button(
        "检索并生成证据分析",
        width="stretch",
        disabled=client is None,
        help="需要在服务器端配置 DEEPSEEK_API_KEY" if client is None else None,
    )

    if (search_clicked or analyze_clicked) and not claim.strip():
        st.warning("请先输入一个论点或研究问题。")
    elif search_clicked or analyze_clicked:
        hits = run_search(claim, top_k)
        if analyze_clicked and hits:
            with st.spinner("智能体正在阅读候选论文摘要并整理证据……"):
                try:
                    st.session_state["analysis"] = analyze_claim(client, claim, hits)
                except Exception as exc:
                    st.error(f"智能分析调用失败：{exc}")

    hits = st.session_state.get("hits", [])
    analysis = st.session_state.get("analysis")
    literature_review = st.session_state.get("literature_review")

    if analysis:
        st.subheader("证据化分析")
        st.info("以下分析只允许引用下方本地论文候选；编号 P1、P2……与候选列表顺序一致。")
        st.markdown(analysis)

    if hits:
        st.subheader(f"本地论文候选（{len(hits)} 篇）")
        st.caption("检索分数只表示文本相关性，不代表论文质量，也不等于论文一定支持该论点。")

        review_col, note_col = st.columns([1, 2])
        review_clicked = review_col.button(
            "✨ 一键生成文献综述",
            type="primary",
            width="stretch",
            disabled=client is None,
            help="需要在服务器端配置 DEEPSEEK_API_KEY" if client is None else None,
        )
        note_col.info("综述仅依据当前候选论文的标题、关键词和摘要生成；正式写作前请核对全文。")

        if review_clicked:
            with st.spinner("智能体正在比较论文观点并撰写综述初稿……"):
                try:
                    searched_claim = st.session_state.get("claim", claim)
                    review_body = generate_literature_review(client, searched_claim, hits)
                    st.session_state["literature_review"] = (
                        f"{review_body}\n\n{build_reference_list(hits)}"
                    )
                    literature_review = st.session_state["literature_review"]
                except Exception as exc:
                    st.error(f"文献综述生成失败：{exc}")

        if literature_review:
            st.markdown(literature_review)
            st.download_button(
                "下载综述（Markdown）",
                data=literature_review,
                file_name="文献综述初稿.md",
                mime="text/markdown",
                width="stretch",
            )

        st.markdown("### 候选论文详情")
        for rank, hit in enumerate(hits, start=1):
            render_hit(hit, rank)

elif page == "数据概览":
    st.title("论文库概览")
    missing_abstracts = int((data["摘要"] == "").sum())
    missing_keywords = int((data["关键词"] == "").sum())
    duplicate_titles = int(data.duplicated(subset=["标题"]).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("论文总数", f"{len(data):,}")
    col2.metric("缺少摘要", missing_abstracts)
    col3.metric("缺少关键词", missing_keywords)
    col4.metric("重复标题记录", duplicate_titles)

    left, right = st.columns(2)
    journal_counts = data.loc[data["发表期刊"] != "", "发表期刊"].value_counts().head(15)
    journal_frame = journal_counts.rename_axis("期刊").reset_index(name="论文数")
    left.plotly_chart(
        px.bar(journal_frame, x="论文数", y="期刊", orientation="h", title="收录最多的 15 种期刊"),
        width="stretch",
    )

    year_counts = data.loc[data["年份"] != "", "年份"].value_counts().sort_index()
    year_frame = year_counts.rename_axis("年份").reset_index(name="论文数")
    right.plotly_chart(
        px.line(year_frame, x="年份", y="论文数", markers=True, title="论文年份分布"),
        width="stretch",
    )

    with st.expander("查看论文数据"):
        st.dataframe(data, width="stretch", hide_index=True)

elif page == "系统说明":
    st.title("这个系统如何工作")
    st.markdown(
        """
### 当前流程

1. **输入论点**：用户用自然语言提出需要寻找证据的观点。
2. **本地召回**：系统把中文拆成连续字符片段，用 TF-IDF 从 3,334 篇论文中找候选。
3. **智能筛选**：配置密钥后，大模型阅读候选摘要，判断它们是支持、限制还是仅提供背景。
4. **证据化输出**：所有论文判断必须引用候选编号；摘要证据不足时必须明确说明。
5. **综述初稿**：检索完成后可一键综合候选论文的共同点、差异、局限和未来研究方向，并下载 Markdown 文件。

### 为什么先这样做

这个版本不需要下载体积很大的向量模型，普通电脑和轻量云平台都能运行，适合作为可验证的第一版。下一阶段会加入语义向量、混合排序和人工评测集，再比较升级前后的检索质量。

### 目前的边界

- 系统依据标题、关键词和摘要，不等于阅读了论文全文。
- 自动生成的文献综述是辅助写作初稿，不能替代对论文全文的阅读、引用核验和人工修改。
- 文本相关性分数不是论文质量评分，也不能自动证明因果关系。
- 数据库存在少量缺失字段和重复标题，后续需要清洗并保留可追溯记录。
- 大模型可能误判，因此界面始终同时展示原始摘要和原文链接，方便人工核验。
"""
    )

st.divider()
st.caption("张紫翔、白耘赫 · 论点驱动的文献推荐智能体（教学原型）")
