"""Local paper retrieval for Chinese claims and research questions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


REQUIRED_COLUMNS = ["标题", "作者", "发表期刊", "时间", "摘要", "关键词", "链接"]


@dataclass(frozen=True)
class SearchHit:
    paper_id: int
    score: float
    title: str
    authors: str
    journal: str
    year: str
    abstract: str
    keywords: str
    link: str


@dataclass(frozen=True)
class LinkAction:
    url: str
    label: str
    uses_title_search: bool


def load_papers(path: str | Path) -> pd.DataFrame:
    """Load the source workbook without modifying it."""
    data = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"论文库缺少字段：{', '.join(missing)}")

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    for column in ["标题", "作者", "发表期刊", "摘要", "关键词", "链接"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["链接"] = data["链接"].str.replace(r"\s+", "", regex=True)
    data["年份"] = data["时间"].map(_format_year)
    data["时间"] = data["时间"].map(_format_date)
    return data


def _format_year(value: object) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "year"):
        return str(value.year)
    match = re.search(r"(?:19|20)\d{2}", str(value))
    return match.group(0) if match else ""


def _format_date(value: object) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()


def _document_text(row: pd.Series) -> str:
    # Repeating title and keywords gives the most descriptive fields extra weight.
    return " ".join(
        [
            row["标题"],
            row["标题"],
            row["关键词"],
            row["关键词"],
            row["摘要"],
            row["作者"],
            row["发表期刊"],
        ]
    )


def build_link_action(title: str, link: str) -> LinkAction:
    """Use stable links directly and replace expiring CNKI links with title search."""
    clean_link = re.sub(r"\s+", "", str(link or ""))
    parsed = urlparse(clean_link)
    query = parse_qs(parsed.query)
    host = parsed.netloc.lower()

    is_web_url = parsed.scheme in {"http", "https"} and bool(host)
    is_transient_cnki = (
        host == "kns.cnki.net"
        and "/kcms2/article/abstract" in parsed.path
        and "v" in query
    )
    is_known_malformed = host in {"link.cnki.netdoi", "link.cnlc.net"}

    if is_web_url and not is_transient_cnki and not is_known_malformed:
        return LinkAction(clean_link, "查看原文", False)

    search_url = "https://kns.cnki.net/kns8s/defaultresult/index?kw=" + quote(
        title.strip(), safe=""
    )
    return LinkAction(search_url, "在知网搜索", True)


def build_reference_list(hits: list[SearchHit]) -> str:
    """Build references from database fields so the model cannot invent them."""
    lines = ["## 参考文献（当前检索结果）", ""]
    for index, hit in enumerate(hits, start=1):
        publication = "，".join(part for part in [hit.journal, hit.year] if part)
        details = "，".join(part for part in [hit.authors, publication] if part)
        action = build_link_action(hit.title, hit.link)
        lines.append(f"- [P{index}] {hit.title}。{details}。[原文或检索入口]({action.url})")
    return "\n".join(lines)


class PaperRetriever:
    """Deployment-friendly Chinese retrieval using character n-gram TF-IDF.

    Character n-grams work without a tokenizer and handle long Chinese claims,
    abbreviations, and partial word overlap better than exact substring search.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)
        documents = self.data.apply(_document_text, axis=1).tolist()
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 8) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.matrix).ravel()
        # Abstract-backed records are more useful for evidence analysis.
        scores[self.data["摘要"].eq("").to_numpy()] *= 0.45

        # Small, interpretable bonuses for direct matches in high-value fields.
        compact_query = re.sub(r"\s+", "", query.lower())
        for index, row in self.data.iterrows():
            title = re.sub(r"\s+", "", row["标题"].lower())
            keywords = re.sub(r"\s+", "", row["关键词"].lower())
            if compact_query and compact_query in title:
                scores[index] += 0.12
            if compact_query and compact_query in keywords:
                scores[index] += 0.08

        top_k = max(1, min(int(top_k), len(self.data)))
        # Retrieve extra rows because the source workbook contains duplicate titles.
        candidate_count = min(len(self.data), max(top_k * 5, top_k))
        candidate_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        ordered_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        hits: list[SearchHit] = []
        seen_titles: set[str] = set()
        for index in ordered_indices:
            row = self.data.iloc[int(index)]
            normalized_title = re.sub(r"\s+", "", row["标题"].lower())
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)
            hits.append(
                SearchHit(
                    paper_id=int(index) + 1,
                    score=float(scores[index]),
                    title=row["标题"],
                    authors=row["作者"] or "作者信息缺失",
                    journal=row["发表期刊"] or "期刊信息缺失",
                    year=row["年份"],
                    abstract=row["摘要"] or "摘要缺失",
                    keywords=row["关键词"] or "关键词缺失",
                    link=row["链接"],
                )
            )
            if len(hits) == top_k:
                break
        return hits
