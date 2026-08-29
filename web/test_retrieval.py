from pathlib import Path
import unittest

from web.agent import generate_literature_review
from web.retrieval import (
    PaperRetriever,
    SearchHit,
    build_link_action,
    build_reference_list,
    load_papers,
)


class _FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = type("Message", (), {"content": "综述测试结果 [P1]"})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


class PaperRetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = Path(__file__).resolve().parent / "papers.xlsx"
        cls.data = load_papers(data_path)
        cls.retriever = PaperRetriever(cls.data)

    def test_database_shape_and_columns(self):
        self.assertEqual(len(self.data), 3334)
        self.assertFalse(self.data["标题"].str.strip().eq("").any())

    def test_blank_query_returns_no_hits(self):
        self.assertEqual(self.retriever.search("   "), [])

    def test_claim_search_is_ranked_and_relevant(self):
        hits = self.retriever.search("人工智能幻觉会影响新闻真实性", top_k=8)
        self.assertEqual(len(hits), 8)
        self.assertEqual([hit.score for hit in hits], sorted((hit.score for hit in hits), reverse=True))
        self.assertEqual(len({hit.title for hit in hits}), len(hits))
        combined = " ".join(hit.title + hit.keywords for hit in hits[:5])
        self.assertTrue("幻觉" in combined or "新闻" in combined)

    def test_transient_cnki_link_falls_back_to_title_search(self):
        action = build_link_action(
            "生成式人工智能驱动新闻编辑流程重构与治理",
            "https://kns.cnki.net/kcms2/article/abstract?v=expired-token&language=CHS",
        )
        self.assertTrue(action.uses_title_search)
        self.assertEqual(action.label, "在知网搜索")
        self.assertIn("kns.cnki.net/kns8s/defaultresult/index?kw=", action.url)

    def test_stable_doi_link_remains_direct(self):
        stable = "https://link.cnki.net/doi/10.1234/example"
        action = build_link_action("示例论文", stable)
        self.assertFalse(action.uses_title_search)
        self.assertEqual(action.url, stable)

    def test_reference_list_uses_retrieved_paper_fields(self):
        hit = SearchHit(
            paper_id=1,
            score=0.8,
            title="示例论文",
            authors="张三",
            journal="示例期刊",
            year="2026",
            abstract="摘要",
            keywords="关键词",
            link="https://example.com/paper",
        )
        references = build_reference_list([hit])
        self.assertIn("[P1] 示例论文", references)
        self.assertIn("张三，示例期刊，2026", references)
        self.assertIn("https://example.com/paper", references)

    def test_literature_review_is_grounded_in_numbered_candidates(self):
        hit = SearchHit(
            paper_id=1,
            score=0.8,
            title="示例论文",
            authors="张三",
            journal="示例期刊",
            year="2026",
            abstract="示例摘要",
            keywords="人工智能",
            link="https://example.com/paper",
        )
        client = _FakeClient()
        result = generate_literature_review(client, "研究问题", [hit])
        request = client.completions.request
        self.assertEqual(result, "综述测试结果 [P1]")
        self.assertEqual(request["temperature"], 0.2)
        self.assertIn("[P1] 标题：示例论文", request["messages"][1]["content"])
        self.assertIn("唯一资料来源", request["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
