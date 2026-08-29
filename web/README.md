# 论点驱动的文献推荐智能体

这是项目的可部署网页版本。用户输入一个论点或研究问题，系统先从本地 3,334 篇论文中检索候选，再由 DeepSeek V4 Flash 依据候选论文的标题、关键词和摘要生成证据化分析或文献综述初稿。

## 第一版架构

```text
用户论点
  ↓
中文字符 TF-IDF 本地检索（无需下载模型）
  ↓
候选论文及原始摘要
  ↓（配置 DeepSeek 密钥后）
证据筛选：支持 / 限制 / 背景
  ↓
带 P1、P2 引用编号的回答
  ↓
一键生成并下载文献综述初稿
```

第一版选择轻量本地检索，是为了让项目先稳定运行和部署。后续应使用人工标注的测试问题比较 TF-IDF、bge-m3 向量检索和混合检索，再决定是否增加大型模型依赖。

## 文件结构

```text
web/
├── app.py                         # Streamlit 页面与交互
├── retrieval.py                   # 本地论文检索工具
├── agent.py                       # 证据化分析与文献综述生成
├── test_retrieval.py              # 检索基线测试
├── papers.xlsx                    # 原始论文库
├── requirements.txt               # 部署依赖
└── .streamlit/
    └── secrets.toml.example       # 密钥配置示例
```

## 本地运行

在 `web` 目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。不配置密钥也可以使用本地检索。

## 启用智能分析和文献综述

先在 DeepSeek 控制台撤销任何曾写进代码的旧密钥并创建新密钥。然后把 `.streamlit/secrets.toml.example` 复制为 `.streamlit/secrets.toml`，只在本机的 `secrets.toml` 中填写：

```toml
DEEPSEEK_API_KEY = "新密钥"
```

不要把真实密钥写入 Python、HTML、Notebook 或 Git 仓库。纯 HTML 页面运行在用户浏览器中，无法安全保存服务端 API Key，因此正式版本统一使用 Streamlit。

## 使用一键文献综述

1. 输入研究问题或论点，候选论文数量建议选择 8–15 篇。
2. 点击“只检索论文”，先检查候选结果是否围绕你的主题。
3. 点击“✨ 一键生成文献综述”。系统会综合当前候选论文，而不是逐篇简单罗列。
4. 检查正文中的 `[P1]`、`[P2]` 是否与下方候选论文一致，再通过“下载综述（Markdown）”保存初稿。

系统只读取数据库中的标题、关键词和摘要，不能代替全文阅读。论文写作中使用综述前，应打开原文核对观点并按学校要求调整引用格式。

## 测试

```bash
python -m unittest test_retrieval.py
```

## 部署到 Streamlit Community Cloud

1. 将项目放入 GitHub 仓库，并确认仓库中没有 `secrets.toml` 或真实密钥。
2. 在 Streamlit Community Cloud 新建应用。
3. 主文件路径选择 `web/app.py`。
4. 在应用的 Secrets 设置中添加 `DEEPSEEK_API_KEY = "新密钥"`。
5. 部署后分别测试“只检索论文”“检索并生成证据分析”和“一键生成文献综述”。

## 下一阶段

1. 清理重复记录和缺失字段，同时保留原始数据副本。
2. 建立 30–50 个真实论点及人工相关性标签，形成检索评测集。
3. 加入 bge-m3 或云端 embedding，与当前 TF-IDF 做混合召回。
4. 增加模型重排、Word 格式导出、用户反馈和调用成本限制。
5. 完成部署验收：安全、准确性、速度、移动端显示和异常处理。
