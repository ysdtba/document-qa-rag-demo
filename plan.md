# PLAN: document-qa-rag-demo → HKEX Annual Report QA（港股年报问答）

## 目标
把这个 RAG demo 改成“港股上市公司年报问答”服务，每一步改进都要有评测数字。
最终简历上要能写：建了 N 题的评测集；检索命中率从 X% 提到 Y%；数字题准确率从 A% 提到 B%；已上线 <url>。

## 仓库现状（先核对代码）
- main.py：FastAPI，接口有 `/`、`/health`、`POST /api/chat`（SSE 流式返回：sources、token、[DONE]）
- data_init.py：读 docs/ 下的 md/txt/pdf，切块后建 Chroma + BM25 索引，存到 storage/
- rag_pipeline.py：RRF 融合，重排（RERANK_MODE=mock/local），带引用的流式生成（默认 mock；设了 OPENAI_API_KEY 后调用兼容 OpenAI 的模型）
- 其他文件：demo/index.html、verify_demo.py（冒烟测试）、test_client.py、Dockerfile、.env.example
- 现在的语料只有 docs/synthetic_otc_rules.md
- README 自己列出的缺口：没有检索质量评测，没有鉴权和限流

## 硬性规则（所有阶段都适用）
1. 只用公开数据：从港交所披露易（hkexnews.hk）或公司官网下载的年报 PDF。不用任何雇主或客户的资料。
2. PDF 不进 git：放在 data/raw/，并加进 .gitignore。提交 data/manifest.csv，字段：ticker、公司、财年、语言、来源网址、下载日期、SHA256。
3. 密钥只放 .env，不提交。
4. eval/questions.jsonl 的标准答案由仓库作者写和核对，你不能改。觉得哪题有错，列出来问我。
5. 不许针对具体题目写特判，也不许硬编码答案。
6. 所有分数都由脚本跑出来，存进 eval/results/，不手填。
7. 一次只改一个变量，改完跑一次评测，记下配置和结果。
8. 生成和评判都用 temperature=0，并记录模型名和版本。

## 开始前
先读完整个仓库，告诉我三件事：现在怎么切块，embedding 和重排各用什么模型，PDF 怎么解析。然后列出阶段 1 要新增和修改的文件。等我确认再写代码。

## 阶段 1（第 1 周）：语料 + 评测 + 基线
- 语料：先用 2 份英文年报（例如 0388 香港交易所、1299 友邦保险，各取最近一个财年），之后扩到 4–5 份。
- 解析：按页抽文本。每个 chunk 带 metadata：ticker、company、fiscal_year、page，能识别时再加 section。引用要能显示页码。
- 评测集 eval/questions.jsonl，每行一个 JSON，字段：
  id, ticker, fiscal_year, question, answer, answer_type（number / text / not_found）, unit, currency, gold_pages（页码列表）, split（dev / test）, notes
  第一批 30 题由我写，之后补到 50 题（dev 30、test 20）。其中约 5 题在年报里找不到答案，用来测模型会不会编。
- 评测脚本 eval/run_eval.py，输出这些指标：
  - 检索：Hit@5（前 5 个 chunk 是否覆盖 gold 页）、MRR@10
  - 回答：数字题按数值比较，先统一单位（百万/十亿）和币种（HK$/US$/RMB），容差 0.5%。文字题用 LLM 评判，分“正确 / 部分正确 / 错误”，附理由。not_found 题看是否正确拒答。
  - 忠实度：用 LLM 评判回答里的说法是否都有引用片段支持；再看引用页是否落在 gold_pages 里
  - 延迟 p50/p95，每题的 token 数和费用
  - 每次运行存成 eval/results/<日期>_<run名>.json，并更新 eval/RESULTS.md 里的对比表
- 评判用的模型不能比生成用的弱。我会人工抽查约 20% 的评判结果。
- 产出：基线分数。只换语料、不改算法，作为对照。

## 阶段 2（第 2 周）：提升检索
下面每一项单独做、单独跑评测，结果写进 RESULTS.md 的对比表：
- 按页和段落切块，调整 chunk 大小和重叠
- 表格单独成块：转成 Markdown，同一张表不拆开
- 按 ticker / fiscal_year 做 metadata 过滤
- 重排（基线已开 local 重排，bge-reranker-base），以下每项单独跑评测：
  - 重排开/关对照：新增 RERANK_MODE=none（直接取 RRF 融合后的前 N 个），量化重排本身带来多少提升
  - 重排前候选数 TOP_K：10 → 20 → 50，看 Hit@5、MRR@10 和延迟 p95
  - 重排后送给 LLM 的数量 RERANK_TOP_N：3 → 5 → 8，在上一步最优的 TOP_K 上调，看回答准确率和每题费用
  - 可选：换多语言重排模型 bge-reranker-v2-m3（配合中文提问、查英文年报）
- 可选：支持用中文提问、查英文年报（查询翻译或多语言 embedding）
调参只看 dev 集。test 集只在阶段末跑一次。

## 阶段 3（第 3 周）：Agent 工具查财务数字
- 从合并损益表、资产负债表、现金流量表里抽 10–15 个关键指标（收入、净利润、总资产、总权益、每股盈利、每股股息、经营现金流等），存进 SQLite 或 DuckDB：
  financials(ticker, fiscal_year, statement, line_item, value, unit, currency, source_page)
  抽完把列表给我人工核对。
- 工具两个：
  - lookup_metric(ticker, fiscal_year, metric)：指标名做同义词匹配
  - calculate(expression)：只允许四则运算，用来算增长率和比率
- 用 function calling 路由：数字题走工具，叙述题走 RAG。最多调用 3 次工具，回答必须带来源页码。
- 给工具写单元测试，用我核对过的数字。
- 评测：数字题准确率，工具开、关各跑一次做对比。

## 阶段 4（第 4 周）：上线 + 收尾
- 部署到可公开访问的平台，例如 Hugging Face Spaces（Docker）或 Render，按免费额度选，额度以官网为准。索引在构建时建好，线上不开放上传。
- 防滥用：
  - 按 IP 限流
  - 限制问题长度和 max_tokens
  - 设每日花费上限，超了就返回提示
  - CORS 只放行自己的页面
- 页面上注明数据来源（附年报链接），并写明仅供演示、不构成投资建议。
- README 用英文写，包括：
  - 一句话介绍
  - 架构图（Mermaid）
  - 结果表（基线 → 最终，dev 和 test 分开列）
  - 每项改进带来的提升
  - 复现步骤
  - 已知局限
- GitHub Actions：跑单元测试和 verify_demo.py。评测要花 API 费用，不放进 CI。

## 每个阶段结束时给我
- 改了哪些文件，为什么改
- 评测结果表，和上一次对比
- 你认为下一步最值得做的一件事