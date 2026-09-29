# NewsIQ demo script

课程材料没有写视频时长。按下面的镜头说完即可。若 NTULearn 页面另有时限，删掉第 7 镜里的分数，前面六镜不要删。

录屏要有声音。先在项目目录启动页面：`streamlit run app.py`，打开 http://localhost:8501。不要打开 `.env`，不要让密钥出现在屏幕上。

不要提节省了多少时间。不要说问答达到了 80%，不要说抽取达到了 75%。

## 1. 打开页面

屏幕上要有页眉这句：BBC News 2004-2005. For editorial sorting, finding archive articles, and this course demo. Not for automated publishing, fact-checking, or legal review.

说：

> This is NewsIQ. It does three things on one BBC archive from 2004 to 2005: classify an article, answer a question with a citation, and extract structured fields. It is a course demo, not a tool for publishing or fact-checking.

## 2. 分类一篇样例

打开 Classify。点 Load sample article，再点 Classify。屏幕上要出现 business，以及置信度大约 0.98，门槛 0.90。

说：

> This article is classified as business. The confidence is above the threshold, so the label is accepted. The threshold was chosen on the validation split, not on this test.

## 3. 弃权

点 Load a text below the threshold，再点 Classify。屏幕上要出现：Not classified. Hand this article to a person.

说：

> This text is below the threshold. The system does not assign a class. It hands the article to a person.

## 4. 语料里能回答的问题

打开 Ask the archive。点 Load an archive question，再点 Ask。等到答案出现。屏幕上要有 egypt，以及 Passage 1, article 261，还有 This call 的美元金额。

说：

> The archive does contain this. The answer is Egypt, and the citation is article 261. The cost of this call is on the screen.

## 5. 语料里没有的问题

点 Load a question outside the archive，再点 Ask。屏幕上要有：The corpus does not contain sufficient evidence. 以及 best cosine 低于 0.50。This call 应为 USD 0.000000。

说：

> This question is outside the 2004 to 2005 archive. The best passage is below the cutoff, so the generator is not called, and the system says the corpus does not contain sufficient evidence.

## 6. 抽取

打开 Extract。点 Load sample article，再点 Extract。屏幕上要有五字段 JSON，以及 This call 的美元金额。侧栏要能看到本会话次数和 USD 0.05 的上限。

说：

> These are the five fields: people, organisations, locations, dates, and topic. This call's cost is shown here. The session stops at 30 calls or 5 cents.

## 7. 收束，这几句必须说

不要切走页面。页眉仍要留在画面里。

说：

> The corpus is BBC News from 2004 to 2005. It must not be used for automated publishing, fact-checking, or legal review. BBC's selection of stories does not represent all media. On the measured test, classification reached a weighted F1 of 0.9798, above the 0.85 objective. Answer correctness was 0.74, so the 0.80 objective was not met. Extraction exact match was 0.484, so the 0.75 objective was not met.

若被问到抽取为什么这么低，只补这一句：

> Exact match scores a whole field as wrong when one item is extra or missing, and a paraphrased topic scores zero. An overlap F1 on the name fields is about 0.80. That number explains the exact score. It does not replace it.
