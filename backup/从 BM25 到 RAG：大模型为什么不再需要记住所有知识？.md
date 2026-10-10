> 核心问题：如何让模型从“记住知识”转向“寻找知识、利用知识”？
从 BM25、稠密检索、对比学习、ANCE 到 RAG，这条技术演进路线的核心，是不断改进模型获取和使用外部知识的能力。

## 1. BM25：为什么关键词检索如此强大？

传统信息检索的代表方法是 **BM25**，它主要根据关键词匹配程度、词频、逆文档频率（IDF）以及文档长度进行排序。

例如，用户查询：

> 谁提出了相对论？

BM25 倾向于找出包含“相对论”等关键词的文档。

但如果用户换一种问法：

> 哪位科学家建立了关于时空关系的新理论？

即使语义相同，由于关键词不同，BM25 也可能漏掉正确文档。

这就是 **Vocabulary Mismatch（词汇失配）**。

于是产生了一个自然的想法：

**能否不再比较关键词，而是直接比较语义？**

## 2. 稠密检索：双塔架构 + 对比学习

稠密检索（Dense Retrieval）使用神经网络，将 Query 和 Document 映射到同一个向量空间。

经典方案是 **双塔架构（Dual Encoder）**：

$$
\mathbf q=f_\theta(q),\qquad
\mathbf d=g_\phi(d)
$$

通过向量相似度进行检索：

$$
s(q,d)=\mathbf q^\top\mathbf d
$$

双塔的关键优势是：

- **独立编码**：Query 和 Document 分别经过 Encoder。
- **离线索引**：文档向量可以预先计算并存储。
- **高效搜索**：在线只需编码 Query，再利用 ANN 搜索相近文档。

但双塔本身并不能保证语义相关的文本距离更近。

因此，需要通过 **对比学习（Contrastive Learning）** 训练向量空间。

给定一个正样本 $d^+$ 和若干负样本 $d_j^-$，典型的 InfoNCE Loss 为：

$$
\mathcal L=
-\log
\frac{\exp(s(q,d^+)/\tau)}
{\exp(s(q,d^+)/\tau)+
\sum_j\exp(s(q,d_j^-)/\tau)}
$$

其核心目标非常简单：

$$
\boxed{s(q,d^+)>s(q,d^-)}
$$

让相关文档更接近 Query，不相关文档更远离 Query。

**然而，早期稠密检索并没有稳定地超越 BM25。为什么？**

## 3. ANCE：问题不是模型不够强，而是负样本太简单

经典论文：[[ANCE — Approximate Nearest Neighbor Negative Contrastive Learning](https://arxiv.org/abs/2007.00808)](https://arxiv.org/abs/2007.00808)

一个重要问题在于：随机负样本往往太容易区分。

例如：

**Query：谁提出了相对论？**

- 正样本：爱因斯坦提出了相对论。
- 简单负样本：今天的苹果价格上涨了。
- 困难负样本：牛顿提出了经典力学理论。

简单负样本与 Query 几乎无关，模型很快就能将其区分，对训练提供的梯度信号也会逐渐减弱。

真正有训练价值的，往往是那些**看起来相关、却不是正确答案的文档**。

ANCE 提出 **Global Hard Negative Mining**：

1. 使用当前双塔模型编码整个文档库。
2. 构建 ANN（Approximate Nearest Neighbor）向量索引。
3. 为每个 Query 检索相似度高但非已知正样本的文档。
4. 将这些困难负样本用于对比学习。
5. 不断更新模型，并异步刷新索引。

由此形成一个训练闭环：

$$
\boxed{
\text{检索困难负样本}
\rightarrow
\text{优化模型}
\rightarrow
\text{挖掘更难的负样本}
}
$$

ANCE 在多个检索基准上显著提升了稠密检索性能，证明了一个重要观点：

> **负样本的质量，往往比单纯增加负样本的数量更重要。**

当然，困难负样本也可能是未标注的正样本（False Negative），因此并不是越难越好。

## 4. RAG：从寻找知识到利用知识

此前的方法主要解决：

> 如何从海量文档中找到相关信息？

但找到文档不等于能够回答问题。

这就引出了 **RAG（Retrieval-Augmented Generation）**。

经典论文：[[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)](https://arxiv.org/abs/2005.11401)

传统语言模型主要依赖参数化记忆（Parametric Memory）回答问题：

$$
p_\theta(y\mid x)
$$

这意味着，模型既要记住知识，又要理解问题并生成答案。

但事实知识往往需要更新，而参数中的知识难以被精确修改。

RAG 提出另一种架构：

$$
\boxed{
\text{Query}
\rightarrow
\text{Retriever}
\rightarrow
\text{External Knowledge}
\rightarrow
\text{Generator}
\rightarrow
\text{Answer}
}
$$

其中：

- **Retriever**：从外部知识库寻找相关证据。
- **Generator**：结合问题和检索证据生成答案。

RAG 实现了**知识记忆与推理生成在系统架构上的部分解耦**。

模型不再必须把所有事实都记进参数，而是可以在需要时检索外部知识。

不过，这不代表语言模型内部不再存储知识，也不意味着检索到正确证据就一定能够正确推理。

## 5. RAG 如何训练：到底优化检索器还是生成器？

RAG 包含两个可训练模块，因此存在不同的优化策略。

**方法一：固定 Generator，训练 Retriever**

保持生成模型不变，让检索器寻找更有助于生成正确答案的文档。

$$
\max_\eta
\log\sum_z
p_\eta(z\mid x)
p_{\theta_{\mathrm{fixed}}}(y\mid x,z)
$$

核心思想：

> 让检索器适应生成器，找到真正有助于回答问题的证据。

这类思路也出现在 [[REPLUG](https://arxiv.org/abs/2301.12652)](https://arxiv.org/abs/2301.12652) 等工作中。


**方法二：固定 Retriever，训练 Generator**

保持检索器不变，让生成器学会更好地理解、整合检索结果。

典型的监督训练目标为：

$$
-\sum_t \log p_\theta(y_t \mid x, D_K, y_{<t})
$$

核心思想：

> 让生成器适应检索器，更好地利用已有证据。

**方法三：联合或交替训练**

同时优化两个模块，或者分阶段交替更新，使二者相互适应。

需要注意，**顺序训练、交替训练和端到端联合训练并不完全等价**；检索的离散 Top-K 操作也使端到端优化更复杂。

这引出了一个非常关键的转变：

$$
\boxed{
\text{Retrieval Relevance}
\neq
\text{Generation Utility}
}
$$

一篇文档在语义上与问题相关，不代表它一定能够帮助生成器给出正确答案。

因此，RAG 不仅要优化“检索相关性”，还要考虑“知识对最终任务的实际价值”。

## 6. 总结

这条发展路线背后，其实包含三个不断深入的问题：

**第一，如何找到知识？**

从关键词匹配发展到语义检索。

**第二，如何更好地学习检索知识？**

从随机负样本发展到 ANCE 的动态困难负样本挖掘。

**第三，如何有效利用知识？**

从独立的检索任务发展到 RAG 中检索与生成的协同优化。

最终，我们得到一个更一般的认识：

> **模型的智能不仅取决于内部参数存储了多少知识，还取决于它能否在合适的时候找到正确的知识，并利用这些知识完成推理。**

RAG 的价值并不是彻底取代参数化记忆，而是提供了一种更灵活的知识获取机制。

进一步的问题是：如果复杂推理需要多次查找信息，那么检索是否应该从一次性的前置步骤，变成推理过程中可以反复调用的操作？

这也自然引向了 **Iterative RAG、Multi-hop RAG 和 Agentic RAG** 等后续方向。

### 参考论文

1. [DPR — Dense Passage Retrieval for Open-Domain Question Answering (2020)](https://arxiv.org/abs/2004.04906)
2. [ANCE — Approximate Nearest Neighbor Negative Contrastive Learning (2020)](https://arxiv.org/abs/2007.00808)
3. [RAG — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)](https://arxiv.org/abs/2005.11401)
4. [REPLUG — Retrieval-Augmented Black-Box Language Models (2023)](