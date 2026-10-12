> **核心问题：如果一个问题没有唯一正确答案，我们应该如何评价大模型的回答，又如何利用这些评价让模型继续变强？**
> 
> 从 SuperCLUE-RAG、RAGAS 到 Rubrics as Rewards（RaR），再到 Chasing the Tail、QuRL 和 Online Rubrics，一条值得关注的研究路线正在形成：**从固定的多维评价，走向题目专属、具有区分度、能够动态调整的奖励标准。**

## 1. 为什么大模型需要更好的奖励标准？

强化学习在数学、代码等任务上已经展现出很强的能力，其中一个重要原因是：**答案虽然难以生成，却相对容易验证。**

例如，数学题可以检查最终答案是否正确，代码任务可以运行测试用例。

这就是 RLVR（Reinforcement Learning with Verifiable Rewards）的基本思想：

$$
R(x,y)=
\begin{cases}
1, & \text{答案正确}\\
0, & \text{答案错误}
\end{cases}
$$

但开放式任务不同。

例如：

> 请分析一家公司的经营状况，并判断其长期竞争优势。

一个高质量回答可能需要兼顾事实准确性、逻辑合理性、信息完整性与证据支持。即使两个回答都没有明显错误，质量也可能相差很大。

传统做法是让 LLM Judge 直接给出整体评分，但这种方式容易受到回答长度、表达风格等因素影响，也难以发现细微的事实与推理问题。

**于是，一个自然的想法是：与其直接判断答案好不好，不如先明确什么样的答案才算好。**

这就是 Rubrics（评分规程）的核心思想。

## 2. Rubrics 并不是新概念：从 RAGAS 到 SuperCLUE-RAG

事实上，细粒度评价早已出现在经典的 RAG 评测体系中。

例如：

| 框架                                                         | 核心思想                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [[RAGAS](https://aclanthology.org/2024.eacl-demo.16/)](https://aclanthology.org/2024.eacl-demo.16/)         | 将 RAG 质量拆分为 Faithfulness、Answer Relevancy、Context Precision 等指标 |
| [[SuperCLUE-RAG](https://github.com/CLUEbenchmark/SuperCLUE-RAG)](https://github.com/CLUEbenchmark/SuperCLUE-RAG) | 从答案规整度、准确度、信息提炼度、文本对齐度等维度评价中文 RAG |
| [[ARES](https://aclanthology.org/2024.naacl-long.20/)](https://aclanthology.org/2024.naacl-long.20/)         | 训练轻量化 Judge，并利用人工数据进行统计校正                 |
| [[RAGBench](https://arxiv.org/abs/2407.11005)](https://arxiv.org/abs/2407.11005)                 | 使用 TRACe 等细粒度标准，诊断证据利用、忠实性和完整性问题    |

这些框架的共同思想是：

**将复杂的整体评价拆解成多个更容易判断的子任务。**

例如，RAGAS 的 Faithfulness 可以近似表示为：

$$
F=
\frac{\text{有证据支持的事实陈述数}}
{\text{回答中的事实陈述总数}}
$$

但通用指标仍然存在局限。

假设问题是：

> 某公司 2024 年营收增长的主要原因是什么？

仅使用 Faithfulness 无法充分判断回答是否遗漏了关键业务、是否混淆同比与环比，或者是否正确解释增长来源。

因为这些要求与具体问题有关。

由此引出一个重要区别：

- **通用指标**：什么是好回答？
- **题目专属 Rubrics**：对于这道题，什么才算好回答？

后者正是 Rubrics as Rewards 的切入点。

## 3. Rubrics as Rewards：让评分标准成为强化学习奖励

论文：[[Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746)](https://arxiv.org/abs/2507.17746)（ICLR 2026）

### 核心方法

RaR 为每个问题生成一套具体 Rubrics，再通过 LLM Judge 评价模型回答，并将评分用于 GRPO 训练。

整体流程如下：

**问题 + 参考答案 → Rubrics 生成 → Policy 采样回答 → Judge 评分 → GRPO 更新**

例如，面对医学诊断问题，Rubrics 可以要求：

- 是否识别需要紧急排查的疾病？
- 是否说明关键诊断依据？
- 是否提出合理的后续检查？
- 是否避免危险或缺乏依据的建议？

论文提出四项 Rubrics 设计原则：

1. **Expert Grounding**：由专家知识或高质量参考答案提供依据。
2. **Comprehensive Coverage**：覆盖重要的评价维度。
3. **Criterion Importance**：区分不同标准的重要性。
4. **Self-Contained Evaluation**：每条标准清晰具体，便于独立判断。

### 两种奖励聚合方式

**Explicit Aggregation：逐项评分后加权求和。**

$$
R(x,y)=
\frac{\sum_{j=1}^{K}w_jc_j(x,y)}
{\sum_{j=1}^{K}w_j}
$$

其中，$c_j$ 表示第 $j$ 项标准是否满足，$w_j$ 是对应权重。

这种方法可解释性强，但难以表达不同标准之间的复杂关联。

**Implicit Aggregation：将全部 Rubrics 提供给 Judge，让其综合评分。**

$$
R(x,y)=f_{\phi}(x,y,\mathcal R_x)
$$

其中，$\mathcal R_x$ 是针对问题 $x$ 的评分标准集合。

论文发现，Implicit Aggregation 整体效果更好。

一个合理的解释是：**评分标准适合明确评价维度，但最终质量未必能通过简单线性加权表示。**

### 实验揭示了什么？

在论文报告的一组 HealthBench 实验中：

| 训练方式 | 得分 | 
| -- | -- | 
| Direct Likert | 23.9% | 
| Reference Likert | 31.7% | 
| RaR：无参考答案的合成 Rubrics | 32.0% | 
| RaR：有参考答案的合成 Rubrics | 35.9% | 
| RaR：人工 Rubrics | 34.8% | 

这说明三个问题：

- 结构化 Rubrics 相比笼统打分具有明显价值。
- 即使没有参考答案，合成 Rubrics 也能提供有效监督。
- 高质量参考知识仍有助于提高奖励质量。

值得注意的是，论文发现较小的 Judge 在获得 Rubrics 后也能取得不错的评分效果。

**这意味着，提高奖励质量不一定需要更强的 Reward Model，也可以通过改善评价任务本身来实现。**

不过，RaR 并没有证明模型可以完全独立、自主地无限提升。Rubrics 仍然可能依赖高质量参考答案或更强模型生成。

它更接近一种利用现有模型知识的 **Self-Bootstrapping**：通过结构化标准，让模型已有的评价能力得到更充分的利用。

## 4. Chasing the Tail：为什么奖励模型最需要区分优秀答案？

论文：[[Chasing the Tail](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8183233dbb325a7e165909042a47e15-Abstract-Conference.html)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8183233dbb325a7e165909042a47e15-Abstract-Conference.html)（ICLR 2026）

RaR 证明 Rubrics 可以作为奖励，但还有一个问题：

> 如果 Rubrics 能区分好答案和坏答案，却无法区分优秀答案与顶尖答案，会发生什么？

答案是：**Reward Over-optimization。**

在强化学习中，Policy 会倾向于选择奖励更高的回答。

考虑 KL 正则化的强化学习目标：

$$
\max_{\pi}
\mathbb E_{y\sim\pi}[R(x,y)]
-\beta D_{\mathrm{KL}}(\pi\Vert\pi_0)
$$

对应的最优策略具有形式：

$$
\pi^*(y\mid x)
\propto
\pi_0(y\mid x)
\exp\left(\frac{R(x,y)}{\beta}\right)
$$

这意味着，高奖励回答的概率会被指数放大。

假设两个回答的真实质量分别为：

- 回答 A：95 分
- 回答 B：98 分

但 Reward Model 错误地认为 A 比 B 更好。

那么 RL 可能不断强化 A，即使 B 才是真正更好的回答。

**关键问题不是 Reward Model 在全部回答上的平均准确率，而是它在高质量回答区域的排序能力。**

论文提出 Refinement-through-Differentiation（RTD）：

1. 收集多个高质量回答。
2. 比较优秀回答之间的细微差异。
3. 提取真正影响质量的关键标准。
4. 将这些标准补充进 Rubrics。

这带来了一个很重要的认识：

**随着 Policy 变强，奖励模型需要提高的不只是判断能力，更是高质量区域的分辨率。**

这也是为什么 Rubrics 不能只停留在“有没有明显错误”这一层。

## 5. QuRL：能否利用外部知识自动生成 Rubrics？

论文：[[QuRL: Rubrics As Judge for Open-Ended Question Answering](https://proceedings.iclr.cc/paper_files/paper/2026/hash/aef34c770664d06eabdfebc5d3d58a9c-Abstract-Conference.html)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/aef34c770664d06eabdfebc5d3d58a9c-Abstract-Conference.html)（ICLR 2026）

RaR 通常使用参考答案帮助生成 Rubrics，但高质量参考答案本身也可能昂贵。

QuRL 提出另一条路线：

**利用互联网中的高质量资料，自动构造题目专属 Rubrics。**

核心流程：

$$
\text{Question}
\rightarrow
\text{Retrieval}
\rightarrow
\text{Knowledge Extraction}
\rightarrow
\text{Rubrics}
\rightarrow
\text{RL}
$$

具体包含：

- **Web Retrieval**：检索与问题相关的外部资料。
- **Meta-Description**：提取真正有用的事实和表达信息。
- **Rubric Generation**：生成题目专属标准。
- **Rubric Filtering**：筛除缺乏区分度或质量不佳的标准。
- **GRPO**：使用 Rubrics 奖励训练模型。

它与 RAG 有一个有趣的区别：

**普通 RAG 检索知识帮助模型回答问题，QuRL 检索知识帮助系统判断什么是好回答。**

论文的消融实验也表明，外部知识和 Rubric Filtering 都会影响最终表现。

不过，QuRL 仍然依赖检索资料的可靠性。外部信息若存在错误，Rubrics 也可能继承这些错误。

因此，它降低了人工构造标准的成本，却没有完全解决监督信号的可信度问题。

## 6. Online Rubrics：评分标准为什么不能随着模型变化？

论文：[[Online Rubrics Elicitation](https://proceedings.mlr.press/v306/rezaei26a.html)](https://proceedings.mlr.press/v306/rezaei26a.html)（ICML 2026）

前面的方法还有一个共同局限：

**Rubrics 通常预先生成，而 Policy 在训练过程中不断变化。**

训练初期，模型可能出现明显的事实错误；训练后期，模型逐渐学会满足现有标准，却可能出现新的问题，例如机械迎合 Rubrics、增加无关细节，或利用 Judge 的评分偏好。

此时：

$$
R_{\mathrm{proxy}}\uparrow
\quad\not\Rightarrow\quad
R_{\mathrm{true}}\uparrow
$$

为了解决这一问题，Online Rubrics 提出了动态评价标准。

核心方法是 **Pairwise Elicitation**：

比较当前 Policy 与对照 Policy 生成的回答：

$$
y_{\mathrm{current}}\sim\pi_{\theta}
$$

$$
y_{\mathrm{control}}\sim\pi_{\mathrm{control}}
$$

然后从两者差异中提取新的 Rubrics：

$$ \Delta\mathcal R_t=E(x,y_{\mathrm{current}},y_{\mathrm{control}}) $$

再经过筛选、去重，补充到原有评分标准中。

这种方法可以发现模型当前出现、但原有 Rubrics 没有覆盖的问题。

与 Chasing the Tail 相比：

- **Chasing the Tail**：重点关注高质量回答之间的区分。
- **Online Rubrics**：重点关注训练过程中不断变化的评价需求。

两者可以互补：前者提高高质量区域的评分精度，后者让评分标准适应新的 Policy 分布。

但动态增加 Rubrics 也不保证奖励一定变好，新标准仍需要质量控制。

## 7. 四篇论文构成了一条怎样的研究路线？

 | 论文 | 核心问题 | 主要思想 | 
 | -- | -- | -- | 
 | Rubrics as Rewards | 如何为开放式任务提供奖励？ | 生成题目专属 Rubrics | 
 | Chasing the Tail | 如何避免高质量回答排序错误？ | 细化高质量回答之间的评分标准 | 
 | QuRL | 如何低成本获得有依据的 Rubrics？ | 外部检索、知识提取与标准过滤 | 
 | Online Rubrics | 如何让奖励适应不断变化的 Policy？ | 动态发现和补充评价标准 | 

这四篇论文不完全是时间上依次继承的关系，而是分别处理 Rubric-based Reward Modeling 的不同挑战。

综合起来，可以得到一个值得研究的框架：

$$
\boxed{
\text{Evidence Grounding}
+
\text{Fine-grained Rubrics}
+
\text{Tail Discrimination}
+
\text{Online Adaptation}
}
$$

即：

- 通过可信证据约束评价标准。
- 用题目专属 Rubrics 明确质量要求。
- 在高质量回答之间保持足够的区分度。
- 随着 Policy 变化动态修订标准。

相比固定的一套通用指标，这种评价系统更有机会适应开放式任务中的复杂质量差异。

但它也更复杂，并带来了额外的计算成本和奖励可靠性问题。

## 8. 一个值得探索的方向：可验证奖励与 Rubrics 结合

并非所有任务都完全不可验证。

例如，搜索相关性分档通常有明确的人工标注，但具体推理还涉及查询理解、实体消歧、证据匹配和逻辑一致性。

这种任务既可以评价最终标签，又需要评价推理质量。

因此，可以考虑构造混合奖励：

$$
R_t=
\lambda R_{\mathrm{label}}
+
(1-\lambda)R_{\mathrm{rubric},t}
$$

其中：

- $R_{\mathrm{label}}$：根据标准标签判断最终结果是否正确。
- $R_{\mathrm{rubric},t}$：使用动态 Rubrics 评价证据和推理质量。

例如，对于“苹果公司”与“苹果种植”的相关性判断，Rubrics 可以检查模型是否正确处理实体歧义，避免将词汇重合误判为语义相关。

这种方案结合了 RLVR 的客观性与 Rubrics 的灵活性。

但有一个重要前提：**推理看起来合理，不代表推理一定正确。**

如果模型只是学会生成符合 Judge 偏好的 CoT，即使 Rubric Reward 提高，真实判断能力也未必改善。

因此，必须通过独立标签、人工评测和分布外测试，验证混合奖励是否带来了真实的泛化收益。

## 9. 总结：大模型真正需要的是更强的 Judge，还是更好的评价问题？

回顾这些工作，可以看到一条清晰的思想演进：

$$
\text{通用多维评价}
\rightarrow
\text{题目专属评价}
\rightarrow
\text{高质量区分}
\rightarrow
\text{动态评价}
$$

最值得思考的是：

**一个模型不擅长直接判断回答质量，并不意味着它完全没有能力识别回答中的具体错误。**

通过合理的 Rubrics，我们可以将复杂判断拆解为更明确的子任务，从而更充分地利用现有模型的知识和评价能力。

这解释了为什么较小的 Judge 也可能产生有用的训练信号，也解释了为什么提高奖励质量并不总是意味着增加模型参数量。

但 Rubrics 不是解决奖励问题的终点。它仍然可能遗漏重要标准、继承外部知识错误，或者被不断优化的 Policy 利用。

因此，我认为未来更有价值的研究问题是：

> **我们能否构建一个由可信证据约束、能够区分高质量回答，并随着模型能力提升而持续完善的奖励系统？**

从这个角度看，Rubrics 的意义已经不只是设计更详细的评分表，而是在探索如何把模型已有的知识和判断能力，转化为更加可靠的学习信号。

---

## 参考论文

1. [[Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746)](https://arxiv.org/abs/2507.17746)，ICLR 2026.
2. [[Chasing the Tail](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8183233dbb325a7e165909042a47e15-Abstract-Conference.html)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8183233dbb325a7e165909042a47e15-Abstract-Conference.html)，ICLR 2026.
3. [[QuRL: Rubrics As Judge for Open-Ended Question Answering](https://proceedings.iclr.cc/paper_files/paper/2026/hash/aef34c770664d06eabdfebc5d3d58a9c-Abstract-Conference.html)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/aef34c770664d06eabdfebc5d3d58a9c-Abstract-Conference.html)，ICLR 2026.
4. [[Online Rubrics Elicitation](https://proceedings.mlr.press/v306/rezaei26a.html)](https://proceedings.mlr.press/v306/rezaei26a.html)，ICML 2026.
5. [[RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/)](https://aclanthology.org/2024.eacl-demo.16/)，EACL 2024.
6. [[ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://aclanthology.org/2024.naacl-long.20/)](https://aclanthology.org/2024.naacl-long.20/)，NAACL 2024.
7. [[RAGBench](https://arxiv.org/abs/2407.11005)](https://arxiv.org/abs/2407.11005)，2024.
8. [[SuperCLUE-RAG](https://github.com/CLUEbenchmark/SuperCLUE-RAG)](https://github.com/CLUEbenchmark/SuperCLUE-RAG)，2024.