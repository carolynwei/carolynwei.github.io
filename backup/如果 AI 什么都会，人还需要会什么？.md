关于 AI，最常见的问题是：

> **AI 会不会取代人？**

但这个问题其实太粗了。

更准确的问题应该是：

> **AI 会取代哪些任务？又会在哪些地方增强人的能力？**

因为 AI 对人的影响，并不只有“替代”这一种。

可以粗略写成：

$$
\text{Substitution}
+
\text{Augmentation}
+
\text{New Tasks}
+
\text{Workflow Redesign}
$$

真正发生变化的，往往不是整个职业消失，而是职业内部的任务重新分配。

# 1. 自动化和增强，不是一回事

先区分三个概念。

## Automation：AI 自己干

原来：

$$
Human \rightarrow Task \rightarrow Output
$$

后来：

$$
AI \rightarrow Task \rightarrow Output
$$

比如 OCR、语音转录、简单翻译、重复的数据整理。

这是真正的“替代”。

## Augmentation：AI 帮人干得更好

如果人的表现是：

$$
P(H)
$$

用了 AI 后：

$$
P(H+A)>P(H)
$$

那就是增强。

比如程序员用 Copilot 写代码，医生用 AI 辅助检索，研究者用 AI 整理论文。

## Complementarity：人机组合超过双方单独工作

更严格的情况是：

$$
P(H+A)>\max(P(H),P(A))
$$

这才是真正的“1+1>2”。

但现实中，这种效果并没有想象中普遍。

不少研究发现，人和 AI 合作虽然经常比人单独工作更好，却未必比单独的 AI 更强。

所以：

$$
Human+AI \not\Rightarrow Better
$$

# 2. 人和 AI 为什么可能互补？

关键不是“谁更聪明”，而是：

> **人和 AI 擅长的东西不同，犯的错误也不同。**

AI 更擅长：

- 大规模信息处理；
- 检索；
- 模式识别；
- 重复执行；
- 快速生成候选方案。

人更擅长：

- 理解复杂语境；
- 决定目标；
- 处理价值冲突；
- 面对异常情况；
- 对结果负责。

所以真正有价值的组合是：

$$
AI=\text{Search + Scale + Pattern}
$$

$$
Human=\text{Goal + Context + Judgement}
$$

如果两者永远在同一个地方犯错，就没有什么互补价值。

真正重要的是：

$$
E_H \neq E_A
$$

# 3. AI 怎么帮助人？

我觉得可以归纳成四类。

## 3.1 把人从低价值杂活里解放出来

人的认知资源有限：

$$
C_{\text{routine}}
+
C_{\text{reasoning}}
$$

如果 AI 能减少：

$$
C_{\text{routine}}
$$

就可以让更多精力留给真正需要判断的工作。

比如：

- 查资料；
- 改格式；
- 写模板代码；
- 整理会议记录；
- 总结文档。

AI 最大的价值之一，不一定是“替你思考”，而是：

> **少让你把脑子浪费在不值得思考的地方。**

## 3.2 把专家变得更高效

医生、律师、程序员、研究者都有一个共同问题：

> 信息太多，人处理不过来。

AI 很适合做：

$$
\text{大量信息}
\rightarrow
\text{候选信息}
$$

然后由人判断。

例如程序员不再把大量时间花在查 API、写 boilerplate code 上，而是更多处理：

$$
\text{Problem Definition}
+
\text{Architecture}
+
\text{Verification}
$$

AI 越会“生成”，人的价值就越向“判断”迁移。

## 3.3 把专家经验扩散给普通人

这一点可能更加重要。

一项针对 5000 多名客服人员的研究发现，生成式 AI 能提高整体工作效率，而提升最大的是经验较少、能力较弱的员工。

原因很直观：

$$
\text{Expert Knowledge}
\xrightarrow{AI}
\text{Scalable Guidance}
$$

以前新人要坐在高手旁边慢慢学。

现在 AI 可以把一部分经验直接“复制”出去。

所以 AI 可能不是：

$$
Expert\rightarrow Super Expert
$$

而是：

$$
Novice\rightarrow Intermediate
$$

## 3.4 扩大一个人“能做什么”

这是最容易被忽略的一点。

比如一个不会前端的人有一个产品想法。

过去：

$$
Idea\rightarrow ?
$$

现在可能：

$$
Idea
\xrightarrow{AI}
Prototype
$$

类似的事情还包括：

- 不会画画的人做视觉原型；
- 不会 SQL 的人分析数据；
- 不会外语的人跨语言交流；
- 不会写代码的人做简单自动化。

所以 AI 不只是：

$$
\text{Do old things faster}
$$

更可能是：

$$
\boxed{
\text{Do things previously impossible for you}
}
$$

这才是真正意义上的能力扩展。

# 4. 但 Human + AI 也可能更差

事情没有这么美好。

AI 有几个典型副作用。

## Automation Bias

AI 一旦给出一个看起来很专业的答案，人会降低警惕。

但：

$$
\text{Fluent}
\neq
\text{Correct}
$$

## Anchoring

如果 AI 先给答案，人后面的判断容易被它带偏。

所以：

$$
AI\rightarrow Human
$$

和：

$$
Human\rightarrow AI
$$

可能完全不同。

在高风险任务里，更合理的流程可能是：

$$
Human_0
\rightarrow
AI
\rightarrow
Human_1
$$

先独立判断，再看 AI 意见。

## Deskilling

短期来看：

$$
Productivity\uparrow
$$

但长期可能：

$$
Skill\downarrow
$$

比如学生每道题都直接问 AI。

作业完成速度上去了，但能力未必提高。

所以不能只测：

$$
Performance_{today}
$$

还要测：

$$
Skill_{future}
$$

## Homogenization

AI 还能带来一个很隐蔽的问题：

> 所有人越来越像。

如果每个人都问同一个模型：

> “给我一个有创意的方案。”

结果可能是：

$$
Average\ Quality\uparrow
$$

但：

$$
Diversity\downarrow
$$

对普通办公未必是坏事。

但对科研、艺术和创新来说，这是个真正的问题。

 # 5. 人真正不可轻易外包的是什么？

AI 越来越强以后，人的价值不会简单消失，而会发生迁移。

一些能力越来越便宜：

$$
\text{Search}
$$

$$
\text{Drafting}
$$

$$
\text{Basic Coding}
$$

而另一些能力反而更重要：

### Problem Formulation

你到底在解决什么问题？

### Verification

AI 给你的东西对不对？

### Taste

十个都不错的答案，哪个真正好？

### Domain Expertise

没有专业知识，你甚至不知道 AI 哪里错了。

### Responsibility

最后谁做决定，谁承担后果？

所以：

$$
Value(\text{Generation})\downarrow
$$

同时：

$$
Value(\text{Evaluation})\uparrow
$$

# 6. 哪些任务适合自动化，哪些适合人机合作？

可以用两个维度理解：

$$
\text{AI Capability}
$$

和：

$$
\text{Need for Human Judgement}
$$

比如 OCR：

AI 很强，人类判断需求又低，适合直接自动化。

医疗决策则不同：

AI 能力可以很强，但：

- 错误成本高；
- 情境复杂；
- 需要解释；
- 需要承担责任。

因此更适合：

$$
Human+AI
$$

而不是：

$$
AI\ Only
$$

# 7. 所以真正应该问什么？

以后再讨论：

> AI 会不会取代人？

可能可以换成三个更准确的问题：

$$
\boxed{\text{What should AI do?}}
$$

$$
\boxed{\text{What should humans do?}}
$$

以及：

$$
\boxed{\text{How should they work together?}}
$$

AI 当然会替代很多任务。

但这不等于：

> 人的价值消失。

更可能发生的是：

> **人的价值从“执行”逐渐迁移到“定义、判断、验证和负责”。**

所以一个好的 AI 系统，不应该只是让：

$$
Human\ Effort\downarrow
$$

而应该让：

$$
Human\ Capability\uparrow
$$

AI 帮助人的最好状态，不是：

> “这件事以后你不用会了。”

而是：

> **“因为有了它，你现在能做以前做不到的事情。”**

这大概才是从 **Automation** 到 **Augmentation** 最重要的区别。

