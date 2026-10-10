
最近读到 Kaiming He 团队的一篇工作：

> **VISTA: A Visual Harness for Reasoning in an Interactive World**

论文链接：[https://arxiv.org/abs/2610.02200](https://arxiv.org/abs/2610.02200)

这篇论文没有提出新的 Transformer，也没有训练新的 World Model。它做的事情非常简单：

> **让多模态模型直接看环境、保存过去的视觉画面，并在需要时主动回看历史。**

但这样一个很简单的 Harness，却让 Claude Opus 5.0 在 ARC-AGI-3 上的 Relative Human Action Efficiency 从 **40.68 提升到 100.00**。

---

## 1. VISTA 在做什么？

VISTA 面对的是一个典型的交互式 Agent 问题：

Agent 被放进一个陌生环境，不知道规则，只能不断：

$$
\text{Observe}
\rightarrow
\text{Reason}
\rightarrow
\text{Act}
$$

例如：

$$
s_t
\xrightarrow{a_t}
s_{t+1}
$$

Agent 根据动作前后的变化，逐渐推断环境规律。

和传统 World Model 不同，VISTA 并没有显式学习：

$$
\hat{s}_{t+1}=F(s_t,a_t)
$$

而是选择：

> **把原始视觉证据保存下来，让现有多模态模型自己推理。**

---

## 2. 三个核心设计

### Visual Observation

VISTA 不直接把数字网格交给语言模型，而是渲染成图像：

$$
I_t=\operatorname{Render}(S_t)
$$

这样模型可以直接利用已有的视觉、空间和物体理解能力。

---

### Lossless Visual Memory

长期交互中会产生大量历史：

$$
I_1,I_2,\dots,I_T
$$

很多 Agent 会把历史压缩成文本摘要或 semantic memory：

$$
M_t=\operatorname{Compress}(I_{1:t})
$$

但这样可能丢失未来才会变得重要的视觉细节。

VISTA 的策略很直接：

$$
\boxed{\text{Keep the evidence}}
$$

历史视觉帧全部保留，需要时再查。

---

### Visual Inspection

真正重要的不只是“记住”，而是 Agent 可以主动决定：

- 看哪一帧历史；
- 放大哪个区域；
- 对比哪些时间点。

于是推理变成：

$$
\text{Reason}
\rightarrow
\text{Inspect}
\rightarrow
\text{Reason}
$$

可以理解成：

$$
\boxed{\text{Active Perception over the Past}}
$$

传统主动感知问的是：

> 我现在应该看哪里？

VISTA 多问了一层：

> **我现在应该重新看过去的哪里？**

---

## 3. 为什么效果这么好？

一个重要原因是，今天的多模态基础模型已经很强。

过去可能是：

$$
\text{weak model}
+
\text{good harness}
=
\text{limited agent}
$$

现在更接近：

$$
\text{strong model}
+
\text{simple harness}
=
\text{strong agent}
$$

VISTA 的实验说明：

> Agent 的瓶颈不一定是模型不会推理，也可能是信息接口设计得不好。

另外，ARC-AGI-3 本身也非常适合这种方法。

很多任务都可以通过：

$$
s_t
\xrightarrow{a_t}
s_{t+1}
$$

比较动作前后的视觉变化来发现隐藏规则。

---

## 4. 这个方法真的很新吗？

从算法角度看，VISTA 的创新其实不算强。

因为这些东西以前都有：

- Observe–Reason–Act；
- 外部 Memory；
- Screenshot-based Agent；
- Crop / Zoom / Inspect；
- External Notes。

所以 VISTA 更像是一篇：

> **系统设计简单，但实验结果非常强的工作。**

它真正有意思的地方不是提出了一个全新的 memory algorithm，而是证明：

> **对于足够强的 foundation model，一个好的 Harness 本身就可能释放出大量潜在能力。**

---

## 5. 相关工作

VISTA 周围有几条很有意思的路线。

### MementoGUI

研究：

> 什么历史值得保存？

它显式考虑：

$$
\text{selection}
+
\text{compression}
+
\text{retrieval}
$$

相比 VISTA 的：

$$
\boxed{\text{Keep everything}}
$$

MementoGUI 更接近：

$$
\boxed{\text{Keep what matters}}
$$

---

### AGMem

AGMem 的观点更激进：

> 保存所有历史图片甚至可能干扰 Agent。

因此它倾向于保存和动作相关的关键视觉证据，而不是完整截图。

---

### Retrodict

Retrodict 强调：

> 一个好的规则，必须能够解释过去发生过的轨迹。

即：

$$
h
\xrightarrow{\text{retrodiction}}
\text{past trajectory}
$$

如果一个规则无法解释历史，它就不是好的 World Model。

---

### Schema

Schema 则更接近传统 World Model。

它要求 Agent 显式构造：

$$
\hat{s}_{t+1}=F(s_t,a_t)
$$

也就是把世界规律写成一个可执行程序，再进行模拟和规划。

---

## 6. 一个很有意思的研究轴

这些工作其实可以放在同一个坐标轴上：

$$
\text{Raw Observation}
\rightarrow
\text{Selected Evidence}
\rightarrow
\text{Semantic Memory}
\rightarrow
\text{Rules}
\rightarrow
\text{Executable Dynamics}
$$

VISTA 站在最左边：

$$
\boxed{\text{Just keep the evidence}}
$$

Schema 更接近最右边：

$$
\boxed{\text{Build an explicit world model}}
$$

它们最终都在回答同一个问题：

> **一个智能 Agent 到底应该如何记住世界？**

---

## 7. VISTA 对 World Model 的启发

传统 World Model 通常希望学习：

$$
z_t=E(o_t)
$$

以及：

$$
z_{t+1}=F(z_t,a_t)
$$

而 VISTA 提供了另一种思路：

$$
\boxed{
\text{Keep observations}
+
\text{Let the model reason}
}
$$

它并不能证明显式 World Model 不再重要。

因为真实物理世界中：

$$
\text{visual history}
\neq
\text{dynamics model}
$$

更可能的未来是：

$$
\boxed{
\text{Visual Memory}
+
\text{Structured State}
+
\text{Dynamics Model}
+
\text{Active Inspection}
}
$$

---

## 8. 最后

我觉得 VISTA 最值得记住的不是某个具体模块，而是两个非常简单的思想：

$$
\boxed{\text{Keep the evidence}}
$$

以及：

$$
\boxed{\text{Let the model decide when to look again}}
$$

它留下了一个很有意思的问题：

> **随着 foundation model 越来越强，我们还需要给 Agent 手工设计多少复杂认知模块？**

也许未来 Agent 的竞争，不只是模型本身有多强，也在于：

> **我们如何让模型观察、记忆和重新访问这个世界。**