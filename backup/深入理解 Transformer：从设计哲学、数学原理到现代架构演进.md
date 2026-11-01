> Transformer 真正重要的创新，不只是提出了一种新的神经网络，而是重新思考了三个基础问题：**信息如何在不同位置之间交互？一个位置的特征如何被加工？深层网络如何持续保存和更新这些信息？**
>
> 从这个角度看，Attention、FFN、Residual Connection、Positional Encoding 等设计并不是毫无关联的模块拼接，而是一套相互配合的计算机制。

## 1. Transformer 是如何被发明的？

### 1.1 从 RNN 的局限开始

在 Transformer 出现之前，序列建模主要依赖 RNN、LSTM 和 GRU。

以 RNN 为例：

$$
h_t=f(h_{t-1},x_t)
$$

每个时间步的隐藏状态依赖前一个时间步。这种结构具有天然的顺序归纳偏置，但也带来两个问题：

**第一，序列计算难以并行。**

由于 $h_t$ 依赖 $h_{t-1}$，必须顺序处理不同时间步。即使 GPU 可以高效执行矩阵运算，也很难消除这种时间依赖。

**第二，长距离依赖的信息传递路径较长。**

如果序列开头的信息需要影响末尾，那么它通常必须经过许多中间状态。虽然 LSTM、GRU 缓解了梯度传播困难，但没有从根本上消除顺序计算。

因此，一个自然的问题是：

**能不能让两个任意距离的 Token 直接交换信息，而不必经过中间状态？**

### 1.2 Attention 最初只是 RNN 的辅助组件

2014 年，Bahdanau 等人在神经机器翻译中引入了重要的 Attention 机制。

传统 Encoder–Decoder 模型通常把源序列的信息压缩到有限的隐藏状态中。Attention 则允许 Decoder 在生成每个目标 Token 时，动态访问 Encoder 不同位置的表示。

其核心形式是：

$$
c_t=\sum_i\alpha_{ti}h_i
$$

其中：

- $h_i$ 是 Encoder 在第 $i$ 个位置的隐藏状态。
- $\alpha_{ti}$ 表示生成目标位置 $t$ 时对源位置 $i$ 的注意力。
- $c_t$ 是当前目标位置获得的上下文信息。

这是一项关键突破，因为信息不再只能通过固定长度的最终状态传递。

但当时 Attention 仍然附着在 RNN 上：它改善了信息检索方式，却没有消除序列递归。

### 1.3 真正的创新：为什么不把 RNN 直接删除？

Transformer 团队提出了一个更激进的问题：

> 既然 Attention 已经可以直接建立不同位置之间的关系，那么序列建模是否还需要 RNN 或 CNN？

这就是论文标题 *Attention Is All You Need* 的核心思想。

2017 年，Vaswani 等人提出 Transformer，使用 Self-Attention 作为主要的跨位置交互机制，不再依赖循环网络或者卷积网络完成序列关系建模。

原论文的设计并非一次性凭空出现。Attention、Residual Connection、LayerNorm、FFN 都已有技术基础。Transformer 的重要创新，是把这些组件组织成一种能够高效并行、表达复杂序列关系的完整架构。

从科研方法上看，这个过程具有启发性：

1. 发现现有结构的瓶颈：RNN 的顺序依赖。
2. 重新抽象问题：序列建模的本质是否是位置之间的信息交互？
3. 提出替代机制：使用 Self-Attention 建立直接联系。
4. 解决新架构的问题：位置表示、特征变换、深层训练和输出生成。
5. 通过大量实验与消融验证设计。

**创新不一定来自增加新的模块，也可能来自发现一个长期被认为必需的模块其实可以被删除。**

---

## 2. 理解 Transformer 的统一框架：信息交互、特征变换与状态更新

设输入序列表示为：

$$
X=[x_1,x_2,\ldots,x_n]\in\mathbb R^{n\times d}
$$

其中：

- $n$ 是 Token 数量。
- $d$ 是每个 Token 的特征维度。
- 每一行是一个 Token，每一列是一个特征通道。

从计算功能出发，可以区分三个问题。

**信息交互（Token Mixing）**：当前位置如何读取其他位置的信息？

**特征变换（Channel Mixing）**：一个位置内部的不同特征如何被组合与加工？

**状态更新（Residual Update）**：新的计算结果如何融入已有表示，而不必完全覆盖原有信息？

这提供了理解 ResNet、CNN 和 Transformer 的统一视角。

### 2.1 Convolution：固定局部邻域的信息交互

以一维标准卷积为例：

$$
y_i=\sum_{\Delta=-r}^{r}W_\Delta x_{i+\Delta}+b
$$

其中 $W_\Delta$ 是偏移量 $\Delta$ 对应的权重矩阵。

它具有三个重要性质：

- **局部性**：标准小卷积核主要读取有限邻域内的信息。
- **参数共享**：相同的卷积核在不同位置重复使用。
- **固定交互规则**：标准卷积核的权重经过训练后，不会因当前输入内容而改变。

普通卷积通常同时完成局部 Token Mixing 和 Channel Mixing。

多层卷积可以扩大感受野，最终实现远距离信息交互，但单层小卷积核一般无法直接连接任意两个远距离位置。

### 2.2 Self-Attention：内容驱动的动态信息交互

单头 Self-Attention 的简化形式是：

$$
y_i=\sum_{j=1}^{n}\alpha_{ij}(X)W_Vx_j
$$

其中：

$$
\alpha_{ij}(X)=
\frac{
\exp(q_i^\top k_j/\sqrt{d_k})
}{
\sum_l\exp(q_i^\top k_l/\sqrt{d_k})
}
$$

与标准卷积的根本区别在于：

$$
\boxed{\alpha_{ij}=\alpha_{ij}(X)}
$$

即交互权重由当前输入动态决定。

因此，Self-Attention 不必固定读取邻近位置，而可以根据内容直接访问任意位置。

它也包含通道特征投影，因此并非完全只进行 Token Mixing。

### 2.3 Position-wise FFN：逐位置的非线性特征变换

原始 Transformer 中：

$$
FFN(x_i)=\operatorname{ReLU}(x_iW_1+b_1)W_2+b_2
$$

其关键性质是：

$$
y_i=f(x_i)
$$

FFN 不直接读取其他 Token 的状态，而是独立地处理当前位置的特征。

因此可以概括为：

| 模块 | 跨 Token 信息交互 | 通道特征变换 | 主要机制 |
|---|---|---|---|
| 标准局部卷积 | 局部 | 有 | 共享的固定卷积核 |
| Self-Attention | 全局或指定范围 | 有 | 输入相关的动态注意力 |
| Position-wise FFN | 无直接交互 | 有 | 共享的逐位置非线性映射 |

需要注意，Attention 整体也是非线性的，因为其权重包含输入相关的 Softmax。不能简单认为 Attention 是线性的、FFN 才是非线性的。

### 2.4 FFN 和 $1\times1$ 卷积的数学联系

如果卷积核大小为 1：

$$
y_i=Wx_i+b
$$

那么它只进行当前位置内部的通道变换。

因此：

$$
\boxed{
FFN\equiv
1\times1\text{ Conv}
\rightarrow\operatorname{ReLU}
\rightarrow1\times1\text{ Conv}
}
$$

这里的等价指逐位置的线性变换和激活操作，而不是空间邻域卷积。

这说明 FFN 与卷积并非完全独立的计算思想。

### 2.5 为什么 Attention 和 FFN 要交替堆叠？

考虑一个 Transformer Block：

$$
X\xrightarrow{\text{Attention}}Y
\xrightarrow{\text{FFN}}Z
$$

Attention 首先聚合其他位置的信息，FFN 随后加工聚合后的表示。

下一层 Attention 将基于新的表示重新计算 Q、K、V，于是得到新的 Token 关系。

因此，深层 Transformer 可以形成：

$$
\boxed{
\text{信息聚合}
\rightarrow
\text{非线性特征变换}
\rightarrow
\text{重新建立关系}
\rightarrow
\text{再次聚合}
}
$$

FFN 虽然没有直接跨 Token 交互，但它处理的输入已经可能包含前面 Attention 汇聚的上下文信息。

这也是为什么它并不只是一个无关紧要的附加模块。

---

## 3. Attention 的数学本质：Q、K、V 是如何完成信息检索的？

### 3.1 Q、K、V 的定义

设输入为：

$$
X\in\mathbb R^{n\times d_{\text{model}}}
$$

经过三个不同的可学习线性投影：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

其中：

- **Query**：当前 Token 想寻找什么信息。
- **Key**：每个 Token 提供什么信息用于匹配。
- **Value**：每个 Token 实际向其他位置提供什么内容。

Q 和 K 负责计算匹配关系，V 负责传递内容。

即使 Self-Attention 中 Q、K、V 来源于相同的输入，它们经过不同投影后也通常并不相等。

### 3.2 Scaled Dot-Product Attention

原论文的核心公式：

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
$$

可以拆解为四个步骤。

**第一步：计算 Query 与 Key 的相似度**

$$
S=QK^\top
$$

如果有 $m$ 个 Query、$n$ 个 Key：

$$
Q\in\mathbb R^{m\times d_k}
$$

$$
K\in\mathbb R^{n\times d_k}
$$

则：

$$
S\in\mathbb R^{m\times n}
$$

**第二步：缩放**

$$
S'=\frac{S}{\sqrt{d_k}}
$$

**第三步：沿 Key 维度做 Softmax**

$$
A_{ij}=
\frac{\exp(S'_{ij})}
{\sum_{l=1}^{n}\exp(S'_{il})}
$$

对于每个 Query，其在所有有效 Key 上的权重之和为 1。

**第四步：加权求和 Value**

$$
O=AV
$$

若：

$$
V\in\mathbb R^{n\times d_v}
$$

则：

$$
O\in\mathbb R^{m\times d_v}
$$

一个非常重要的结构性质是：

**Attention 输出的位置数量由 Query 决定，而不是由 Key 或 Value 决定。**

这也是理解 Cross-Attention 和 Residual Connection 的关键。

### 3.3 为什么必须除以 $\sqrt{d_k}$？

假设 Q 和 K 的各维分量相互独立，且均值为 0、方差为 1。

对单个 Query 和 Key：

$$
S=q^\top k=\sum_{i=1}^{d_k}q_ik_i
$$

因为：

$$
\mathbb E[q_ik_i]=0
$$

且：

$$
\operatorname{Var}(q_ik_i)=1
$$

在相应独立性假设下：

$$
\operatorname{Var}(S)
=
\sum_{i=1}^{d_k}\operatorname{Var}(q_ik_i)
=d_k
$$

所以：

$$
\operatorname{Std}(S)=\sqrt{d_k}
$$

随着维度增大，未缩放点积的典型幅度也会增加。

于是：

$$
S'=\frac{S}{\sqrt{d_k}}
$$

得到：

$$
\boxed{\operatorname{Var}(S')=1}
$$

因此，$\sqrt{d_k}$ 不是随意设置的超参数，而是具有明确的统计动机。

需要强调，这个方差结论依赖独立性、均值和方差假设，并不是对任意 Q/K 分布都严格成立。

### 3.4 为什么点积幅度过大会影响梯度？

对于 Softmax：

$$
p_i=\frac{e^{s_i}}{\sum_j e^{s_j}}
$$

其导数：

$$
\frac{\partial p_i}{\partial s_j}
=
p_i(\delta_{ij}-p_j)
$$

当某个 Logit 远大于其他 Logit，Softmax 输出可能接近：

$$
[1,0,0,\ldots]
$$

这时不少导数项会接近 0，造成 Softmax 饱和，使优化变得困难。

缩放有助于控制 Logit 的典型尺度，避免随着 $d_k$ 增大而自然变得过于极端。

但缩放不能保证所有梯度都不会消失。

这里还可以从温度参数理解：

$$
A=\operatorname{softmax}(S/T)
$$

- $T$ 越大，分布通常越平缓。
- $T$ 越小，分布通常越尖锐。

Transformer 使用 $T=\sqrt{d_k}$ 控制注意力分数的尺度。

---

## 4. 为什么需要 Multi-Head Attention？

单头 Attention 只产生一套注意力权重及其对应的聚合表示。

原论文提出 Multi-Head Attention，让多个 Head 在不同的可学习表示子空间中进行信息交互。

第 $i$ 个 Head：

$$
head_i=
\operatorname{Attention}
(QW_i^Q,KW_i^K,VW_i^V)
$$

最终：

$$
\boxed{
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(head_1,\ldots,head_h)W^O
}
$$

不同 Head 可能学习不同的依赖关系，例如局部语法、远距离引用、位置匹配等。

但这些功能不是预先指定的，也不能保证所有 Head 都互不冗余。

### 4.1 原论文的主要维度

Transformer Base：

| 参数 | 数值 |
|---|---:|
| $d_{\text{model}}$ | 512 |
| Head 数量 $h$ | 8 |
| $d_k$ | 64 |
| $d_v$ | 64 |
| $d_{ff}$ | 2048 |
| Encoder 层数 | 6 |
| Decoder 层数 | 6 |

对于长度为 $n$、Batch Size 为 $B$ 的 Self-Attention：

$$
X:(B,n,512)
$$

Q/K/V 拆头后：

$$
Q,K,V:(B,8,n,64)
$$

注意力分数：

$$
QK^\top:(B,8,n,n)
$$

每个 Head 聚合结果：

$$
O_i:(B,n,64)
$$

合并 Head：

$$
\operatorname{Concat}(O_1,\ldots,O_8):(B,n,512)
$$

经过 $W^O$ 后，仍为：

$$
(B,n,512)
$$

### 4.2 Head 数量越多越好吗？

不一定。

在固定总投影宽度的条件下：

$$
d_k=d_v=\frac{d_{\text{model}}}{h}
$$

Head 数增加时，每个 Head 的维度减小。

若总 Q/K/V 投影维度保持 $d_{\text{model}}$，忽略 Bias，一个标准 Multi-Head Attention 的主要权重参数量为：

$$
P_{\text{MHA}}=4d_{\text{model}}^2
$$

对于 Base：

$$
P_{\text{MHA}}=4\times512^2=1,048,576
$$

因此，在这种配置下，将 8 个 Head 改成 16 个 Head，并不必然增加主要投影参数量。

原论文的消融实验也表明，Head 数量和翻译表现并不呈简单的单调关系。

---

## 5. Encoder–Decoder：Self-Attention、Cross-Attention 与 Mask

原始 Transformer 不是单纯堆叠 Attention 的网络，而是一个完整的 Encoder–Decoder 架构。

### 5.1 Encoder 的计算

原始 Encoder 每层包含两个子层：

1. Multi-Head Self-Attention。
2. Position-wise FFN。

两者都带 Residual Connection 和 LayerNorm。

原始 Transformer 采用 **Post-LN**：

$$
Z=\operatorname{LN}(X+\operatorname{MHA}(X))
$$

$$
Y=\operatorname{LN}(Z+\operatorname{FFN}(Z))
$$

Encoder 主要负责产生源序列的上下文表示。

其 Full Self-Attention 可以在单层内让任意两个有效源位置直接交互。

### 5.2 Decoder 的三个子层

原始 Decoder 每层包含：

1. Masked Multi-Head Self-Attention。
2. Encoder–Decoder Cross-Attention。
3. Position-wise FFN。

其中最关键的区别是 Q/K/V 的来源。

| Attention 类型 | Query 来源 | Key / Value 来源 |
|---|---|---|
| Encoder Self-Attention | Encoder | Encoder |
| Decoder Masked Self-Attention | Decoder | Decoder |
| Cross-Attention | Decoder | Encoder |

Cross-Attention 的本质是：

**Decoder 根据当前生成状态，从 Encoder 提供的源序列信息中进行检索。**

### 5.3 为什么 Cross-Attention 的 Residual 只连接 Decoder？

这是原始架构图中非常容易忽略的一个细节。

设：

$$
D\in\mathbb R^{T\times d}
$$

是 Decoder 当前隐藏状态；

$$
E\in\mathbb R^{S\times d}
$$

是 Encoder 输出。

Cross-Attention：

$$
Q=DW_Q
$$

$$
K=EW_K,\qquad V=EW_V
$$

其输出：

$$
C=\operatorname{MultiHead}(Q,K,V)
$$

具有与 Decoder 相同的序列长度：

$$
C\in\mathbb R^{T\times d}
$$

于是原论文执行：

$$
\boxed{
D'=\operatorname{LN}(D+C)
}
$$

而不是：

$$
D'=\operatorname{LN}(D+E+C)
$$

这有两个原因。

**首先，Encoder 和 Decoder 的序列长度可能不同。**

例如：

$$
E:(7,512),\qquad D:(5,512)
$$

它们不能直接逐元素相加。

但 Cross-Attention 可以通过：

$$
A:(5,7)
$$

把 Encoder 的信息映射为：

$$
C:(5,512)
$$

从而与 Decoder 状态相加。

**更本质的原因：Residual 用于保留当前正在更新的状态。**

在 Cross-Attention 中：

- Decoder 是信息接收者。
- Encoder 是信息提供者。
- Cross-Attention 是根据 Decoder 的需求，从 Encoder 检索信息。
- Residual 则保留 Decoder 已有的隐藏状态。

因此：

$$
\boxed{
\text{新 Decoder 状态}
=
\text{原 Decoder 状态}
+
\text{检索到的信息}
}
$$

这里的原状态指进入 Cross-Attention 子层前的 Decoder 表示，并不是最开始的 Output Embedding。

即使没有把 Encoder 输出直接加入 Residual，梯度仍然可以通过 Cross-Attention 的 K/V 路径传播回 Encoder。

一个非常有用的统一理解是：

> **Residual 保留当前状态，Attention 从其他状态读取信息。**

在 Self-Attention 中，由于 Q/K/V 来自同一表示，这一区别不太明显；在 Cross-Attention 中，两者的职责变得非常清楚。

### 5.4 为什么 Decoder 需要 Causal Mask？

自回归语言生成要求：

$$
p(y_1,\ldots,y_T\mid x)
=
\prod_{t=1}^{T}p(y_t\mid y_{<t},x)
$$

预测当前位置时，不能读取未来目标 Token。

因此，在 Decoder Self-Attention 中需要限制信息访问范围。

例如长度为 4 的 Causal Mask：

$$
M=
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
$$

Attention 变为：

$$
\boxed{
A=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)
}
$$

这样被屏蔽位置的权重为 0。

Mask 必须在 Softmax 之前正确应用；仅在 Softmax 之后把部分权重清零，一般会破坏权重归一化。

除了 Causal Mask，还有 Padding Mask，用来屏蔽无效的补齐位置。

两者解决的问题不同。

### 5.5 为什么 Shifted Right 和 Causal Mask 必须配合？

假设目标序列为：

```text
我 爱 你 <eos>
```

训练时：

```text
Decoder input: <bos> 我 爱 你
Target:         我   爱 你 <eos>
```

**Shifted Right** 让当前位置的输入不直接包含当前位置的目标标签。

**Causal Mask** 防止当前位置通过 Attention 偷看右侧输入，其中可能包含当前待预测的答案。

如果只使用 Shifted Right、不使用 Mask，仍可能通过右侧位置发生标签泄漏。

如果只使用 Mask、不右移，当前位置本身就可能包含正确答案。

因此，两者承担不同但互补的作用。

### 5.6 为什么训练可以并行，推理却仍然需要逐 Token 生成？

训练时，完整的目标序列已知，可以使用 Teacher Forcing，让所有位置同时计算预测损失。

Causal Mask 保证虽然同时计算，但每个位置只能访问允许的前缀。

而推理时，未来 Token 尚未生成，需要逐步采样或选择：

$$
y_1\rightarrow y_2\rightarrow y_3\rightarrow\cdots
$$

因此：

**Transformer 消除了层内的序列递归依赖，但并没有消除自回归生成本身的时间依赖。**

现代模型使用 KV Cache 复用历史 K/V，降低重复计算，但通常仍然逐步决定新 Token。

---

## 6. Positional Encoding：没有 RNN 后，模型如何理解顺序？

### 6.1 Self-Attention 为什么不能天然理解位置？

考虑没有 Positional Encoding、也没有位置相关 Mask 的 Full Self-Attention。

设 $P$ 是 Token 的置换矩阵。

可以证明：

$$
\boxed{
SA(PX)=P\,SA(X)
}
$$

这意味着如果交换输入 Token 的顺序，输出会对应地交换顺序。

这叫做 **Permutation Equivariance（排列等变性）**。

它不等于排列不变性。Self-Attention 不是对输入顺序变化完全没有响应，而是输出会随输入位置同步置换。

更进一步，对于 Cross-Attention，如果 Key 和 Value 被相同方式置换：

$$
\boxed{
\operatorname{Attention}(Q,PK,PV)
=
\operatorname{Attention}(Q,K,V)
}
$$

这里的置换不会改变加权聚合结果。

因此，如果 Encoder 完全没有源序列位置信息，仅仅重新排列源 Token，就可能无法让 Decoder 区分不同的词序。

这揭示了位置编码的必要性。

不过，带 Causal Mask 的 Self-Attention 已经具有顺序相关的结构约束，不能直接套用上述无 Mask 的排列等变结论。

### 6.2 固定 Sinusoidal Positional Encoding

原论文使用：

$$
PE(pos,2i)=
\sin\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
$$

$$
PE(pos,2i+1)=
\cos\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
$$

不同维度对应不同频率，相邻的 Sin/Cos 两维使用相同频率。

输入表示为：

$$
\boxed{
X_0=\sqrt{d_{\text{model}}}\operatorname{Embedding}(X)+PE
}
$$

Embedding 的缩放与 Attention Logit 的 $\sqrt{d_k}$ 缩放是两项不同设计。

Sinusoidal PE 的主要特点：

- 不需要训练额外的位置参数。
- 可以为未见过的位置直接计算编码。
- 不同频率提供多尺度位置表示。
- 相对位置偏移具有线性变换结构。

### 6.3 为什么 Sin/Cos 能表示相对位置？

定义：

$$
\theta=\frac{pos}{\omega_i},
\qquad
\Delta=\frac{k}{\omega_i}
$$

其中：

$$
\omega_i=10000^{2i/d_{\text{model}}}
$$

根据三角函数加法公式：

$$
\begin{bmatrix}
\sin(\theta+\Delta)\\
\cos(\theta+\Delta)
\end{bmatrix}
=
\begin{bmatrix}
\cos\Delta&\sin\Delta\\
-\sin\Delta&\cos\Delta
\end{bmatrix}
\begin{bmatrix}
\sin\theta\\
\cos\theta
\end{bmatrix}
$$

对于固定偏移 $k$，变换矩阵只与 $k$ 和对应频率有关，而不依赖绝对位置 $pos$。

因此：

$$
\boxed{
PE(pos+k)
\text{ 可以由 }
PE(pos)
\text{ 通过依赖 }k\text{ 的线性变换表示}
}
$$

这是原论文使用 Sinusoidal PE 的重要数学动机。

但这并不保证模型能可靠地泛化到任意长度。能够计算未见位置的编码，不等于一定能够正确理解未见过的长序列。

### 6.4 为什么要直接相加，而不是拼接？

原论文采用：

$$
X=Embedding(X)+PE
$$

这样可以保持隐藏维度 $d_{\text{model}}$ 不变。

如果拼接：

$$
X=[Embedding(X);PE]
$$

则隐藏维度可能增加，需要调整后续网络或额外增加投影层。

相加是一种简洁的设计选择，并不是唯一合理的实现方式。

---

## 7. ResNet 与 Transformer：Residual 不仅是优化技巧

### 7.1 Residual Connection 的起源

ResNet 的基本形式是：

$$
\boxed{
H(x)=x+F(x)
}
$$

其中 $F(x)$ 学习相对于输入的更新。

如果某层不需要改变输入，只要：

$$
F(x)=0
$$

就能得到恒等映射：

$$
H(x)=x
$$

这降低了学习恒等映射的难度，有助于缓解深层网络的退化和优化问题。

### 7.2 为什么 Residual 有助于梯度传播？

对于：

$$
y=x+F(x)
$$

其 Jacobian 为：

$$
\boxed{
\frac{\partial y}{\partial x}
=
I+\frac{\partial F(x)}{\partial x}
}
$$

反向传播时，梯度包含一条恒等路径。

因此，即使残差分支的梯度很弱，也可能通过这条路径传播信息。

但不能据此认为残差连接能够完全消除所有梯度消失或爆炸问题。不同梯度路径仍可能相互影响。

### 7.3 原论文 Post-LN 与现代 Pre-LN

原始 Transformer 使用：

$$
\boxed{
y=\operatorname{LN}(x+F(x))
}
$$

即 Post-LN。

其 Jacobian：

$$
J_y=
J_{\operatorname{LN}}(x+F(x))
(I+J_F(x))
$$

而 Pre-LN：

$$
\boxed{
y=x+F(\operatorname{LN}(x))
}
$$

对应：

$$
J_y=
I+J_F(\operatorname{LN}(x))J_{\operatorname{LN}}(x)
$$

可以看到，Pre-LN 在整个残差块层面保留了更加直接的恒等梯度路径。

后来的理论与实验研究发现，这种差别会影响深层 Transformer 在初始化时的梯度分布与优化稳定性。

但 Pre-LN 并非在所有条件下都严格优于 Post-LN，它们仍涉及训练稳定性、表示演化和网络深度等方面的权衡。

### 7.4 更深刻的解释：Residual Stream 是共享的状态空间

现代 Mechanistic Interpretability 研究提供了另一个视角：

**Residual Connection 不只是为了让梯度传播，更让不同层可以在同一个持续存在的表示空间中读写信息。**

考虑 Pre-LN：

$$
x^{(l+1)}
=
x^{(l)}
+
F_l(\operatorname{LN}(x^{(l)}))
$$

忽略某些具体实现细节，可以把不同层看成对 Residual Stream 进行连续更新的模块：

- Attention 从其他 Token 读取信息。
- FFN 读取当前 Token 的特征，并产生新的变换结果。
- Residual 将这些结果加入已有状态。
- 后续层继续基于更新后的状态计算。

因此，Transformer 不是每一层都完全重新生成表示，而是在已有表示基础上持续叠加、修改和组织信息。

一种有启发性的概括是：

$$
\boxed{
\text{Transformer Block}
=
\text{信息交互}
+
\text{特征计算}
+
\text{共享状态更新}
}
$$

这是一种分析模型计算的视角，而不是对所有架构变体的严格定义。

---

## 8. FFN 的新解释：它可能是一种分布式 Key–Value Memory

原论文将 FFN 描述为逐位置的前馈网络。

但后续研究发现，它还可以被解释为一种分布式记忆机制。

考虑忽略 Bias 的 FFN：

$$
FFN(x)=\operatorname{ReLU}(xW_1)W_2
$$

设隐藏层宽度为 $m$。

将 $W_1$ 的第 $i$ 列记为 $k_i$，$W_2$ 的第 $i$ 行记为 $v_i$，则：

$$
\boxed{
FFN(x)=
\sum_{i=1}^{m}
\operatorname{ReLU}(xk_i)v_i
}
$$

可以理解为：

- $k_i$：某种输入模式的匹配方向。
- $\operatorname{ReLU}(xk_i)$：该模式被激活的程度。
- $v_i$：该模式激活后向输出写入的特征方向。

因此，FFN 不只是笼统地做非线性变换，还可以通过参数存储某些可被输入激活的模式，并根据匹配程度组合对应的特征。

Geva 等人在 *Transformer Feed-Forward Layers Are Key-Value Memories* 中系统研究了这种解释。

但是需要避免一个过度简化：

**一个 FFN 神经元并不一定对应一条清晰、独立的人类知识。**

模型可能使用分布式表示，同一个神经元可能响应多个模式，同一个语义特征也可能分布在多个神经元构成的方向上。

后来的 Superposition、Sparse Autoencoder 以及模型知识编辑研究，正是在进一步分析这些分布式特征的组织方式与因果作用。

---

## 9. Transformer 为什么能够形成复杂算法？Induction Heads 的启示

传统理解通常认为 Attention 负责关联 Token，FFN 负责加工特征。

但 Mechanistic Interpretability 研究发现，多个 Attention Head 在不同层之间可以组成具有明确功能的计算电路。

一个典型例子是 **Induction Head**。

假设上下文中出现：

```text
A B C ... A
```

模型可能根据此前出现过的 `A B` 模式，预测当前位置的 `A` 后面也是 `B`。

一种可能的实现方式是：

1. 某个 Head 将相邻 Token 信息移动到合适位置。
2. 后续 Head 利用这些信息匹配历史中重复出现的模式。
3. 模型读取历史模式后面出现的 Token，用于当前预测。

这种机制展示了一个重要事实：

**简单 Attention 运算通过跨层组合，可以实现更复杂的信息检索算法。**

相关研究也发现 Induction Heads 的形成与某些 In-Context Learning 行为存在关联。

但不能认为所有上下文学习、复杂推理或知识检索都能够被 Induction Heads 单独解释。

其真正启发是：理解 Transformer 不应只研究某一层某个 Head，而应进一步研究模块之间如何形成可组合的计算路径。

---

## 10. Attention Weight 不等于因果解释

可视化 Attention Matrix 很有用，但不能简单认为：

$$
\alpha_{ij}\text{ 大}
\Longrightarrow
x_j\text{ 对最终输出贡献一定大}
$$

原因是：

1. Attention Weight 只决定 Value 的加权系数。
2. 不同 Value 的方向和幅值可能不同。
3. Multi-Head 的输出还要经过线性投影。
4. Residual 和 FFN 会继续改变隐藏状态。
5. 后续层可能增强、抵消或重新组合这些信息。

因此，一个 Token 被高度关注，并不必然意味着它对最终预测具有最大的因果影响。

*Attention is not Explanation* 等研究对此提出了重要质疑。

更可靠的机制分析不仅需要观察 Attention，还需要进行干预：

- 删除某个 Head。
- 修改特定 Attention Score。
- 替换某一层隐藏状态。
- 对 FFN 特征进行定向干预。
- 观察输出行为是否按预期变化。

这区分了**相关性可视化**与**因果机制验证**。

---

## 11. 训练机制：为什么更好的 Perplexity 不一定意味着更好的翻译？

### 11.1 原始训练配置

Transformer Base 的代表性设置包括：

- Adam 优化器。
- $\beta_1=0.9$，$\beta_2=0.98$。
- $\epsilon_{\text{Adam}}=10^{-9}$。
- Warmup Steps 为 4000。
- Dropout 为 0.1。
- Label Smoothing 为 0.1。

学习率调度：

$$
lr=
d_{\text{model}}^{-0.5}
\min\left(
step^{-0.5},
step\cdot warmup^{-1.5}
\right)
$$

因此前期学习率逐渐上升，之后按 Step 的平方根倒数衰减。

Warmup 主要服务于优化稳定性，不是专门防止过拟合的机制。

### 11.2 Label Smoothing 的含义

普通分类使用 One-Hot 目标：

$$
q=e_y
$$

Label Smoothing 则使用：

$$
\boxed{
q=(1-\epsilon)e_y+\epsilon u
}
$$

其中 $u$ 是词表上的均匀分布。

例如 4 个类别，$\epsilon=0.1$：

$$
q=[0.925,0.025,0.025,0.025]
$$

它不再鼓励模型无限制地把正确类别概率推向 1，有助于减少过度自信。

### 11.3 为什么 Label Smoothing 可能降低 PPL，却提高 BLEU？

这里先明确：PPL 越低越好，因此应当说 Label Smoothing 可能**使 PPL 变差（升高）**，而不是降低 PPL。

Perplexity 定义为：

$$
\boxed{
PPL=
\exp\left(
-\frac1N\sum_{t=1}^{N}
\log p_\theta(y_t\mid y_{<t},x)
\right)
}
$$

它衡量的是模型给参考 Token 分配的概率。

对于单个 Token：

$$
PPL=\frac1{p(\text{正确词})}
$$

例如：

- 正确词概率 0.99，对应单 Token PPL 约 1.01。
- 正确词概率 0.90，对应单 Token PPL 约 1.11。

即使两个模型预测的最高概率类别完全相同，只要模型对正确词的置信度不同，PPL 就可能不同。

而 Accuracy 主要衡量预测类别是否正确。

BLEU 则衡量生成译文和参考译文之间的 n-gram 匹配，并包含长度惩罚：

$$
BLEU=
BP\cdot
\exp\left(
\sum_{n=1}^{N}w_n\log p_n
\right)
$$

因此，PPL、Accuracy、BLEU 衡量的是不同方面：

| 指标 | 主要衡量内容 |
|---|---|
| PPL | 对参考 Token 分配的概率 |
| Token Accuracy | 预测类别是否正确 |
| BLEU | 最终生成译文与参考译文的匹配程度 |

Label Smoothing 可能使模型对参考答案的置信度降低，却通过正则化改善泛化与最终生成表现。

原论文的相关消融结果为：

| Label Smoothing | PPL ↓ | BLEU ↑ |
|---|---:|---:|
| 0 | 4.67 | 25.3 |
| 0.1 | 4.92 | 25.8 |
| 0.2 | 5.47 | 25.7 |

这说明不同指标并不必然同步改善，也说明更强的 Label Smoothing 不一定更好。

注意，这里讨论的是相应评估口径下的 PPL，不应把使用平滑目标的训练损失直接等同于 One-Hot 参考标签下的 PPL。

**核心结论：优化概率拟合、提高分类正确率与改善最终生成质量，不一定是完全相同的目标。**

但不能推断 Label Smoothing 总会损害 PPL，或更低的 PPL 对生成质量没有价值。

---

## 12. Transformer 的计算复杂度与后续效率改进

### 12.1 为什么 Self-Attention 有平方复杂度？

设：

$$
Q,K,V\in\mathbb R^{n\times d}
$$

计算：

$$
QK^\top
$$

需要：

$$
O(n^2d)
$$

计算 Attention Matrix 与 Value 的乘积，也需要相同量级的运算。

因此，标准完整 Attention 核心计算：

$$
\boxed{
O(n^2d)
}
$$

考虑 Q/K/V 和输出投影后，通常还包含：

$$
O(nd^2)
$$

所以完整 Attention 模块更全面的计算复杂度是：

$$
\boxed{
O(n^2d+nd^2)
}
$$

标准显式实现还需要存储：

$$
A\in\mathbb R^{n\times n}
$$

因此注意力矩阵本身具有平方级空间开销。

这与 RNN 形成了不同的权衡：

- RNN 有较强的序列递归依赖。
- Self-Attention 具有更好的序列并行性。
- 但完整 Attention 在长序列下计算和存储成本昂贵。

因此，**并行能力强，不等于任何长度下计算量都更低。**

### 12.2 FlashAttention：相同公式，不同计算组织

FlashAttention 的重要贡献不是改变 Attention 的数学定义，而是改变它的计算方式。

传统实现通常会显式构造或读写较大的 Attention Matrix。

FlashAttention 使用：

- 分块计算。
- Online Softmax。
- GPU 内存层级感知的计算安排。
- 尽可能减少高带宽显存与片上内存之间的数据搬运。

其目标仍是计算精确的 Softmax Attention，而不是通过近似删除部分关系。

总体算术复杂度并未因此变成线性，但中间存储和内存访问效率得到了显著改善。

这说明一个重要思想：

**理论计算复杂度、实际内存访问成本和硬件运行速度，是三个不同的优化维度。**

### 12.3 GQA 与 MLA：为什么不必为每个 Query Head 都保存完整 K/V？

原论文的 MHA 为多个 Head 分别计算 Q/K/V。

但自回归推理需要维护历史 K/V Cache。

为降低缓存与带宽成本，后续出现了：

**MQA / GQA**

多个 Query Head 可以共享一组或少量组 K/V Head。

例如 8 个 Query Head 共享 2 组 K/V，在每个 KV Head 维度不变时，KV Cache 的相关元素数量可以下降到原来的四分之一。

**MLA**

进一步使用低秩压缩的潜在表示保存 K/V 相关信息，并通过适当的投影实现注意力计算。

这些改进实际上都在研究一个深层问题：

> 模型究竟需要保存多少历史信息，才能完成有效的上下文检索？

---

## 13. 现代 Transformer 改进了哪些原始设计？

Transformer 在成为主流基础架构后，很多组件都经历了重新设计。

| 原始设计 | 后续代表性方案 | 主要研究动机 |
|---|---|---|
| Post-LN | Pre-LN | 改善深层优化稳定性 |
| LayerNorm | RMSNorm | 简化归一化计算 |
| Sinusoidal PE | RoPE、ALiBi | 改进位置与相对距离表示 |
| ReLU FFN | SwiGLU | 加强门控与特征变换 |
| Dense FFN | MoE | 增加模型容量并稀疏激活参数 |
| Multi-Head Attention | GQA、MLA | 减少 KV Cache 和带宽开销 |
| 标准 Attention 实现 | FlashAttention | 提升 IO 效率 |
| 全量 Attention | 线性 Attention、SSM、混合结构 | 改善长序列效率 |

这些改进不是互斥的，也没有一个模型必须全部采用。

### 13.1 RoPE：位置不一定要加在 Embedding 上

原始 Sinusoidal PE 将位置向量加到 Token Embedding。

RoPE 则直接对 Q/K 施加位置相关的旋转。

设：

$$
\tilde q_m=R(m)q_m
$$

$$
\tilde k_n=R(n)k_n
$$

其中 $R(m)$ 为二维旋转块构成的矩阵。

于是：

$$
\begin{aligned}
\tilde q_m^\top\tilde k_n
&=q_m^\top R(m)^\top R(n)k_n\\
&=\boxed{q_m^\top R(n-m)k_n}
\end{aligned}
$$

因此位置相关部分自然依赖相对偏移 $n-m$。

这带来一种重要的设计思想：

**位置信息不一定属于某个 Token 自身，也可以被直接编码到两个 Token 的关系之中。**

RoPE 具有良好的数学结构，但并不保证无限长度外推。

### 13.2 SwiGLU：FFN 从普通激活变成门控计算

原始 FFN：

$$
FFN(x)=W_2\operatorname{ReLU}(W_1x)
$$

以列向量记法表示，一种 SwiGLU 结构为：

$$
\boxed{
FFN(x)=W_o\left[
\operatorname{SiLU}(W_gx)\odot W_ux
\right]
}
$$

其中 $\odot$ 是逐元素乘法。

它通过一个分支产生门控信号，对另一个分支的特征进行调制。

因此，FFN 不仅进行非线性映射，还能够更灵活地控制不同特征的贡献。

### 13.3 MoE：让不同 Token 激活不同的参数

Mixture of Experts 将 FFN 扩展为多个专家网络，并通过 Router 选择部分专家参与计算。

$$
\boxed{
y=\sum_{e\in S(x)}g_e(x)FFN_e(x)
}
$$

其中 $S(x)$ 是当前输入选中的专家集合。

这使得模型可以在增加总参数容量的同时，控制每个 Token 实际激活的计算量。

从信息路由的角度看：

- Attention 动态选择读取哪些 Token。
- MoE 动态选择调用哪些计算模块。

二者是不同层面的动态路由机制。

---

## 14. Attention 真的是序列建模的唯一选择吗？

2017 年 Transformer 挑战了 RNN 的必要性。

后来，Mamba 等研究又进一步挑战了完整 Attention 的必要性。

Mamba 使用选择性状态空间模型进行序列建模。

一个简化的状态更新形式为：

$$
\boxed{
h_t=A_t h_{t-1}+B_t x_t
}
$$

$$
y_t=C_t h_t
$$

其中状态更新参数可以依赖输入。

这类模型的思路是：

**不一定显式保存全部历史 Token 的 Key/Value，也可以维护一个持续更新的状态，用于压缩和传递历史信息。**

其潜在优势是高效处理长序列。

但压缩状态也带来信息保留能力的权衡：完整 Attention 可以直接检索历史位置，有限状态模型则需要决定哪些信息值得被保留。

因此，后续研究逐渐走向混合设计：

$$
\boxed{
\text{高效历史状态压缩}
+
\text{必要时进行显式检索}
}
$$

从这个角度看，现代序列建模真正的问题可能是：

> 哪些历史信息需要精确保留？哪些可以压缩？什么情况下应该调用哪种计算机制？

---

## 15. 亲手验证：一个理解 Transformer 的 Anatomy Lab

要真正理解上述思想，最有效的方法之一不是调用预训练模型，而是从零实现一个小型 Transformer，并系统地删除或替换组件。

建议构建一个 `Transformer Anatomy Lab`。

### 15.1 模型配置

使用极小规模的 Encoder–Decoder：

| 参数 | 建议配置 |
|---|---:|
| Encoder / Decoder 层数 | 2 / 2 |
| $d_{\text{model}}$ | 64 |
| Head 数量 | 4 |
| $d_k=d_v$ | 16 |
| $d_{ff}$ | 128 |
| 词表大小 | 32 |
| 序列长度 | 4–12 |
| 位置编码 | Sinusoidal |
| 归一化 | 首先采用 Post-LN |

第一遍手写 Attention、Multi-Head、Mask、FFN、Residual 和 PE，不使用现成的 `nn.Transformer` 或 `nn.MultiheadAttention`。

### 15.2 三个合成任务

**任务 A：Sequence Reversal**

```text
Input:   3 8 1 6 4
Output:  4 6 1 8 3
```

重点验证位置编码、Cross-Attention 与顺序建模。

**任务 B：Key–Value Retrieval**

```text
Input:   A:7 B:2 C:9 | Query:B
Output:  2
```

重点验证内容相关检索、Q/K/V 分工及多头机制。

**任务 C：Autoregressive Generation**

训练时使用 Shifted Right + Causal Mask，推理时逐 Token 生成。

重点验证 Teacher Forcing、Mask 与实际生成之间的差异。

### 15.3 最值得做的六组消融实验

| 实验 | 理论问题 | 观察指标 |
|---|---|---|
| 删除 PE | 模型是否可以理解顺序？ | Exact Match、长度泛化 |
| 删除 Causal Mask | 是否存在未来信息泄漏？ | Teacher-forced vs 自回归准确率 |
| 删除缩放因子 | Logit 为什么需要尺度控制？ | 方差、Entropy、梯度 |
| 单头 vs 多头 | 多头是否提供额外表示能力？ | 准确率、参数量、注意力图 |
| 删除 FFN | Attention 自身是否足够？ | 收敛和最终表现 |
| Post-LN vs Pre-LN | 归一化位置如何影响优化？ | 梯度范数、训练稳定性 |

特别推荐三个实验。

**实验一：无位置编码的 Sequence Reversal**

利用排列等变性与 Cross-Attention 对 Key/Value 同步置换的不变性，可以说明在没有其他源序列顺序信息的条件下，标准模型无法普遍区分同一组 Token 的所有不同排列。

这是可以先证明、再通过实验观察的结构性限制。

**实验二：移除 Causal Mask**

不使用 Mask 的 Decoder 可能在 Teacher Forcing 下得到很高准确率，却在真正自回归推理时表现明显下降。

这是理解信息泄漏最直观的实验之一。

**实验三：随机 Q/K 的缩放实验**

设置：

$$
d_k\in\{8,32,128,512\}
$$

比较未缩放与缩放后的：

- Logit 标准差。
- Softmax 分布。
- 注意力熵。
- 相关梯度尺度。

其中熵定义为：

$$
H(A_i)=-\sum_jA_{ij}\log A_{ij}
$$

这能够把 $\sqrt{d_k}$ 的理论推导变成可观察的数值实验。

所有消融实验都应固定数据划分、训练预算和其他配置，最好使用多个随机种子。

不能为了符合预期而筛选结果：某些模块在简单任务上可能是冗余的，理论上有效的机制也未必在所有训练配置下带来更高准确率。

### 15.4 实验的真正目标

不应该只要求模型的 Loss 下降，而应当形成：

$$
\boxed{
\text{理论假设}
\rightarrow
\text{可检验预测}
\rightarrow
\text{控制变量实验}
\rightarrow
\text{观察结果}
\rightarrow
\text{机制解释}
}
$$

这比完成一个普通的文本生成 Demo 更接近真实科研过程。

---

## 16. 回到根本：Transformer 的核心究竟是什么？

经历近十年的发展，我们已经能够从不同层次理解 Transformer。

### 第一层：数学计算

Self-Attention 通过输入相关的矩阵运算实现跨位置的信息聚合。

FFN 通过逐位置的非线性映射进行特征加工。

Residual Connection 让已有表示能够持续参与后续计算。

### 第二层：表示与学习机制

Multi-Head Attention 在不同表示子空间中建立关系。

FFN 可以形成分布式特征与记忆结构。

Residual Stream 为不同层的信息读写提供共享表示。

多层模块的组合可能形成复杂的检索和计算电路。

### 第三层：系统与效率

高效 Attention 实现优化内存访问。

GQA、MLA 等方法压缩推理时的缓存成本。

MoE 动态选择计算模块。

状态空间和混合架构重新权衡精确检索与压缩记忆。

### 第四层：研究方法

从最初删除 RNN，到后来重构 PE、FFN、Attention 和归一化，Transformer 的演进说明：

**一个架构的成功，并不意味着它的每个组件都是必需或最优的。**

事实上，后来的研究者不断发现：

- 有些结构约束来自数学上的表达需要。
- 有些结构来自训练稳定性的考虑。
- 有些结构是特定硬件条件下的效率选择。
- 还有一些组件仅仅是当时有效、后来能够被替换的实现方案。

所以，学习 Transformer 最好的方法，不是孤立记住各个模块的公式，而是不断追问：

**这个模块解决什么问题？如果删除它会怎样？它依赖什么数学假设？它的作用能否被实验验证？是否存在更简单或更高效的替代方案？**

最终，可以用一个统一的研究视角概括 Transformer：

$$
\boxed{
\text{Transformer}
=
\text{动态信息路由}
+
\text{特征与记忆计算}
+
\text{持续状态更新}
}
$$

这不是对所有 Transformer 变体的严格定义，而是一个帮助理解架构功能与后续研究方向的框架。

Transformer 的历史也揭示了一条更普遍的科研规律：

> **重要的架构创新，往往不是不断叠加复杂组件，而是重新发现问题的本质，质疑原本被认为必不可少的设计，再通过理论与实验找到更简洁的计算机制。**

---

## 参考文献与延伸阅读

### 原始架构与历史

1. Vaswani et al., 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
2. Bahdanau et al., 2014. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).
3. He et al., 2015. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
4. The Harvard NLP Group. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
5. Steven Levy, WIRED, 2024. [8 Google Employees Invented Modern AI. Here's the Inside Story](https://www.wired.com/story/eight-google-employees-invented-modern-ai-transformers-paper/).

### 深层优化与位置编码

6. Xiong et al., 2020. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).
7. Su et al., 2021. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
8. Shazeer, 2020. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

### 机制可解释性

9. Geva et al., 2021. [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).
10. Elhage et al., 2021. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
11. Olsson et al., 2022. [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
12. Jain & Wallace, 2019. [Attention is not Explanation](https://arxiv.org/abs/1902.10186).

### 高效注意力与新架构

13. Dao et al., 2022. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
14. Ainslie et al., 2023. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245).
15. DeepSeek-AI, 2024. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434).
16. Gu & Dao, 2023. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
