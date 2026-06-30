# IRS 基础知识 — 面向 IRS-assisted Bistatic ISAC 项目的定向速成

> 本文件是"基于 Paper 4 (CRB-Rate Tradeoff) 做 IRS 辅助 ISAC 改进版"的知识铺垫。
> 只讲做这个项目需要的知识，不涉及 IRS 的全部内容。

---

## 一、IRS 是什么

**IRS（智能反射面）** 是一块由大量无源反射单元组成的平面。每个单元可以独立调节入射信号的**相位**（不能调幅度，因为是无源的）。

- N 个反射单元，每个单元给信号乘一个复系数 e^{jθₙ}
- **只能调相位**，不能放大信号
- 功耗极低（没有射频链路）
- 反射单元数量可以很大（64/256/1024 等）

### 核心能力

通过调节 θ₁,...,θ_N，让反射信号在特定方向**相干叠加**（constructive interference）或在特定方向**相消**（destructive interference）。

IRS 的本质：**用大量低成本无源单元，把电磁环境从"被动"变成"可编程"的。**

---

## 二、IRS 的数学模型

### 2.1 反射矩阵（核心中的核心）

```
反射相移矩阵：Θ = diag(e^{jθ₁}, e^{jθ₂}, ..., e^{jθ_N})    ← N×N 对角矩阵

约束条件：|Θₙₙ| = 1      ← "单位模约束" (unit-modulus constraint)
         θₙ ∈ [0, 2π)   ← 每个相移可连续调（或离散调, 如 1-bit: {0,π})
```

这个 `|Θₙₙ| = 1` 约束是 IRS 优化中最关键、也最麻烦的地方——它是**非凸**的。

**为什么非凸？**
- 凸集要求集合中任意两点的连线仍在集合内
- 复平面上的单位圆 |z|=1 上的两点的连线会穿过圆内部 → 不在圆上 → 非凸

所以我们不能直接用 CVXPY 求解带这个约束的优化问题，需要特殊处理。

### 2.2 级联信道（IRS 的存在感）

IRS 不影响发射端和接收端的天线模型，它只是给信号增加了一条反射路径。

```
BS → IRS 信道:     G ∈ ℂ^{N×Mt}    （IRS 处有 N 个单元，BS 有 Mt 根天线）
IRS → 目标信道:    h_r ∈ ℂ^{1×N}    （目标角度到 IRS 的 steering vector）
IRS → CU 信道:     h_rc ∈ ℂ^{1×N}   （CU 方向到 IRS 的 steering vector）

等效 BS → 目标 via IRS:
  h_eq_target = h_r · Θ · G            (1×Mt)
              = Σ_{n=1}^{N} h_r[n] · e^{jθₙ} · G[n, :]

等效 BS → CU via IRS:
  g_eq_cu = h_rc · Θ · G               (1×Mt)
```

**直觉：** 每对 (IRS 单元 n, 路径) 贡献一个 e^{jθₙ} 加权的信道分量。调节 θₙ 相当于调这些分量的相位，让它们在目标方向**叠加增强**或者**对 CU 方向干扰最小**。

### 2.3 IRS 信道模型（生成方式）

实际仿真中有两种做法：

**方法 A：基于距离的路径损耗 + steering vector（推荐，和 Paper 4 保持一致）**
```
路径参数和 Paper 4 完全一样（K0=-30dB, alpha0=2.5, d0=1m）

BS → IRS: G = sqrt(L_br) * steering_vector_at_IRS(angle_from_BS)
IRS → 目标:   h_r_Tx = sqrt(L_rt) * steering_vector(angle_to_target)   （IRS 侧）
IRS → RX: 同理

等于是把 Paper 4 的 path_loss_linear() 和 steering_vector() 复用到 IRS 链路上
```

**方法 B：随机信道（简单但物理意义弱）**
```
G = (randn(N, Mt) + 1j*randn(N, Mt))/sqrt(2)
```

---

## 三、你的场景中 IRS 怎么工作

### 3.1 系统模型（Bistatic ISAC + IRS）

```
                            ┌─────┐
               BS → Target ─→│目标 │──→ Sensing RX（直射径）
                 (直达径)    └─────┘
                    │
    BS ──→ IRS ─────┤
                    │
                    └──→ CU（IRS 反射径）
                BS ──→ CU（直射径）
```

链路拆解：

| 链路 | 建模 | 对应 Paper 4 的什么 |
|:---|:---|:---:|
| BS → Target（直射） | 同 Paper 4，alpha, a(φ) | Eq.(6) |
| BS → IRS → Target（反射） | G (BS→IRS) → Θ → h_r (IRS→Target) | 新增 |
| BS → CU（直射） | 同 Paper 4，Rician h | Eq.(62) |
| BS → IRS → CU（反射） | G (BS→IRS) → Θ → h_rc (IRS→CU) | 新增 |

**关键参数：**
- BS 位置、IRS 位置、目标位置 → 决定距离和角度
- IRS 大小 N（反射单元数）
- IRS 位置自由度（可优化部署）

### 3.2 加入 IRS 后对 CRB 的影响

IRS 反射信号经目标反射后到达 Sensing RX，等效于**增强了 target方向的回波信号强度**：

```
原始（无 IRS）CRB:
  CRB ∝ σ²_s / (2T|α|² · a^H R a · ||b_dot||²) × (1 + 1/γ_ran)
  
加 IRS 后:
  a^H R a → (a + a_IRS)^H R (a + a_IRS)
  
其中 a_IRS 是 IRS 反射路径的等效 steering vector，包含 Θ 的调相效果
```

如果 IRS 相移 Θ 调得好 → a + a_IRS 在目标方向叠加增强 → a^H R a 增大 → CRB 降低

### 3.3 加入 IRS 后对通信的影响

IRS 反射信号也会到达 CU，既可能是帮助也可能是干扰：

```
直射径信号: h^H R_c h（有用信号）
IRS 反射径: g_eq_cu · R_c · g_eq_cu^H（可能有用也可能干扰——取决于相位）

SINR = h^H R_c h / (h^H R_s h + g_eq_cu · (R_c+R_s) · g_eq_cu^H + σ²_c)
          ↑ 原始的         ↑ 原始的                ↑ IRS 带来的额外干扰/帮助
```

如果 IRS 相调解得好 → 反射信号在 CU 方向可以变成**建设性叠加**，甚至提升通信速率。

---

## 四、IRS 相移的优化方法

由于 |Θₙₙ| = 1 是非凸约束，需要特殊处理。以下是主流方法的对比：

### 方法 1：SDR（Semidefinite Relaxation）— 推荐入门

**核心思想：** 把 θ 向量提成 rank-1 矩阵，松弛掉 rank-1 约束。

```
令 v = [e^{jθ₁}, e^{jθ₂}, ..., e^{jθ_N}]^T    (N×1)
则 v 的二次型: v^H A v = tr(A · v v^H)

定义 V = v v^H, 则 V 是 rank-1 的 PSD 矩阵

松弛: 去掉 rank-1 约束, 只保留 V ⪰ 0
      → 问题变成 SDP, 可以用 CVXPY 求解

求解后: 从 V 恢复 v（随机化方法 randomizarion）
  - 对 V 做 Cholesky 分解: V = LL^H
  - 生成随机向量 ξ ~ CN(0, I)
  - 取 v_hat = Lξ, 再归一化每个元素到单位圆上
  - 试多次，选目标函数值最好的那个
```

**优点：** SDP 理论成熟，CVXPY 直接支持，容易上手
**缺点：** N 较大时变量规模膨胀（N×N 矩阵），随机化恢复可能有性能损失

### 方法 2：SCA（和 Paper 4 的 SCA 思路完全一致！）

**核心思想：** 在 |e^{jθₙ}|=1 附近做一阶泰勒展开。

```
|e^{jθₙ}| = 1 等价于: Re(e^{jθₙ}) ≤ 1 且 -Re(e^{jθₙ}) ≤ -1

但更常用的做法: 在目标函数中用一阶泰勒展开近似非凸项
  
具体来说，如果问题包含 v^H A v（v 的二次型）:
  在当前点 vₖ 处，用一阶近似:
  v^H A v ≈ 2·Re(vₖ^H A (v - vₖ)) + vₖ^H A vₖ  (凸化)

然后求解凸近似子问题 → 更新 vₖ → 迭代直至收敛
```

**优点：** 和 Paper 4 的 SCA 同一套思路，复用已有代码经验
**缺点：** 需要仔细设计凸近似，质量依赖初始点

### 方法 3：AO（Alternating Optimization）— 框架层面的思路

这不是 IRS 特有的优化方法，而是**整体问题的求解框架**：

```
问题: min CRB(Θ, R_c, R_s)
      s.t. SINR ≥ γ₀, tr ≤ P, R_c,R_s ⪰ 0, |Θₙₙ|=1

AO 框架:
  repeat until convergence:
    Step 1: 固定 Θ, 优化 R_c, R_s
            → 这就是 Paper 4 的 SCA 求解器（case2_solver.py）！
            → 只需把信道换成 IRS 增强后的信道即可
    
    Step 2: 固定 R_c, R_s, 优化 Θ
            → 用 SDR 或 SCA 求解 IRS 相移子问题
    
    检查 CRB 的变化量是否小于阈值
```

**AO** 的好处是两个子问题都是（相对）好解的：
- Step 1：直接复用 case2_solver.py 的 SCA
- Step 2：SDP（SDR）或凸近似

---

## 五、和具体论文的关系

| 论文 | 和本项目的关系 |
|:---|:---|
| **Paper 1** (IRS Overview, 2025) | 本项目的大背景。综述给了 IRS-ISAC 的全局图景但公式不够细。本项目的"具体代码"正好补上 Paper 1 缺失的实操细节。 |
| **Paper 4** (CRB-Rate Tradeoff, 2026) | 本项目的基础。波束赋形优化、SCA 算法、CRB 公式全部复用或扩展。 |
| 喻翔昊 **WCL 2025** (Active-IRS Detection) | Paper 6，Level 3 进阶论文。等你做完本项目再读，会理解更深——它用了类似的方法但做的是目标检测。 |

---

## 六、你对 IRS 需要记住的

| 要记住的 | 一句话 |
|:---|:---|
| IRS 能做什么 | 调反射信号的相位，让它在特定方向聚焦 |
| 约束是啥 | \|e^{jθₙ}\| = 1，非凸 |
| 怎么建模 | Θ = diag(e^{jθₙ})，级联信道 h_r Θ G |
| 怎么优化 | AO 框架：固定 IRS 优化波束赋形 → 固定波束赋形优化 IRS |
| 和 Paper 4 的关系 | 波束赋形部分直接复用 case2_solver.py |

**IRS 虽然是新技术，但在你的项目里它只是一个"加了几条路径、多了一个 Θ 变量"的扩展。Paper 4 的代码是你已经跑通的起点。**

下一步进入技术路线图阶段时，我会把这个知识变成代码。
