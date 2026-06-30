# IRS-assisted Bistatic ISAC — CRB-Rate Tradeoff

基于 Paper 4 (CRB-Rate Tradeoff, TWC 2026) 扩展 IRS 辅助版本。

## 快速开始

```bash
# 完整仿真（5 场景对比）
python run_simulation.py

# IRS 位置优化扫描
python run_position_scan.py


## 文件结构

| 文件 | 说明 |
|:---|:---|
| **入口脚本** | |
| `run_simulation.py` | AO 主仿真 — 跑全部 5 个场景的 CRB-Rate tradeoff |
| `run_position_scan.py` | IRS 位置优化扫描 |

| **库模块** | |
| `config.py` | 参数集中管理 |
| `steering_vectors.py` | ULA 导向矢量及其导数 |
| `channels.py` | 信道生成（Paper 4 + IRS 扩展） |
| `crb.py` | CRB 计算（Case 2 + IRS 封装） |
| `rate.py` | 通信速率计算（Case 2 + IRS 封装） |
| `sca_solver.py` | Case 2 波束赋形优化 — SCA 求解器 |
| `irs_solver.py` | IRS 相移优化 — SDR 求解器 |
| `plot_irs.py` | 画图工具（CRB-Rate tradeoff 曲线） |

## 场景矩阵

| 场景 | IRS | Target 直射 | CU 直射 |
|:---|:---:|:---:|:---:|
| LoS, no IRS | ❌ | ✅ | ✅ |
| NLoS, no IRS | ❌ | ❌ | ✅ |
| NLoS, IRS N=16 | ✅ N=16 | ❌ | ✅ |
| NLoS, IRS N=32 | ✅ N=32 | ❌ | ✅ |
| LoS, IRS N=32 | ✅ N=32 | ✅ | ✅ |

## 关键参数 (config.py)

| 参数 | 默认值 | 含义 |
|:---|:---:|:---|
| `N_irs` | 16 | IRS 反射单元数 |
| `pos_irs` | [190.0, 5.0] | IRS 坐标（靠近 Target） |
| `SDR_TRIALS` | 100 | SDR 随机化次数 |
| `AO_MAX_ITER` | 20 | AO 最大迭代次数 |
| `CAL_ALPHA` | 1e-32 | 目标信道校准因子 |

> 融合 CRB-Rate Tradeoff, TWC 2026 复现 + IRS 辅助增强版
> 作者：张桓毓
> 整合日期：2026-06-30

---

## 如何阅读本文档

本文档是项目的**总索引**，按以下路径引导你从零读懂代码：

```
第一部分：CRB-Rate Tradeoff 基础（必读）—— 理解原始 ISAC 框架
  ↓
第二部分：CVXPY 与 SCA 深入（必读）—— 理解求解器工作原理
  ↓
第三部分：速查表 —— 二次型 ↔ 迹转换
  ↓
第四部分：IRS 扩展（必读）—— IRS 增强版的所有新增内容
  ↓
第五部分：代码变更对照 —— 逐文件对比原始和 IRS 版本
```


---

# 第一部分：CRB-Rate Tradeoff 基础

## 1.1 导向矢量 — `steering_vectors.py`

### 理论

Eq.(5)：ULA 导向矢量

N 根天线排成直线，信号从 θ 方向来，每根天线的相位差：

```
a(θ) = [e^{-jπ(N-1)sinθ}, e^{-jπ(N-3)sinθ}, ..., e^{jπ(N-1)sinθ}]^T
```

`(N-1-2n)` 这个结构让相位关于阵列中点对称：
- n=0（第 1 根天线）：相位 = -π(N-1)sinθ
- n=N-1（最后一根）：相位 = π(N-1)sinθ
- 中间的天线相位 ≈ 0（阵列中点 = 参考点）

### 代码

```python
# steering_vectors.py L31-34
n = np.arange(N)
phase = -np.pi * (N - 1 - 2 * n) * spacing * np.sin(theta)
return np.exp(1j * phase).reshape(-1, 1)
```

Eq.(64)：导向矢量导数

```python
b_dot = 1j * π * spacing * cos(θ) * D @ b    # D = diag(-(N-1), -(N-3), ..., (N-1))
```

物理意义：衡量角度变化时接收信号的变化率。||ḃ||² 越大 → 角度分辨力越高 → CRB 越小。

---

## 1.2 信道模型 — `channels.py`

### 路径损耗 Eq.(63)

```
L(d) = K₀_lin × (d/d₀)^(-α₀)

K₀ = -30 dB (1m 处损耗), α₀ = 2.5 (损耗指数), d₀ = 1m
```

数值感受：
| 距离 | 路径损耗 |
|:---|:---:|
| 1m | -30 dB |
| 200m | -87.5 dB |
| 1000m | -105 dB |

### 目标信道系数 Eq.(6)

```
|α|² = β / (L(d_BT) × L(d_TR)) × CAL

d_BT = 200m (BS→Target), d_TR = 200m (Target→RX)
β = 1 (反射系数), CAL = 1e-32 (校准因子)
```

|α|² 是**双跳路径**（BS→Target→RX）的总增益。它出现在 CRB 公式的**分母**上——|α|² 越大，回波越强，CRB 越小。

### BS→CU 信道 Eq.(62) — Rician 衰落

```python
h = √L(d_bc) × [ √(K/(K+1)) × h_los + √(1/(K+1)) × h_nlos ]
```

- h_los：指向 CU 方向的 steering vector
- h_nlos：Rayleigh 随机衰落
- Kc = 1：LoS 和 NLoS 各半

---

## 1.3 通信速率 — `comm_rate.py`

### Case 1：纯高斯信号 Eq.(4)

```python
SINR = h^H R_c h / σ²_c
Rate = log₂(1 + SINR)
```

### Case 2：叠加信号 Eq.(13)-(14)

```python
SINR = h^H R_c h / (h^H R_s h + σ²_c)   # 分母多了感知信号的干扰
Rate = log₂(1 + SINR)
```

| | Case 1 | Case 2 |
|:---|:---|:---|
| 信号 | s(t) ~ CN(0, Rc) | x(t) = s(t) + x₀(t) |
| 变量 | 一个 Rc | Rc + Rs 两个 |
| 功率约束 | tr(Rc) ≤ P | tr(Rc) + tr(Rs) ≤ P |
| SINR 分母 | 只有噪声 σ²_c | 噪声 + 感知干扰 h^H R_s h |

---

## 1.4 CRB 计算 — `crb_calc.py`

### Case 1 Eq.(22)

```
CRB₁ = σ²_s / (2T|α|² · a^H R_c a · ||ḃ||²) × (1 + 1/γ_ran)
γ_ran = |α|² · a^H R_c a · ||b||² / σ²_s
```

前半部分：基本 CRB（与确定性信号相同）
后半部分 (1 + 1/γ_ran)：高斯信号的"惩罚项"
- γ_ran 小（低 SNR）→ 惩罚项 ≈ 1/γ_ran，CRB 显著增大
- γ_ran → ∞（高 SNR）→ 惩罚项 → 1，高斯 ≈ 确定性

### Case 2 Eq.(45)

```
CRB₂ = σ²_s / (2T|α|² × F)
F = a^H R_s a × ||ḃ||² + γ_ran/(1+γ_ran) × a^H R_c a × ||ḃ||²
```

比 Case 1 多了 `a^H R_s a` 项——感知信号 R_s 也贡献到 CRB 中。

---

## 1.5 波束赋形优化 — `beamforming_opt.py`

### Problem (P2)：SINR 约束的 CRB 最小化（Case 1）

```
max  a^H R_c a
s.t. h^H R_c h / σ²_c ≥ γ₀     (SINR)
     R_c ⪰ 0                    (PSD)
     tr(R_c) ≤ P                (功率)
```

### Proposition 1：闭式最优解

判断条件：`P × |h^H a*|² ≥ Mt × γ₀ × σ²_c`
- 成立 → **MRT 解**：R_c = P × a*a^T / ||a||²（全部功率指向目标）
- 不成立 → **Rank-2 解**：在 h 方向和 a 方向之间最优分配功率

> 注意：`beamforming_opt.py` 在 IRS 版本中**不再使用**，因为 Case 2 问题 (P4) 没有闭式解，改用 SCA。

---

## 1.6 Case 2 优化问题 (P4)

这是 IRS 项目**实际使用的**优化问题：

```
max  A_s + γ_ran/(1+γ_ran) × A_c
s.t. h^H R_c h / (h^H R_s h + σ²_c) ≥ γ₀   (SINR)
     R_c ⪰ 0, R_s ⪰ 0                        (PSD)
     tr(R_c) + tr(R_s) ≤ P                    (功率)
```

其中 A_s = a^H R_s a × ||ḃ||², A_c = a^H R_c a × ||ḃ||²

**非凸来源**：γ_ran/(1+γ_ran) × A_c 项中，γ_ran 本身依赖 R_c，导致目标函数不是 R_c 的线性函数。

**SCA 解法**：在当前点做泰勒展开 → 凸子问题 → 迭代至收敛（详见第二部分）。

---

# 第二部分：CVXPY 与 SCA 深入（对应 `case2_solver.py`）

## 2.1 CVXPY 是什么

CVXPY 是一个凸优化建模框架，它把用户写的数学问题翻译成求解器能理解的标准形式。

```
你写的数学问题 (P4,k)
    │
1. DCP 检查 —— 确认问题是凸的
    │
2. 编译 (Canonicalization)
    │   把复数拆成实数：R = X + jY
    │   复 PSD 约束 → [X -Y; Y X] ⪰ 0（64×64 实矩阵）
    │
3. 矩阵组装 —— 变量约束组装成 A·x = b
    │
4. 调用数值求解器 (SCS)
    │   交替方向乘子法迭代求解
    │
5. 映射回你的变量
    │
你得到: Rc_opt = Rc_var.value
```

**为什么用迹形式？**
```python
# 不要这样写（CVXPY 可能报错）：
a.conj().T @ R_var @ a

# 要这样写（安全）：
cp.trace(R_var @ np.outer(a.conj(), a))
```

原因是 CVXPY 的 DCP 检查器处理迹形式更可靠（解析器对复数二次型的支持有限）。

## 2.2 SCA 逐行解析

### L60-64：可行性检查

```python
max_SINR = P * np.linalg.norm(h)**2 / sigma2_c
if gamma_0 > max_SINR: return None
```

物理极限：**全部功率指向 CU 方向时能得到的最大 SINR**。如果用户要求的 γ₀ 超过此值，永远不可行。

### L69-71：预计算常量

```python
C = alpha_sq * np.linalg.norm(b)**2 / sigma2_s         # γ_ran 的常数部分
H_mat = np.outer(h_tilde, h_tilde.conj())                # SINR 约束矩阵
M_mat = np.outer(a_flat.conj(), a_flat)                  # 目标函数矩阵
```

为什么预计算？这些量在 SCA 迭代中不变，不必每次重算。

### L76-91：初始化

```python
Rc_power = γ₀·(P·h_a_corr_sq + 1) / (||h̃||² + γ₀·h_a_corr_sq)
Rc = Rc_power · hh^H / ||h||²       # 指向 CU 方向
Rs = (P - Rc_power) · a_norm·a_norm^H  # 剩余功率指向目标
```

初始化的目标是保证一开始就**满足 SINR 约束**：
- 先算 Rc 需要多少功率才能克服 Rs 的干扰 → 满足 SINR
- 剩余功率全给 Rs → 用于感知
- Cap 到 95% 总功率：留 5% 给 Rs（即使 γ₀ 很大）

### L103-109：泰勒展开（SCA 核心）

当前点 `x_k = a^H R_c a`（R_c 在目标方向的投影功率）

```
原始非线性项：γ_ran/(1+γ_ran) · x   其中 γ_ran = C·x
一阶展开：h(x) ≈ h(xₖ) + h'(xₖ)·(x - xₖ)
h'(xₖ) = C · (2 + C·xₖ) / (1 + C·xₖ)²
```

### L111-127：凸子问题

```python
Rc_var = cp.Variable((Mt, Mt), complex=True)    # 优化变量
Rs_var = cp.Variable((Mt, Mt), complex=True)

# 目标（线性化后）
obj = cp.real(cp.trace(Rs_var @ M_mat) + h_prime * cp.trace(Rc_var @ M_mat))

constraints = [
    Rc_var >> 0,                               # PSD
    Rs_var >> 0,                               # PSD
    cp.real(cp.trace(Rc_var) + cp.trace(Rs_var)) <= P,  # 功率
    cp.real(cp.trace(Rc_var @ H_mat)) >=        # SINR
        gamma_0 * (cp.real(cp.trace(Rs_var @ H_mat)) + 1.0),
]
```

### L129-148：收敛检查

```python
Rc_change = ||Rc_new - Rc_old||_F / ||Rc_old||_F  # Frobenius 范数相对变化
if max(Rc_change, Rs_change) < tol: break           # tol = 1e-4
```

### L151-152：数值修复

```python
Rc = (Rc + Rc.conj().T) / 2   # 强制 Hermitian 对称
```

SCS 是一阶求解器，返回的矩阵可能有 ~1e-15 的非对称性，这句修复它。

---

### SCA 完整流程

```
输入 (γ₀, h, a, P, ...)
    │
    ▼
┌─────────────────────────────┐
│ 可行性检查                   │
│ γ₀ > P·||h||²/σ²_c? → None │
└─────────────────────────────┘
    │ 通过
    ▼
┌─────────────────────────────┐
│ 预计算常量 C, H_mat, M_mat   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 初始化可行点 (Rc, Rs)        │
└─────────────────────────────┘
    │
    ▼
┌──── SCA 循环 (max 50 iter) ──┐
│                              │
│ Step 1: 泰勒展开             │
│  xₖ = a^H R_c a             │
│  h' = C·(2+Cx)/(1+Cx)²      │
│                              │
│ Step 2: 求解凸子问题 (CVXPY) │
│  max tr(RsM) + h'·tr(RcM)   │
│  s.t. Rc,Rs ⪰ 0, tr≤P       │
│                              │
│ Step 3: 收敛检查             │
│  ||Rc-Rc_old|| < tol? → 停  │
│                              │
└──────────────────────────────┘
    │
    ▼
输出 (Rc_opt, Rs_opt, info)
```

---

# 第三部分：速查表

## 二次型 ↔ 迹 转换

```python
# 在 CVXPY 中，用迹形式代替直接二次型更安全

# a^H R a            → cp.trace(R_var @ np.outer(a, a.conj()))
# a^T R a*           → cp.trace(R_var @ np.outer(a.conj(), a))
# h^H R h            → cp.trace(R_var @ np.outer(h, h.conj()))
# tr(R)              → cp.trace(R_var)
# R ⪰ 0 (PSD)        → R_var >> 0
```

**原因**：CVXPY 的 DCP 检查器对迹形式的解析最可靠。

## SCS 求解器状态

| 状态 | 含义 |
|:---|:---|
| "optimal" | 找到最优解，所有约束满足 |
| "optimal_inaccurate" | 找到解但精度略低（通常可用） |
| "infeasible" | 问题无解（约束过紧） |
| "unbounded" | 目标可以无限大 |

## 重要参数

| 参数 | 值 | 含义 |
|:---|:---:|:---|
| Mt | 32 | BS 发射天线数 |
| Mr | 32 | Sensing RX 天线数 |
| T | 1024 | 符号数 |
| P | 1W (30 dBm) | BS 最大发射功率 |
| σ²_c, σ²_s | -80 dBm | CU/RX 噪声功率 |
| K₀ | -30 dB | 1m 处路径损耗 |
| α₀ | 2.5 | 路径损耗指数 |
| d₀ | 1m | 参考距离 |
| CAL_ALPHA | 1e-32 | 目标信道校准因子 |

---

# 第四部分：IRS 扩展

## 4.1 IRS 基础知识

### IRS 是什么

**IRS（智能反射面）**：一块由大量无源反射单元组成的平面。每个单元可独立调节入射信号的**相位**（不能调幅度，无源）。

```
Θ = diag(e^{jθ₁}, e^{jθ₂}, ..., e^{jθ_N})    ← N×N 对角矩阵
|Θₙₙ| = 1                                       ← 单位模约束（非凸！）
```

**核心能力**：调 θ₁,...,θ_N 让反射信号在特定方向相干叠加。

### 级联信道模型

```
BS → IRS 信道:     G ∈ ℂ^{N×Mt}    (纯 LoS MIMO)
IRS → Target:      h_r ∈ ℂ^{1×N}   (LoS steering)
IRS → CU:          h_rc ∈ ℂ^{1×N}  (Rician 衰落)

等效 BS→Target via IRS:
  a_irs = G^T @ (h_r^T * v)   其中 v = [e^{jθ₁}, ..., e^{jθ_N}]^T

等效 BS→CU via IRS:
  h_irs = G^T @ (h_rc^T * v)
```

**物理含义**：BS 发射的信号经 IRS 反射的路径，在**目标端**和**直射径叠加**，在 **CU 端**也和直射径叠加。

## 4.2 有效信道（代码核心）

### a_eff — 等效感知导向矢量

```python
a_eff = a_dir + G.T @ (h_r * v)     # LoS: 直射 + IRS 反射
a_eff = G.T @ (h_r * v)              # NLoS: 仅 IRS 反射
```

**IRS 的全部影响被 a_eff 吸收**。CRB 公式中所有出现 a 的地方都被替换为 a_eff，公式结构不变。

### h_eff — 等效 CU 信道

```python
h_eff = h + G.T @ (h_rc * v)         # 直射 + IRS 反射
```

同样，SINR 约束中用 h_eff 替换 h，速率公式结构不变。

### 为什么 a_eff 和 h_eff 这样计算？

```
BS 发射 x → IRS 收到: z = G @ x           (N×1)
IRS 相移: z' = Θ @ z = v * z              (逐元素相乘)
IRS 反射向目标: h_r @ z' = Σ h_r[n]·v[n]·(G[n,:] @ x)
                 = (G^T @ (h_r * v))^T @ x
所以 a_eff = a + G^T @ (h_r * v)          (加上直射径 a)
```

## 4.3 IRS 信道生成（代码）

### G = generate_irs_bs_channel(Mt, N, d_br, φ_br)

```python
G = √L(d_br) × a_irs(φ_br) @ a_bs(φ_br)^H    # N×Mt 秩-1 矩阵
```

- a_bs(φ_br)：BS 朝向 IRS 的 steering vector (Mt×1)
- a_irs(φ_br)：IRS 接收来自 BS 方向的 steering vector (N×1)
- 外积 → 秩-1 → 纯 LoS（IRS 通常在高处，直视径主导）

### h_r = generate_irs_target_channel(N, d_rt, φ_rt)

```python
h_r = √L(d_rt) × a_irs(φ_rt)^H                # 1×N 行向量
```

### h_rc = generate_irs_cu_channel(N, d_rc, φ_rc, Kc)

和 BS→CU 信道同样的 Rician 模型（LoS + NLoS），区别在 IRS 而非 BS。

## 4.4 IRS 相移优化方法

### 方法 1：波束对齐（快速，闭式）

NLoS 场景中 IRS 是唯一感知路径：

```python
v[n] = exp(-j × (angle(G[n,0]) + angle(h_r[0,n])))
```

补偿 BS→IRS→Target 两段路径的传播相位，让 N 条反射路径在目标处**同相叠加**。

### 方法 2：SDR（半定松弛，`irs_solver.py`）

固定 R_c, R_s 后优化 IRS 相移，非凸问题：

```
Step 1: 构建线性映射 A = G^T @ diag(h_r)     (a_eff = a + A@v)
                               C = G^T @ diag(h_rc)    (h_eff = h + C@v)

Step 2: 把目标 CRB 和 SINR 约束写成 v 的二次型
        v^H M_obj v + 2 Re(l_obj^H v) + const_obj
        v^H M_sinr v + 2 Re(l_sinr^H v) + const_sinr ≥ 0

Step 3: 增广矩阵 → SDP
        V_aug = [v; 1] [v; 1]^H
        max tr(M_obj_aug @ V_aug)
        s.t. tr(M_sinr_aug @ V_aug) ≥ 0
             V_aug ⪰ 0, V_aug[N,N] = 1
        → 松弛 rank-1 约束 V = vv^H

Step 4: 随机化恢复 (SDR_TRIALS=100)
        从 V ≈ LL^H 采样 → 投影到单位圆 → 保留 CRB 最小的 v
```

## 4.5 AO 框架（`main_irs.py`）

### 整体结构

```
对于每个 γ₀:
  初始化 v = 1 (无相移)
  while not converged:
    Step 1: 固定 Θ = diag(v), 优化 R_c, R_s
            → solve_p4_sca(γ₀, h_eff(Θ), a_eff(Θ), ...)
            （复用 case2_solver.py，输入已含 IRS 效果）

    Step 2: 固定 R_c, R_s, 优化 Θ
            → solve_irs_sdr(R_c, R_s, a, G, h_r, h_rc, ...)

    更新 a_eff, h_eff
    CRB 相对变化 < AO_TOL? → break
```

### 场景调度策略

| 场景 | 调度方式 | 原因 |
|:---|:---|:---|
| LoS, no IRS | 直接 SCA | 无 IRS，单步求解 |
| NLoS, no IRS | 全部不可行 | 无感知路径 |
| NLoS + IRS | **波束对齐 + SCA** | 无直射径，IRS 唯一路径，v 有闭式解 |
| LoS + IRS | **完整 AO** | 直射 + IRS 并行，需要交替优化 |

**为什么 NLoS 不用 AO？**
NLoS 中无直射径，IRS 是唯一感知路径。最优 v 就是把所有 IRS 反射路径**对齐到目标方向**——这有闭式解（波束对齐），不需要迭代。

---

## 4.6 位置优化（`position_scan.py`）

### 原理

```
||a_eff||² ∝ L(d_br) × L(d_rt) ∝ (d_br × d_rt)^(-α₀)

d_br = distance(BS, IRS)
d_rt = distance(IRS, Target)
```

最小化 d_br × d_rt = 最小化级联路径损耗 = 最大化 ||a_eff||² = 最小化 CRB。

### 加速版扫描流程

```
Step 1: d_br × d_rt 几何代理   (198点, <0.01s)
Step 2: ||a_eff||² 有效信道增益 (198点, N=16+N=32, <0.05s)
Step 3: 最优位置 SCA 验证      (1点, <3s)
                                 合计: ~3s (旧版全网格SCA ~18min)
```

**关键发现**：||a_eff||² 排名与完整 SCA CRB 排名**完全一致**，因此无需全网格 SCA 验证。

### 最优位置分析

当前配置 `pos_irs = [190, 5]`：
- BS(0,0), Target(200,0), RX(400,0)
- IRS 靠近 Target：d_br=190m, d_rt=11m, d_br×d_rt=2,100
- 旧位置 (50,30)：d_br×d_rt=8,900
- **CRB 改善 ~36 倍**

---

# 第五部分：代码变更对照

## 文件级总览

| 原始文件 | IRS 文件 | 变更 |
|---|---|---|
| `steering_vectors.py` | 同 | 未变 |
| `channels.py` (97行) | `channels.py` (245行) | +IRS 信道生成 + 有效信道计算 |
| `crb_calc.py` (134行) | `crb_calc.py` (161行) | +`compute_crb_irs()` (封装 CRB Case 2) |
| `comm_rate.py` (60行) | `comm_rate.py` (80行) | +`compute_rate_irs()` (封装 Case 2 速率) |
| `case2_solver.py` (155行) | 同 | 完全复用 |
| `beamforming_opt.py` | **删除** | Case 1 闭式解，IRS 项目不用 |
| `main_case2.py` | `main_irs.py` | AO 框架重写 |
| `plot_results.py` | `plot_results.py`(161行) | +IRS 对比画图 (log-scale) |
| — | `config.py` | **新增**：参数集中管理 |
| — | `irs_solver.py` | **新增**：SDR 求解 IRS 相移 |
| — | `plot_combined.py` | **新增**：合并场景画图 |
| — | `position_scan.py` | **新增**：位置优化扫描 |
| — | `docs/` | **新增**：文档目录 |

## 各模块变更细节

### config.py（新增）

| 参数 | 默认值 | 含义 |
|:---|:---:|:---|
| `N_irs` | 16 | IRS 反射单元数 |
| `pos_irs` | [190.0, 5.0] | IRS 坐标 |
| `SDR_TRIALS` | 100 | SDR 随机化次数 |
| `AO_MAX_ITER` | 20 | AO 最大迭代次数 |
| `AO_TOL` | 1e-4 | AO 收敛容差 |

### channels.py 新增函数

| 函数 | 输入 | 输出 | 作用 |
|:---|:---|:---|:---|
| `compute_distance()` | pos1, pos2 | 距离 | 几何辅助 |
| `compute_angle()` | pos_from, pos_to | 角度 (rad) | 几何辅助 |
| `generate_irs_bs_channel()` | Mt, N, d_br, φ_br | G (N×Mt) | BS→IRS 信道 |
| `generate_irs_target_channel()` | N, d_rt, φ_rt | h_r (1×N) | IRS→Target 信道 |
| `generate_irs_cu_channel()` | N, d_rc, φ_rc, Kc | h_rc (1×N) | IRS→CU 信道 |
| `compute_effective_a()` | a_dir, G, h_r, v | a_eff (Mt×1) | 等效感知导向矢量 |
| `compute_effective_h()` | h, G, h_rc, v | h_eff (Mt×1) | 等效 CU 信道 |

### irs_solver.py（新增）

| 函数 | 作用 |
|:---|:---|
| `_build_a_eff_linear_map()` | A = G^T @ diag(h_r) — 把 v 映射到 a_eff 的贡献 |
| `_build_h_eff_linear_map()` | C = G^T @ diag(h_rc) — 把 v 映射到 h_eff 的贡献 |
| `solve_irs_sdr()` | SDP + 随机化，求解最优 v |
| `_randomization()` | 从 V ≈ LL^H 采样，找最好 rank-1 解 |
| `_compute_crb_given_aeff()` | 给定 a_eff 后的 CRB 计算 |

---

## 附录：公式对照

| 符号 | CRB-Rate Tradeoff | IRS 版本 |
|:---|:---|:---|
| 导向矢量 | a | a_eff = a + G^T@(h_r*v) |
| CU 信道 | h | h_eff = h + G^T@(h_rc*v) |
| CRB 结构 | Case 2 Eq.(45) | 同左，a → a_eff |
| 速率结构 | Case 2 Eq.(13) | 同左，h → h_eff |
| 波束赋形 | R_c, R_s (SCA 优化) | 同左（case2_solver 复用） |
| IRS 相移 | — | Θ = diag(v), v_n = e^{jθ_n}（SDR 优化） |

---

## 学习路径建议

| 顺序 | 文件 | 目标 |
|:---:|:---|:---|
| 1 | `steering_vectors.py` | 理解 steering vector 的物理含义 |
| 2 | `channels.py`（原始部分） | 理解路径损耗和目标信道 |
| 3 | `crb_calc.py` | 理解 CRB 公式的结构 |
| 4 | `comm_rate.py` | 理解 SINR 和速率公式 |
| 5 | `case2_solver.py` | **重点**：理解 SCA 迭代原理 |
| 6 | `config.py` | 理解所有参数 |
| 7 | `channels.py`（IRS 部分） | IRS 信道建模 |
| 8 | `irs_solver.py` | IRS 相移优化的 SDR 方法 |
| 9 | `main_irs.py` | AO 框架整合 |
| 10 | `position_scan.py` | 位置优化原理 |

---

> **一句话总结**：本项目 = CRB-Rate Tradeoff 的 CRB-Rate 框架 + IRS 信道建模 + SDR 相移求解 + AO 交替优化。
> 所有 IRS 的影响被 a_eff 和 h_eff 吸收，核心公式（CRB、速率、SCA）结构不变。
