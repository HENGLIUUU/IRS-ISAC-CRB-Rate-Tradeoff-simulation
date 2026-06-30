# Design: IRS-assisted Bistatic ISAC CRB-Rate Tradeoff

> 基于 Paper 4 (CRB-Rate Tradeoff, TWC 2026) 加入 IRS 辅助的改进版实现。
> 信号模型固定为 Case 2（叠加信号：高斯信息信号 + 确定性感知信号）。

---

## 1. 系统模型

### 1.1 几何布局

```
坐标系（二维平面）:

                      IRS (50, 30)       
                     🪞  N = 16/32
                    ╱
       ┌────── BS ─╱──→ 目标 (200, 0) ──→ Sensing RX (400, 0)
       │    (0, 0)     ↘
       │              ↘
       └─→ CU (φ_cu = 0.3 rad)
```

| 实体 | 坐标 | 说明 |
|:---|:---:|:---|
| BS (Tx) | (0, 0) | Mt = 32 根天线，沿用 Paper 4 |
| IRS | (50, 30) | N ∈ {16, 32}，靠近 BS 的建筑立面部署 |
| 目标 (Target) | (200, 0) | 感知目标，沿用 Paper 4 |
| Sensing RX | (400, 0) | Mr = 32 根天线，沿用 Paper 4 |
| CU | 方向 φ_cu=0.3 rad, 距离 1000m | 通信用户，沿用 Paper 4 |

### 1.2 信道模型

#### 直射径（同 Paper 4，不变）

| 链路 | 模型 | 公式 |
|:---|:---|:---:|
| BS → Target | 目标信道系数 α | [eq:6] |
| BS → CU | Rician 衰落 h | [eq:62] |
| Target → RX | steering vector b, b_dot | [eq:5, 64] |

#### IRS 反射径（新增）

```
BS → IRS 信道:     G ∈ ℂ^{N×Mt}    路径损耗 + steering (BS 方向 → IRS)
IRS → Target:     h_r ∈ ℂ^{1×N}    steering (IRS → 目标方向)
IRS → CU:         h_rc ∈ ℂ^{1×N}   Rician 衰落 (IRS → CU 方向)

等效 BS→Target via IRS:  h_eq_target = h_r · Θ · G      (1×Mt)   [new]
等效 BS→CU via IRS:      g_eq_cu     = h_rc · Θ · G     (1×Mt)   [new]
```

其中:
- `Θ = diag(e^{jθ₁}, ..., e^{jθ_N})` — IRS 反射相移矩阵
- `|Θₙₙ| = 1` — 单位模约束

#### 有效信道（直射 + IRS 合并）

```
目标方向:  a_eff(Θ) = a + (h_eq_target)^T     (Mt×1)   [new]
CU 信道:   h_eff(Θ) = h + g_eq_cu^T           (Mt×1)   [new]
```

### 1.3 信号模型（Case 2: 叠加信号）

```
发射信号: x(t) = s(t) + x₀(t)
  - s(t) ~ CN(0, R_c)    高斯信息信号（给 CU 传数据）
  - x₀(t) 确定性感知信号（用于感知目标）

协方差: E[x(t)x(t)^H] = R_c + R_s
  - 功率约束: tr(R_c) + tr(R_s) ≤ P
```

### 1.4 性能指标

#### CU 通信 SINR [eq:13, 14 扩展]

```
γ_c(Θ, R_c, R_s) = h_eff^H R_c h_eff / (h_eff^H R_s h_eff + σ²_c)

约束: γ_c ≥ γ₀
```

#### CRB for DoA estimation [eq:45 扩展]

```
a_eff^H(Θ) 替换原 a^H 后，公式结构不变:

CRB(Θ, R_c, R_s) = σ²_s / (2T|α|² · F(Θ, R_c, R_s))

其中:
  A_s = a_eff^H R_s a_eff · ||b_dot||²
  A_c = a_eff^H R_c a_eff · ||b_dot||²
  γ_ran = |α|² · a_eff^H R_c a_eff · ||b||² / σ²_s

  F = A_s + γ_ran/(1+γ_ran) · A_c
```

---

## 2. 优化问题

```
(P):  min  CRB(Θ, R_c, R_s)
      s.t. γ_c(Θ, R_c, R_s) ≥ γ₀            [SINR 约束]
           R_c ⪰ 0, R_s ⪰ 0                  [半正定]
           tr(R_c + R_s) ≤ P                  [功率约束]
           |Θₙₙ| = 1, n = 1,...,N             [单位模约束, 新增非凸]
```

---

## 3. 算法设计：AO（Alternating Optimization）

### 3.1 AO 外层框架

```
Algorithm 1: AO for IRS-assisted ISAC CRB minimization
────────────────────────────────────────────────────────
 1: 初始化 Θ⁽⁰⁾ = I（或随机相位）
 2: repeat k = 1, 2, ...
 3:    计算有效信道: a_eff(Θ⁽ᵏ⁻¹⁾), h_eff(Θ⁽ᵏ⁻¹⁾)
 4:    ── Step 1: 固定 Θ，优化 R_c, R_s ──
 5:    (R_c⁽ᵏ⁾, R_s⁽ᵏ⁾) = SCA_solver(γ₀, h_eff, a_eff, ...)  ← 复用 case2_solver.py
 6:    
 7:    ── Step 2: 固定 R_c, R_s，优化 Θ ──
 8:    Θ⁽ᵏ⁾ = IRS_SDR_solver(R_c⁽ᵏ⁾, R_s⁽ᵏ⁾, G, h_r, h_rc, ...)
 9:    
10:    ── 收敛检查 ──
11:    ΔCRB = |CRB⁽ᵏ⁾ - CRB⁽ᵏ⁻¹⁾| / CRB⁽ᵏ⁻¹⁾
12: until ΔCRB < ε 或 k ≥ K_max
13: return R_c, R_s, Θ
```

### 3.2 Step 1: Beamforming 子问题（复用）

固定 Θ → 有效信道 a_eff, h_eff 定值 → 问题退化为 Paper 4 的 (P4)

⇒ **直接调用 `case2_solver.solve_p4_sca()`**，只需把 h, a 替换为 h_eff, a_eff

**改动量：** 0 行新算法代码，只改输入参数。

### 3.3 Step 2: IRS 相移子问题（SDR）

固定 R_c, R_s → 只有 Θ 是变量 → 子问题:

```
(P_IRS):  min  CRB_IRS(v)
          s.t. γ_c(v) ≥ γ₀
               |vₙ| = 1, n = 1,...,N

其中 v = [e^{jθ₁}, e^{jθ₂}, ..., e^{jθ_N}]^T
```

SDR 解法:

```
1. 定义 v ∈ ℂ^N, V = vv^H, 由 rank-1 约束
2. 松弛 rank-1 → V ⪰ 0（SDP）
3. CRB 和 SINR 约束写成 v^H A v / v^H B v 形式
4. 用 Charnes-Cooper 变换或二分法转化为 SDP
5. CVXPY (SCS) 求解
6. 随机化恢复: v* = randomization(V, trials=100)
7. Θ = diag(v*)
```

**随机化恢复细节：**

```
 randomization(V, trials=100):
   L = cholesky(V)  →  V = LL^H
   最优值 best_obj = -inf
   for t = 1..trials:
     ξ ~ CN(0, I_N)         ← 标准复高斯随机向量
     v_tilde = L @ ξ          ← 采样
     v = v_tilde / |v_tilde|  ← 归一化到单位圆
     obj = eval_objective(v)  ← 算目标函数
     if obj > best_obj:
       best_obj = obj; v_best = v
   return v_best
```

---

## 4. 代码架构

```
crb_sim_case1/
├── main_irs.py              [新增] AO 框架主入口
│
├── steering_vectors.py      [复用] ULA 导向矢量 (Eq.5, Eq.64)
├── channels.py              [扩展] 增加 IRS 信道生成函数
├── crb_calc.py              [扩展] 增加 IRS 版的 CRB 计算
├── comm_rate.py             [扩展] 增加 IRS 版的速率计算
├── beamforming_opt.py       [复用] Proposition 1 闭式解 (baseline)
├── case2_solver.py          [复用] SCA 求解器 (AO 的 Step 1)
├── irs_solver.py            [新增] SDR 求解 Θ (AO 的 Step 2)
├── plot_results.py          [扩展] 增加"有/无 IRS 对比"画图功能
│
├── config.py                [新增] 参数集中管理（README 准则第 4 条）
├── results/                 [使用] 结果保存目录
│
└── README.md                [新增] 项目说明
```

### 各文件职责

| 文件 | 类型 | 职责 |
|:---|:---:|:---|
| `main_irs.py` | **新增** | 参数设置 → AO 迭代 → 画图 → 保存结果。全流程驱动 |
| `config.py` | **新增** | `Mt`, `Mr`, `N`, `P`, `σ²`, 位置坐标等所有仿真参数。README 准则第 4 条 |
| `channels.py` | **扩展** | 新增 `generate_irs_channel()`, `compute_effective_channels()` |
| `crb_calc.py` | **扩展** | 新增 `compute_crb_irs()`，内部使用 `a_eff`, `h_eff` |
| `comm_rate.py` | **扩展** | 新增 `compute_rate_irs()`，内部使用 `h_eff` |
| `irs_solver.py` | **新增** | `solve_irs_sdr()`, `randomization()`，SDR 求解 + 随机化恢复 |
| `case2_solver.py` | **复用** | `solve_p4_sca()` 不变，主调传 `h_eff`, `a_eff` 进去 |
| `plot_results.py` | **扩展** | 新增 `plot_irs_comparison()`，同一张图上画有/无 IRS 的 CRB-Rate 曲线 |

---

## 5. 实验计划

| # | 实验 | 参数 | 预期结果 |
|:---:|:---|:---|:---|
| **E1** | **Baseline 复现** | 无 IRS, Case 2 | 验证 case2_solver.py 跑出的曲线和 Paper 4 一致 ✅ |
| **E2** | **IRS 效果验证** | N=16, 有 IRS vs 无 IRS | 有 IRS 的 CRB 曲线整体低于无 IRS |
| **E3** | **IRS 规模扫描** | N=16, 32 | N=32 的 CRB 优于 N=16（但可能有收益递减） |
| **E4** | **IRS 位置敏感性** | IRS at (50,30) vs (100,20) vs (30,50) | 位置不同，CRB 改善幅度不同 |
| **E5** | **SDR 随机化次数** | trials=10/50/200 | 随机化越多，解越稳定，但计算量线性增加 |

### 输出指标

每轮实验需要保存的变量:
- `gamma_0` (dB) — SINR 约束扫描向量
- `CRB` (rad²) — 感知 CRB 值
- `Rate` (bps/Hz) — CU 通信速率
- 文件名格式: `irs_N16_pos50_30_gamma_scan_20260630.npz`

---

## 6. 编码规范（遵循 ISAC-README.txt）

| 条目 | 要求 |
|:---|:---|
| 变量名 | 规范命名，首次使用加注释说明对应论文变量 |
| 优化算法 | 仅用 SDR/SCA + CVXPY 内置求解器 |
| 模块化 | 每个功能独立函数，加 `[eq:xx]` 标记公式编号 |
| 参数管理 | 集中到 `config.py`，不在各文件里硬编码 |
| 结果保存 | `results/` 目录，文件名带时间戳 + 关键参数 |
| 随机种子 | `np.random.seed(0)` 固定 |
| 注释格式 | 统一风格，变量名与论文一致 |

---

## 7. 与 Paper 6（Active-IRS Detection）的关系

本项目做完后，Paper 6 会更易懂：
- Paper 6 用了 **Active IRS**（带放大能力的 IRS），而本项目用的 **Passive IRS**
- Paper 6 做的是 **目标检测**（检测概率），本项目是 **参数估计**（CRB）
- 两篇共享的核心技术：级联信道建模、SDR 解相移、AO 框架
- **做完本项目 → Paper 6 的 IRS 部分你跳过，专注看 Active vs Passive 的区别和检测部分即可**
