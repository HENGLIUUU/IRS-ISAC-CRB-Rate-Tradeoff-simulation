# Code Guide — Active IRS-ISAC 项目

> 按「运行顺序」来组织，从入口到核心算法，每步讲清楚在干什么。

---

## 入口文件

### `run_simulation.py` — 总控制台

```
main()
  ├── 初始化信道（只跑一次）
  ├── 建场景表（9 个场景）
  ├── 循环跑每个场景 → scan_scenario()
  ├── 保存数据
  └── 画图
```

看 `scenarios` 列表就知道了，每个元组一行就是一个场景：

```python
("NLoS+IRS N=16 (P)", True, 16, True, False, N_gamma)
  #   名字         用IRS?  N  遮挡? Active? 点数
```

---

## 场景层

### `scenario.py` — 跑一条曲线

对每个 γ₀（-10dB ~ 19dB）做：

```
for 每个 γ₀:
    if NLoS + IRS:
        v = irs_beam_align(h_r, G)      # 闭式波束对齐
        if active:
            v = v × min(A_MAX, 安全功率增益)
        a_eff = compute_effective_a(...)
        h_eff = compute_effective_h(...)
        Rc, Rs = solve_p4_sca(...)       # SCA 求解

    if LoS + IRS:   → ao_optimize(...)  # 交替优化
    if LoS 基线:     → solve_p4_sca(...)
```

warm start：相邻 γ₀ 点的解作为初始值传给 SCA，加速收敛。

---

## 信道层

### `channel_constant.py` — 信道初始化

```
init_geometry_and_channels()
  └─ 算距离 d_bt, d_br, d_rt, d_rc
  └─ 算 steering vector a, b, b_dot
  └─ 算 |α|²

generate_irs_channels(N, pos_irs, geo)
  └─ 生成 G, h_r, h_rc（信道矩阵）
```

### `channels.py` — 信道数学公式

| 函数 | 公式 | 含义 |
|:---|:---|:---|
| `path_loss_linear(d)` | L(d) = K₀(d/d₀)^(-α₀) | 路径损耗 |
| `generate_rician_channel()` | h = √L × (√(K/(K+1))·h_los + √(1/(K+1))·h_nlos) | CU 信道 |
| `generate_irs_bs_channel()` | G = √L·a_irs·a_bs^H | BS→IRS |
| `generate_irs_target_channel()` | h_r = √L·a_irs | IRS→Target 列信道 |
| `irs_beam_align()` | v = e^{-j(∠G + ∠h_r)} | 波束对齐 |
| `compute_effective_a()` | a_eff = a + G^T(v ⊙ h_r) | *CRB-Rate Tradeoff for Bistatic ISAC* 的 `a_eff^T x` 感知约定 |
| `compute_effective_h()` | h_eff = h + G^H(conj(v) ⊙ h_rc) | 统一列信道约定下的等效通信信道 |

---

## 优化求解层

### `sca_solver.py` — SCA 波束赋形优化

```
solve_p4_sca(gamma_0, h, a, ...)
  ├── 可行性检查（γ₀ 太高直接返回 None）
  ├── 初始化 Rc, Rs（先保通信，再给感知）
  └── SCA 迭代（最多 50 次）
        ├── 泰勒展开 h_prime = Cx(2+Cx)/(1+Cx)²
        ├── 解凸子问题 (CVXPY)
        └── 收敛检查
```

### `irs_solver.py` — SDR IRS 优化 + AO 框架

**`solve_irs_sdr()`** — 固定 Rc,Rs 优化 IRS 相移

```
① 把 CRB 和 SINR 约束写成 v 的二次型
② 松弛 rank-1 → 凸 SDP（CVXPY 求解）
③ 随机化恢复（100 次采样取最优）
```

Passive：`diag(V) = 1` | Active：`diag(V) ≤ A_MAX²` + 功率约束

**`ao_optimize()`** — 交替优化框架（LoS+IRS 场景用）

```
for 每次 AO 迭代:
    Step 1: 固定 Θ, 优化 Rc,Rs → SCA
    Step 2: 固定 Rc,Rs, 优化 Θ → SDR
    检查 CRB 是否收敛
```

---

## 指标计算

### `crb.py` — CRB 公式

```
CRB = σ² / (2T|α|² × F)

F = a_eff^T R_s a_eff* × ||b_dot||² + γ/(1+γ) × a_eff^T R_c a_eff* × ||b_dot||²
```

a → a_eff 就是 IRS 的替换，公式结构不变。

### `rate.py` — 通信速率

```
SINR = h_eff^H R_c h_eff / (h_eff^H R_s h_eff + σ²)
Rate = log₂(1 + SINR)
```

### `steering_vectors.py` — 导向矢量

```
a(θ) = [e^{-jπ(N-1)sinθ}, ..., e^{jπ(N-1)sinθ}]^T
```

---

## 配置

### `config.py` — 所有参数

| 参数 | 默认 | 说明 |
|:---|---:|:---|
| Mt, Mr | 32 | 天线数 |
| N_irs_list | [16,32,64,128] | IRS 扫描范围 |
| A_MAX | 8 | Active IRS 最大增益（~18dB）|
| P_RIS_dBm | 10 | RIS 功率预算 |
| pos_irs | [190,5] | IRS 位置 |
| CAL_ALPHA | 1.0 | 不使用经验校准；前向/返回路损分开建模 |

---

## 关键词速查

| 你在代码里看到 | 对应含义 |
|:---|:---|
| `Rc` | 信息信号协方差矩阵（通信用）|
| `Rs` | 感知信号协方差矩阵（雷达用）|
| `a_eff` | 含 IRS 反射的等效导向矢量 |
| `h_eff` | 含 IRS 反射的等效 CU 信道 |
| `v` | IRS 反射系数向量；Passive 为单位模，Active 可调幅 |
| `A_MAX` | Active IRS 最大幅度 |
| `γ₀` (gamma_0) | SINR 阈值（通信质量要求）|
| `γ_ran` (gamma_ran) | 感知 SNR |
| `α²` (alpha_sq) | 目标反射系数 |
| `a_dir` | 直射径 steering vector |
| `b, b_dot` | RX 端 steering vector 及其导数 |
| `G` | BS→IRS 信道矩阵 |
| `h_r` | IRS→Target 信道 |
| `h_rc` | IRS→CU 信道 |
| SCA | 逐次凸逼近（优化 Rc,Rs）|
| SDR | 半定松弛（优化 v）|
| AO | 交替优化（SCA ⟲ SDR）|

---

## Active IRS 噪声和功率

通信端的 IRS 噪声与等效信道严格分开：

```text
h_eff = h + G^H diag(conj(v)) h_rc
sigma_IRS,CU^2 = sigma_I^2 ||diag(conj(v)) h_rc||^2
SINR = h_eff^H Rc h_eff
       / (h_eff^H Rs h_eff + sigma_c^2 + sigma_IRS,CU^2)
```

给定 `Rc, Rs, v` 后，有源 IRS 的实际输出功率为

```text
P_out = sum_n |v_n|^2 ([G(Rc+Rs)G^H]_nn + sigma_I^2).
```

`scenario.py` 会保存 `irs_noise`、`irs_output_power` 和 `irs_gain`。
前向 IRS 噪声被目标反射到 sensing RX 的功率也会保存为
`forwarded_sensing_noise`。默认几何下它远小于 sensing RX 热噪声，
所以 CRB 保留 *CRB-Rate Tradeoff for Bistatic ISAC* 的白噪声模型。

---

## 必须记住的模型边界

- 当前工程不是 *Cramér-Rao Bound Optimization for Active
  RIS-Empowered ISAC Systems* 或 *Active IRS-Enabled Integrated
  Sensing and Communications with Extended Targets* 的完整复现。
- 上述两篇论文采用回波再次经过 RIS 后返回 BS 的双程感知拓扑。
- 当前工程是 IRS 只辅助 BS→Target 前向照射、独立 sensing RX 接收。
- 因此当前 CRB 是“*CRB-Rate Tradeoff for Bistatic ISAC* 双站 CRB
  + IRS 等效前向信道”。
- 主图采用 `alignment` 基线；AO/SDR 通过本地小规模回归验证，主图
  不能标成完整 AO 最优结果。
- `POWER_ACCOUNTING` 可选 `same_bs_power` 或 `equal_total_power`；
  每个场景的实际 BS 功率会写入结果元数据。
