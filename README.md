# IRS-assisted Bistatic ISAC — CRB-Rate Tradeoff 仿真

通信感知一体化（ISAC）仿真项目。在双站 ISAC 系统中引入智能反射面（IRS），对比不同配置下的感知性能（CRB）与通信速率的折衷关系。

## 背景

- **论文**: Song et al., "CRB-Rate Tradeoff for Bistatic ISAC", TWC 2026
- **场景**: BS(0,0) → Target(200,0) → RX(400,0)，IRS 部署在 (190,5)
- **问题**: Target 在 NLoS（直射径被遮挡）时，BS 无法直接感知，靠 IRS 反射恢复感知能力

## 环境配置

```bash
pip install numpy matplotlib cvxpy
```

## 运行

```bash
# 完整仿真（7 个场景对比）
python run_simulation.py

# IRS 位置优化扫描（找最佳部署位置）
python run_position_scan.py
```

结果保存在 `results/` 目录下（`.npz` 数据 + `.png` 曲线图）。

## IRS 位置优化

当前 IRS 部署在 `pos_irs = [190, 5]`（靠近 Target），这个位置由 `run_position_scan.py` 基于 `||a_eff||²` 代理指标扫描确定（~3s 完成）。

如果 BS、Target 或 RX 位置改变，运行以下命令重新找最优位置：

```bash
python run_position_scan.py
```

扫描原理：`||a_eff||² ∝ L(d_br) × L(d_rt)`，最小化 `d_br × d_rt` 等价于最大化有效信道增益。将 `config.py` 中 `pos_irs` 设为输出的最优位置即可。

## 场景说明

| 场景 | IRS | Target 直射 | 作用 |
|:---|:---:|:---:|:---|
## 场景

|      场景     | IRS | Target 直射 |
| LoS baseline  | 无   |    ✅     |
| NLoS+IRS N=16 | N=16 |    ❌     |
| NLoS+IRS N=32 | N=32 |    ❌     |
| NLoS+IRS N=64 | N=64 |    ❌     |
| NLoS+IRS N=128| N=128|    ❌     |
| LoS+IRS N=32  | N=32 |    ✅     |

## 结论

1. **NLoS 场景下无 IRS 完全无法感知**，IRS 能有效恢复 NLoS 感知能力
2. **CRB 随 IRS 单元数 N 以 1/N² 的规律改善**，增大 N 可显著提升感知精度
3. **LoS 场景下 IRS 几乎无性能增益**，直射径通畅时 IRS 影响可忽略

```
├── run_simulation.py        ← 主仿真入口（场景表驱动）
├── run_position_scan.py     ← IRS 位置优化扫描
├── config.py                ← 参数集中管理
├── channel_constant.py      ← 共用信道常量（几何 + 信道矩阵）
├── scenario.py              ← 场景扫描（SINR 循环取点）
├── channels.py              ← 信道生成公式（CRB-Rate Tradeoff + IRS）
├── crb.py / rate.py         ← CRB / 通信速率计算
├── sca_solver.py            ← SCA 波束赋形优化求解器
├── irs_solver.py            ← SDR IRS 相移优化 + AO 框架
├── steering_vectors.py      ← ULA 导向矢量
├── plot_irs.py              ← 结果可视化
└── README.md
```

## 运算逻辑

整个仿真流程可以概括为**一次初始化，多次求解**：

```
① 初始化（一次性）
   config.py → channel_constant.py 算距离/信道
                        ↓
② 场景循环（7 个场景: LoS / NLoS+IRS N=16/32/64/128 / LoS+IRS N=32）
                        ↓
③ γ₀ 扫描（80 个 SINR 阈值点）
   for γ₀ in -10dB ~ 19dB:
       │
       ├─ LoS baseline → sca_solver.py（SCA 求 Rc,Rs）
       │
       ├─ NLoS+IRS → irs_beam_align（闭式 IRS 相移）
       │              └─ sca_solver.py（SCA 求 Rc,Rs）
       │
       └─ LoS+IRS  → ao_optimize（交替优化）
                       ├─ sca_solver.py（固定 IRS 优化波束）
                       └─ irs_solver.py（固定波束优化 IRS）
                        ↓
④ 评估
   crb.py → 感知精度     rate.py → 通信速率
                        ↓
⑤ 输出
   results/*.npz → plot_irs.py → results/*.png
```

### 关键设计

- **信道只算一次**：所有 γ₀ 点共用同一组信道（改变 SINR 不影响物理信道）
- **NLoS 用波束对齐不用 AO**：无直射径时 IRS 最优相移有闭式解
- **LoS+IRS 才用 AO**：直射径和 IRS 径并存，需交替优化找折衷

| 参数 | 默认值 | 说明 |
|:---|:---:|:---|
| `Mt`, `Mr` | 32 | BS / RX 天线数 |
| `T` | 1024 | 符号数 |
| `P_dBm` | 30 | BS 发射功率 (1W) |
| `N_irs_list` | [16, 32, 64, 128] | IRS 单元数扫描 |
| `pos_irs` | [190, 5] | IRS 部署位置（由 run_position_scan.py 扫出） |
| `CAL_ALPHA` | 1e-32 | 目标信道校准因子（按原论文实验数据标定，相对对比不受影响） |
| `N_gamma` | 100 | SINR 扫描点数 |

## 求解器警告说明

运行时 SCS 求解器可能报 `Solution may be inaccurate`，这是 SCS（一阶求解器）的正常现象，不影响结果。
