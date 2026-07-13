# IRS-ISAC: Passive / Active IRS 辅助双站 ISAC 仿真

基于 CRB-Rate Tradeoff for Bistatic ISAC (TWC 2026) 和 CRB Optimization for Active RIS-Empowered ISAC (TWC 2024) 的仿真项目。

---

## 核心结果

| 配置 | CRB 最低 (rad²) | 改善 |
|:---|---:|---:|
| LoS 基线 | 1.77e-06 | — |
| Passive IRS N=16 | 1.02e+08 | 1x |
| Passive IRS N=128 | 1.59e+06 | 64x |
| Active IRS N=16 (A_MAX=8) | 1.59e+06 | 64x (18 dB) |
| Active IRS N=128 | 2.48e+04 | 4096x |

## 结果图

| 图 | 说明 |
|:---|:---|
| [crb_vs_sinr.png](results/crb_vs_sinr.png) | 9 场景主对比（LoS + Passive + Active）|
| [crb_vs_N.png](results/crb_vs_N.png) | Passive vs Active N 曲线 |
| [crb_vs_N_multi.png](results/crb_vs_N_multi.png) | 不同 A_MAX 下 N 曲线 |
| [crb_vs_amax.png](results/crb_vs_amax.png) | 放大倍数 A_MAX 的影响 |
| [crb_vs_pbs.png](results/crb_vs_pbs.png) | BS 发射功率的影响 |

## 快速开始

```bash
pip install numpy matplotlib cvxpy
python run_simulation.py
```

## 文件

```
run_simulation.py   主仿真
config.py           参数
channel_constant.py 信道初始化
scenario.py         SINR 扫描
channels.py         信道公式
crb.py / rate.py    CRB / 速率
sca_solver.py       SCA 求解器
irs_solver.py       SDR + AO 框架
steering_vectors.py 导向矢量
plot_irs.py         画图
```

## 参考文献

- CRB-Rate Tradeoff for Bistatic ISAC, Song et al., IEEE TWC 2026
- CRB Optimization for Active RIS-Empowered ISAC Systems, Zhu et al., IEEE TWC 2024 (arXiv:2309.09207)
- An overview on IRS-enabled sensing and communications for 6G, Song et al., Sci. China Inf. Sci. 2025
