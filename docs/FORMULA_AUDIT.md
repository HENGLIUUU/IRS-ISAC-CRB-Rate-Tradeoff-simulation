# 公式与代码审查（2026-07-23）

## 审查范围

- *CRB-Rate Tradeoff for Bistatic ISAC*
- *Cramér-Rao Bound Optimization for Active RIS-Empowered ISAC Systems*
- *Active IRS-Enabled Integrated Sensing and Communications with Extended
  Targets*（PDF 正文标题与本地文件名中的 VTC 标题不一致）

## 与 *CRB-Rate Tradeoff for Bistatic ISAC* 的对应关系

| 论文公式 | 代码 | 状态 |
|---|---|---|
| CU 接收信号与 SINR，式 (12)、(13) | `rate.py`, `sca_solver.py` | 一致；Active IRS 噪声作为额外噪声加入 |
| 双站目标矩阵 `H = alpha b a^T`，式 (6) | `channels.py`, `crb.py` | 已修正为转置约定 |
| Case 2 CRB，式 (45)、(46) | `crb.py` | 已修正为 `a^T R a*` |
| P4 约束，式 (57a)-(57c) | `sca_solver.py` | 一致 |
| SCA 分解与一阶下界，式 (58)-(60) | `sca_solver.py` | 已修正权重 |

*CRB-Rate Tradeoff for Bistatic ISAC* 的 SCA 线性目标中，通信
协方差项的正确系数（省略公共的
`||b_dot||²`）为

```text
C*x_k*(2 + C*x_k) / (1 + C*x_k)^2
```

旧代码少了 `x_k`，会改变优化结果。

## Active IRS 通信侧

*Cramér-Rao Bound Optimization for Active RIS-Empowered ISAC Systems*
式 (2)、(3) 和 *Active IRS-Enabled Integrated Sensing and
Communications with Extended Targets* 式 (2)、(3) 都表明：

- 等效通信信道包含 IRS 反射的 BS 信号；
- IRS 放大噪声不属于等效信道；
- 放大噪声功率进入 SINR 分母；
- IRS 系数受单元幅度约束；
- Active IRS 还受输出功率约束。

当前 `channels.py`、`rate.py`、`irs_solver.py` 已按这些原则实现。快速
统一增益基线使用保守总功率上界；SDR 使用给定 `Rc+Rs` 下的精确单程
输出功率。

*Cramér-Rao Bound Optimization for Active RIS-Empowered ISAC Systems*
式 (4)-(5) 还明确区分两次 RIS 噪声：第一次注入的 `z0` 经
RIS-target-RIS-BS 多段衰减后被论文近似忽略；回波返回 RIS 后新注入的
`z1` 则保留在感知噪声协方差中。当前双站拓扑只有前向 IRS 和独立
sensing RX，没有第二次经过 IRS，因此不存在可直接照搬的 `z1` 项。
代码会计算被目标反射的第一遍 IRS 噪声作为
`forwarded_sensing_noise` 诊断；默认参数下它只有 sensing RX 噪声的
约 `1.39e-10`，故继续采用 *CRB-Rate Tradeoff for Bistatic ISAC*
的白噪声 CRB。

## 与两篇双程 Active-IRS 感知论文的模型边界

三者的感知拓扑不同：

- 当前工程：BS 发射，IRS 只辅助前向照射目标，独立 sensing RX 接收
  `Target -> RX` 回波；
- *Cramér-Rao Bound Optimization for Active RIS-Empowered ISAC
  Systems*：点目标、RIS-target-RIS-BS 双程反射，BS 接收回波；
- *Active IRS-Enabled Integrated Sensing and Communications with Extended
  Targets*：扩展目标响应矩阵 `E`、IRS-target-RIS-BS 双程反射，BS 接收。

因此上述两篇双程感知论文的噪声协方差、FIM、CRB 和 Active IRS
双程功率约束不能直接套入当前拓扑。当前 `compute_crb_irs()` 是

```text
CRB-Rate Tradeoff for Bistatic ISAC 的 CRB + 等效前向目标 steering vector
```

它是明确的单程扩展，不是上述双程感知模型的完整 CRB 复现。

## 路径损耗分解

arXiv 官方 TeX 源只写出 `alpha = beta * sqrt(L1)`，没有继续定义
`L1` 如何由两段距离生成，也没有公开关联代码。工程采用可审计的标准
级联分解：

```text
a_forward = sqrt(L_BT) * a
|alpha_return|² = |beta|² * L_TR
```

因此直达链路的总功率增益为

```text
|beta|² * L_BT * L_TR
```

而 IRS 场景可以把 `a_forward` 替换成含 BS→IRS→Target 衰减的
`a_eff`，不重复计算前向路损。原来的逆路损加 `CAL_ALPHA=1e-32`
经验标定已删除。该分解是工程明确采用的物理假设；由于论文未定义
`L1`，不能声称它复现了作者未公开的绝对 CRB 数值设置。

## 自动验证

本地测试套件当前覆盖：

- 通信等效信道展开恒等式；
- *CRB-Rate Tradeoff for Bistatic ISAC* 目标信号
  `a_eff^T x` 展开恒等式；
- `a^T R a*` CRB 公式；
- SCA 权重的有限差分导数检查；
- Active IRS 噪声幅度平方缩放；
- Active IRS 统一增益功率上界；
- Active IRS 实际输出功率与矩阵迹公式的展开恒等式；
- SDR 恢复后的幅度、SINR 和 IRS 功率检查。
- SCA Taylor 函数的切点相等、全局下界和真实目标单调性；
- AO 最终返回值在更新 `Rc/Rs` 后的独立 SINR、幅度和 IRS 功率复查。
- 随机信道生成不改变 NumPy 全局随机状态；
- 前向 IRS 感知噪声与其秩一协方差迹的展开恒等式。

## 2026-07-23 完整仿真后的约束活跃性审计

最终完整 40 点、9 场景本地回归中，所有场景均有 40/40 个可行点，
通信 SINR 约束的最小裕量约为 0 dB。生成结果按仓库规则不提交到 Git。

这组参数下，有源 IRS 曲线需要谨慎解释：

- N=16、32、64、128 的统一增益都等于 `A_MAX=8`；
- 10 dBm 总输出功率约束没有激活；
- 保守输出功率上界依次为 -11.82、-8.81、-5.80、-2.79 dBm；
- N=16 的真实输出功率在代表性三个 SINR 点为 -11.83 至 -16.86 dBm；
- CU 端 IRS 噪声最多为接收机热噪声的 `4.87e-7`；
- 前向感知 IRS 噪声最多为 sensing RX 热噪声的 `1.11e-9`。

因此当前 Active/Passive 差异主要是固定幅度增益造成的，不是放大噪声
与输出功率约束共同作用形成的折中。代码现在会为新仿真额外保存
`irs_gain` 和 `irs_output_power`，便于直接检查约束是否激活。最终
NPZ 已包含全部诊断字段，且所有 72 个数组均为有限数。

本地 Active-IRS 参数分区验证进一步表明，当前 -80 dBm
单元噪声要分别提高到约 -6.15、-10.04、-13.08、-16.13 dBm，N=16、
32、64、128 的 CU 端放大噪声才会达到接收机噪声量级。这不是建议采用
这些参数，而是说明当前 IRS→CU 路损下，通信侧噪声效应离可见区很远。
下一步应优先核对论文的噪声定义是“每单元噪声功率”还是包含噪声系数、
带宽和前级增益的等效噪声，然后再选择物理合理的扫描范围。

直达 BS-CU 与 IRS-CU 链路现在使用独立的局部随机种子 46 和 47；
旧实现会重置全局随机状态，并令两条链路的 NLoS 样本出现人为相关。
不同 N 的 IRS-CU 信道仍共享种子 47，以保持可重复的阵元数比较；由于
导向矢量采用阵列中点作为相位参考，不应把不同 N 的完整信道误解为简单
的数组前缀。

AO 现在从闭式目标相位对齐解初始化，而不是全 1 相位。N=8、0 dB 的
小型回归中，Passive AO 相对对齐基线 CRB 变化约 `-7.6e-5 dB`，
Active AO 约 `-2.1e-8 dB`；至少不再出现 AO 比简单基线差 4.49 dB
的初始化伪象。该 JSON 属于本地验证产物，不提交到 Git。
