# LLR 抗差估计、Bias 与方差分量估计：理论和迭代方案

## 1. 目标和适用范围

本文给出 LLR 正常点参数估计的统一统计模型。目标是在**不对输入观测
sigma 作经验性分段缩放**的前提下，同时处理：

1. 测站和时期相关的系统距离 bias；
2. 单个正常点的大粗差；
3. 测站、反射器和时期之间的随机精度差异；
4. 非线性动力学/几何模型下的参数迭代。

前提是已经能够给出可信的测站 bias 变更时期，并能给出
`station x period` 的初始 VCE 分组。输入 sigma 保留为每条观测的
先验相对精度；VCE 只估计其组尺度，而不改写原始 sigma。

本文中的推荐层次为：

```text
物理与 bias 参数的 Gauss-Newton 解算
        <- 抗差等价权迭代
                <- VCE 方差分量迭代
```

因此，Gauss-Newton 是最内层，抗差 IRLS 是中层，VCE 是最外层。

## 2. 文献依据

本方案的 SLR 依据如下。

* Sahin, Cross, Sellers (1992), *Variance component estimation applied to
  satellite laser ranging*, Bulletin Geodesique 66, 284--295,
  [DOI](https://doi.org/10.1007/BF02033189)。该文将 Helmert VCE 嵌入
  SLR 参数估计，以组协方差的乘性尺度表示方差分量，并重复“参数解算 ->
  VCE -> 重解算”。它同时指出，未建模的系统误差不应被误解释为随机权重。
* Yang, Cheng, Shum, Tapley (1999), *Robust estimation of systematic errors
  of satellite laser range*, Journal of Geodesy 73, 345--349,
  [DOI](https://doi.org/10.1007/s001900050252)。该文先以高崩溃点尺度获得
  可靠初值，再估计 station-dependent range/time bias，并以三段等价权函数
  迭代抑制粗差；其明确指出普通最小二乘存在 masking。
* Li et al. (2024), *Improving multiple LEO combination for SLR-based
  geodetic parameters determination using variance component estimation*,
  Journal of Geodesy 98, [DOI](https://doi.org/10.1007/s00190-024-01880-z)。
  该文比较 satellite、station 与 satellite-station pair 三种 VCE 分组；
  pair 分组表现最好，但短弧中观测不足会使 VCE 不可靠。作者以较长的月弧
  导出稳定权重，再用于周弧解算。
* Teunissen and Amiri-Simkooei (2008), *Least-squares variance component
  estimation*, Journal of Geodesy 82, 65--82,
  [DOI](https://doi.org/10.1007/s00190-007-0157-x)。本文使用其 LS-VCE
  表达式作为一般多方差分量的理论基础。

Li et al. 的结论支持把 `station x reflector` 作为**随机模型的候选交互组**，
但不支持把 station-reflector bias 无约束地与所有物理参数同步自由估计。
其 SLR 处理中，长期 satellite-station range bias 单独标定后作为已知改正
使用，以避免短弧 bias 与站坐标等参数的强相关。

## 3. 统一观测模型

令第 `i` 条 LLR 正常点的观测减计算值为 `l_i`。在当前非线性状态
`x^(k)` 附近线性化：

```math
l = A delta_x + B delta_b + e + o.
```

符号如下。

* `delta_x`：月球/地球动力学、反射器坐标、站坐标等科学参数的修正；
* `delta_b`：已知测站 bias 时段对应的 range-bias 参数；
* `A`、`B`：分别为上述两类参数的设计阵；
* `e`：零均值随机测量误差；
* `o`：稀疏的大粗差或污染项。

下文以 `H = [A B]` 表示全部待估参数的设计阵。只要 bias 仍是未知量，
它就必须包含在 `H` 内；不能先用忽略 bias 的残差做 VCE 或粗差判别。

### 3.1 Bias 是函数模型，不是随机模型

对已知的测站变更时期，定义互斥的组 `h = (station, period)`，并引入：

```math
b_i = b_h,   i in h.
```

也就是相应行的 `B[i,h] = 1`。区间应优先为不重叠分割；若使用“长期基准
bias + 短期事件修正”的重叠区间，必须有明确物理含义，并检查参数相关系数
和秩亏。

`b_h` 解释的是组内残差的均值偏移。VCE 解释的是去除均值后残差的离散度。
若某组残差均值显著不为零，先检查或扩展 `B`，不能仅把该组的方差分量调大。
这是 Sahin et al. 对 SLR VCE 最重要的限制条件。

### 3.2 随机模型

记输入文件提供的一程标准差为 `sigma_i0`，且 `g(i)` 为第 `i` 条观测的
VCE 组。最小的可实施模型是：

```math
Q(theta) = sum_g theta_g Q_g,
Q_g[i,i] = sigma_i0^2  if g(i)=g, otherwise 0,
theta_g > 0.
```

故有效方差为 `theta_g sigma_i0^2`，其中 `sqrt(theta_g)` 是对**输入 sigma
的组尺度校准**。这保留了同组内原始 sigma 的相对信息。

如果诊断表明 formal sigma 只反映法点内部散布、而总误差还存在与其无关的
底噪，则可扩展为：

```math
Var(e_i) = theta_g sigma_i0^2 + tau_parent(g)^2.
```

第二项应先定义在数据足够的父组，例如 `station x era`，不宜在每个小
`station x reflector x period` 组内同时估计两个分量。第一阶段建议仅采用
乘性模型；只有其标准化残差仍与 `sigma_i0` 系统相关时，才引入底噪项。

### 3.3 分组的层次

建议定义嵌套的随机组，而不是一次性让最细组全部自由：

```text
G1: station x period                         主模型
G2: station x reflector x period             交互对照模型
G3: 更长的 station x reflector 校准窗口       稀疏组的权重来源
```

Li et al. 的 satellite-station pair 结果是 `G2` 的直接 SLR 类比。它只能说明
pair 级随机异方差可能存在；若 `station x reflector x period` 残差有稳定的
均值结构、libration/phase/elevation 依赖或参数相关性，则应先作为函数模型
缺项处理，而不是宣布为方差差异。

## 4. 固定权阵下的参数估计

在给定 `theta`、抗差权 `w_i` 和当前线性化点时，定义工作权阵：

```math
P = diag( w_i / (theta_{g(i)} sigma_i0^2) ).
```

若有约束或先验，记其法方程为 `N_c, n_c`。一次 Gauss-Newton 更新为：

```math
(H^T P H + N_c) delta = H^T P l + n_c.
```

随后以阻尼系数 `lambda` 更新：

```math
x <- x + lambda delta_x,
b <- b + lambda delta_b.
```

必须重新计算光行时、O-C 和设计阵，而不能把几何类参数的更新只留在线性残差中。
在固定 `theta,w` 下，重复该步骤直到参数更新、目标函数或后验 WRMS 收敛。

这个步骤只解决“在当前随机模型下哪个参数最优”；它不决定哪条观测是粗差，
也不决定不同组应该有多大方差。

## 5. 抗差估计

### 5.1 为什么不用原始 3sigma/5sigma

`|v_i| > k sigma_i0` 把输入 sigma 当成最终残差标准差，忽略了：

1. bias 和物理参数已吸收的部分；
2. 参数估计引起的残差杠杆效应；
3. 组方差分量 `theta_g`；
4. 粗差对普通最小二乘参数和残差的 masking。

最终统计检验应基于残差协方差。对无抗差权的 GLS，

```math
Q_vv = Q(theta) - H (H^T Q(theta)^(-1) H)^(-1) H^T,
t_i = v_i / sqrt(Q_vv[i,i]).
```

`t_i` 才是用于最终粗差报告的标准化残差。IRLS 内部也可使用该量；若为降低
计算复杂度而使用 `v_i / sqrt(theta_g sigma_i0^2)`，最终仍必须以 `Q_vv`
重新检查。

### 5.2 高崩溃点初始尺度

Yang et al. 以中位数绝对偏差构造初始尺度：

```math
s0 = median_i |r_i| / 0.6745,
```

其中初始 `r_i` 是以当前先验 sigma 标准化的残差。LLR 中应在可用的父组上
计算，例如 station x era；小组不单独计算 MAD。若组内 bias 已由 `B` 表示，
应在估计该 bias 后再计算 MAD。

该步骤只用于获得不受少量大粗差支配的初始尺度。它不是对观测 sigma 的经验
重标定，也不应作为最终 VCE 结果。

### 5.3 三段等价权函数

采用 Yang et al. 所用的红降型三段 M 估计可写为：

```math
w(t) =
  1,                                                   |t| <= c0
  c0/|t| * ((c1-|t|)/(c1-c0))^2,        c0 < |t| <= c1
  0,                                                   |t| > c1.
```

文中给出的典型范围为 `c0 = 1.0--1.5`、`c1 = 3.0--6.0`；它们应以 LLR
仿真或历史保留集校准，而不能把 SLR 数值直接宣布为 LLR 的固定阈值。

首轮可以按 `|r_i|/s0` 做保守临时隔离，以获得不被大粗差拉偏的 bias 初值。
Yang et al. 的做法是在后续等价权迭代中允许这些观测重新参与；LLR 也应保留
此 re-entry，而非首轮永久删除。

### 5.4 IRLS 循环

在固定 `theta` 下：

1. 令 `w_i=1`，或使用上一外层循环的稳定权；
2. 以第 4 节方法把 `x,b` 解至 Gauss-Newton 收敛；
3. 回代计算残差、标准化残差 `t_i` 和父组 MAD 尺度；
4. 用 `w(t_i)` 更新等价权；
5. 若权值、参数或有效观测集变化仍超过阈值，回到第 2 步。

这保证 bias 是由未被少数大粗差支配的观测估计的。抗差权只抑制孤立污染；
一个组内成片的均值残差不能靠降权“解决”。

### 5.5 抗差与 VCE 的统计接口

标准 LS-VCE 假设残差来自未污染的高斯模型。把已经 redescend 的 `w_i`
直接当作真实协方差逆阵再代入 LS-VCE，会改变二次型期望，导致方差分量偏小。
因此应明确选择以下两种实现级别之一。

**第一阶段，推荐实现：稳健预处理后的 LS-VCE。** IRLS 收敛后，将 `w_i`
足够小的观测标记为污染候选；VCE 仅在稳定内点上以 `Q(theta)^(-1)` 计算，
但保留被排除行及理由。该法容易验证，但应在报告中称为“robust-preedited
LS-VCE”，不是严格的联合稳健 VCE。

**第二阶段，严格联合稳健 VCE。** 以 `psi(t)=w(t)t` 替换残差二次型，并使用
高斯一致性因子 `c_psi = E[psi(Z)^2]` 校正各组尺度；同时对 LS-VCE 的得分
向量和信息阵作相应修正。只有在第一阶段结果稳定后才值得实现，因为其公式和
协方差评定比普通 LS-VCE 复杂得多。

无论采用哪一种，VCE 绝不能在明显大粗差仍具全权的残差上执行。

## 6. LS-VCE 的细节

### 6.1 一般形式

对 `Q(theta) = sum_g theta_g Q_g`，令 `W=Q(theta)^(-1)`，

```math
P_A_perp = I - H (H^T W H)^(-1) H^T W.
```

LS-VCE 的方程为：

```math
N_theta gamma = l_theta,

N_theta[g,h] = 1/2 tr(Q_g W P_A_perp Q_h W P_A_perp),
l_theta[g]   = 1/2 v^T W Q_g W v.
```

`gamma_g` 是当前迭代相对组尺度的修正因子；一次更新为：

```math
theta_g <- theta_g * gamma_g.
```

计算时必须保证 `theta_g > 0`。推荐在 `eta_g=log(theta_g)` 空间中对更新阻尼，
例如 `eta <- eta + rho log(gamma)`，其中 `0 < rho <= 1`。这比允许 Helmert
方程产生非正方差分量后再修补更稳定。

`H` 在上述公式中包含 bias 列。因此，VCE 使用的是已经扣除所有待估物理参数
和 bias 后的残差自由度。

### 6.2 互斥分组的简化与冗余度

当组互斥、`Q_g` 为对角块时，可保留每组的法方程贡献：

```math
N_g = H_g^T W_g H_g,
r_g = n_g - tr( N^(-1) N_g ).
```

`r_g` 是组有效冗余度，不等于普通观测数 `n_g`。它决定该组方差是否可估；
仅有很多但几乎全部被 bias 或其他参数吸收的观测，并不能提供可靠 VCE。

用于调试的近似诊断为：

```math
gamma_g approximately (v_g^T W_g v_g) / r_g.
```

最终多组结果仍应以一般 LS-VCE 方程为准，因为组之间会通过共同参数耦合。

### 6.3 VCE 外层循环

Li et al. 的 SLR 流程是在每次参数解后回代残差，估计方差分量，更新协方差，
再解算。其多 LEO 实验通常 3--4 次达到稳定，并以最多 10 次作为保护上限。

LLR 外层循环一次应执行：

1. 固定 `theta`，完成一轮 IRLS 收敛的非线性参数估计；
2. 用稳定内点或严格稳健得分构造 LS-VCE；
3. 解出 `gamma`，做正性约束和对数阻尼，更新 `theta`；
4. 检查 `max_g |log(gamma_g)|`、全局物理参数变化、bias 变化和组内点集变化；
5. 未共同收敛则回到第 1 步。

Li et al. 对短弧使用较宽松的相对尺度比和 10 次上限；LLR 的生产解应采用
更直接的收敛准则，例如所有 `|log(gamma_g)|` 小于预设容限，并记录每组的
有效冗余度和迭代轨迹。

## 7. 完整循环

### 7.1 解算前的固定输入

在进入迭代前固定下列内容，避免用同一批残差反复选择模型：

1. 确定性物理模型、数据时间覆盖和正常点有效性规则；
2. station-bias 的互斥时段表；
3. VCE 候选组表：至少 `station x period`，可选嵌套
   `station x reflector x period`；
4. 每条正常点的原始 `sigma_i0`、bias 组、VCE 组和父组；
5. 基础绝对屏蔽规则：解析失败、光行时未收敛、非法测量值以及物理上不可能的
   极端 O-C。此处不使用 3sigma/5sigma。

### 7.2 Bias 标定与生产解

若解算弧很长且每个 bias 时段有足够冗余，`b_h` 可和全局参数一起估计。若要
对短弧、周解或分段解使用这些 bias，则采用与 Li et al. 相同的两阶段思想：

1. **长弧标定解**：在包含完整 bias 时段的长数据弧内，稳健估计 `b_h`，并
   估计或校验其不确定度；
2. **生产解**：将 `b_h` 固定为标定值，或以 `b_h ~ N(b_h*, C_b)` 的软约束
   引入。不要在每个短弧重新自由估计同一 bias。

这样能减少 bias 与反射器坐标、动力学参数和短弧随机权之间的相互吸收。

### 7.3 伪代码

```text
prepare records, H-bias group map, VCE group map, sigma0
theta_g <- 1 for all groups
w_i <- 1 for all usable observations

for VCE outer iteration q:
    for robust iteration r:
        for Gauss-Newton iteration k:
            evaluate nonlinear LLR model at x, b
            form l, H=[A B], P=diag(w_i/(theta_g(i)*sigma0_i^2))
            solve constrained normal equations for delta_x, delta_b
            update x, b; re-evaluate model
            stop when fixed-weight nonlinear solve converges

        back-substitute residuals and residual covariance
        estimate robust scales from parent groups using MAD
        update w_i with the chosen M-estimator
        stop when w, x, b and active set converge

    form LS-VCE equations from robust-preedited inliers
      or from consistency-corrected robust VCE scores
    solve for gamma_g; enforce positivity and apply the bounded theta_g update directly
    theta_g <- theta_g * gamma_g
    stop only when theta, x, b, w and group diagnostics jointly converge

freeze the final stochastic model
compute standardized residuals without leverage correction and final outlier flags
re-solve once after final, documented data decision
```

最后一次“冻结后重解算”很重要：它把估计阶段的临时抗差权与发表/交付阶段的
明确数据选择分开，避免每一轮都永久删点。

## 8. 观测不足与权重回退

Li et al. 的核心工程结论是：pair 分组较好，但短弧的低产站会给出不可靠 VCE，
甚至降低参数精度；将月解权用于周解可恢复稳定性。LLR 应采用同一原则。

对每个细组 `station x reflector x period`，先检查 `n_g`、`r_g`、零权观测比例
和 VCE 方程条件数。若不足，则按以下优先级回退：

1. 在该站-反射器的更长时间窗口中标定 `theta_g`，并将其作为本期固定权；
2. 回退为同一 `station x period` 的方差分量；
3. 再回退为 station x era 的父组。

不要把稀疏细组的自由 VCE 结果直接应用到物理参数解。长窗口权重是“用更多
同类观测提高方差分量可估性”，不是重新引入人工 sigma 表。

## 9. 最终粗差判定与报告

抗差权不是最终科学数据剔除清单。最终处理应在 `x,b,theta` 冻结后进行：

1. 计算 `t_i=v_i/sqrt(Q_vv[i,i])`；
2. 采用预先声明的全局显著性控制、最大标准化残差检验或模拟校准阈值；
3. 将被拒绝点、其初始/最终权、bias 组、VCE 组、原始 sigma、残差和原因
   全部输出；
4. 删除或固定零权后再做一次完整解算，报告参数变化。

固定 3sigma 或 5sigma 只能作为内部参考，不应再直接与 `sigma_i0` 相乘作为
唯一的最终判据。多个时期、测站和反射器同时检验时，还必须处理多重检验问题。

## 10. 诊断和验收

每个 VCE 组至少输出：

* 观测数、有效冗余度、最终方差分量及其迭代轨迹；
* bias 估值和不确定度；
* 残差均值、中位数、MAD、WRMS、标准化残差分位数；
* 零权/低权/最终剔除的观测比例；
* 与全局物理参数和 reflector 坐标的相关性。

接受 `station x reflector x period` 细分模型的条件应包括：

1. 相比 `station x period`，它在留出时段或留出 session 的预测误差上改善；
2. 各组标准化残差的中心接近零、尺度接近一；
3. 全局物理参数不因合理改变时间边界而发生超出不确定度的漂移；
4. 改善不是靠把某些早期时期或 APOLLO 观测整体降至近零权获得；
5. 每个细组均有足够冗余，或明确使用了长窗口/父组回退。

抗差 M 估计后的常规 `(H^T P H)^(-1)` 不能单独作为最终参数协方差。最终
不确定度应至少报告 VCE 后的权阵；对抗差带来的附加不确定性，宜采用 sandwich
协方差或按 station-period block 的重采样评估。

## 11. 与当前 LLROPS 架构的关系

当前实现已具备非线性 `LlrAdjustment`、station interval bias 和流式法方程。
接入本文方案时，估计器需要增加：

1. 每条观测的不变 `sigma_i0`、VCE 组键和抗差权；
2. 每组法方程贡献、残差、有效冗余度和 VCE 方程所需的 trace 项；
3. 固定 `theta,w` 的 Gauss-Newton 内层、IRLS 中层和 VCE 外层控制器；
4. 细组的长窗口权重导入/父组回退；
5. 逐观测和逐组的可审计诊断输出。

现有只存总 `N,W,lPl` 的法方程不足以在不重跑观测的情况下完成 VCE；至少应
保存或重建每个 VCE 组的贡献。对当前 LLR 正常点规模，重跑流式观测方程以得到
残差和组统计通常是可接受且更不易出错的实现起点。
