# LLR 分组 VCE、分段 Bias 与 IGGⅢ 抗差联合估计技术设计

**文档状态：实现版**  
**目标读者：负责将算法接入现有 LLR 解算代码库的开发者 / Codex**

---

## 1. 目的与适用范围

本文档定义一套用于 LLR（Lunar Laser Ranging，月球激光测距）normal point 数据处理的联合随机模型与抗差估计流程，包含：

1. 按测站、设备或观测系统划分的分组方差分量估计；
2. 按指定测站和时间区间设置的分段 range bias；
3. 预拟合残差上的稳健 Bias 初值估计；
4. 基于 MAD 的组尺度启动值；
5. 基于 IGGⅢ 等价权函数的抗差估计；
6. IGGⅢ 等价权下的简化 Helmert 分组 VCE；
7. 参数、Bias、抗差权和方差分量的联合迭代；
8. 完整的诊断、日志和测试要求。

本文档面向已有 LLR 函数模型和参数估计框架的代码库。现有轨道积分、月球转动、测站、反射器、地球定向参数及其他物理模型不在本文档中重新定义。

---

## 2. 当前版本的固定决策

当前实现采用以下固定方案：

1. 随机模型只估计每个 VCE 组的 NP 正式精度比例因子 $s_g^2$；
2. 不估计加性噪声底；
3. 不估计观测夜公共相关误差；
4. NP 正式精度 $\sigma_{i,\mathrm{NP}}$ 用于构造组内相对权；
5. VCE 分组采用 EPM2023a 结果表中的测站和系统划分；
6. 为覆盖当前输入，Apache、CERGA MeO/IR、Matera 和 Wettzell 分组均扩展至 `present`；
7. Bias 区间采用给定的 28 个区间，并新增 Wettzell 2018-01-01 至今的全时段 Bias；
8. Bias 区间允许重叠，重叠 Bias 按加法叠加；
9. 抗差估计采用 IGGⅢ 等价权函数；
10. IGGⅢ 阈值固定为
    $$
    k_0=1.5,\qquad k_1=6.0;
    $$
11. 正式抗差前，先用稳健 Bias 初值和 MAD 组尺度完成初始化；
12. VCE 多余度采用 Helmert 迹公式，不采用“分数观测数”形式；
13. IGGⅢ 严格零权观测不计入当前有效观测数；
14. 随机模型收敛后，固定最终 $s_g^2$ 和 IGGⅢ 因子，再执行一次完整最终解算。

---

## 3. 符号与数据约定

### 3.1 主要符号

| 符号 | 含义 |
|---|---|
| $n$ | 总观测数 |
| $u$ | 当前可估参数秩 |
| $G$ | VCE 组数 |
| $\boldsymbol O$ | 观测值 |
| $\boldsymbol C(\boldsymbol x_0)$ | 初始参数处的计算值 |
| $\boldsymbol l=\boldsymbol O-\boldsymbol C$ | 预拟合观测减计算值 |
| $\boldsymbol A$ | 现有物理参数的设计矩阵 |
| $\boldsymbol B$ | Bias 指示矩阵 |
| $\boldsymbol M=[\boldsymbol A\ \boldsymbol B]$ | 完整设计矩阵 |
| $\Delta\boldsymbol x$ | 物理参数改正 |
| $\Delta\boldsymbol b$ | Bias 参数改正 |
| $\boldsymbol v$ | post-fit residual |
| $\sigma_{i,\mathrm{NP}}$ | 第 $i$ 个 NP 的正式标准差 |
| $s_g$ | 第 $g$ 组标准差比例因子 |
| $s_g^2$ | 第 $g$ 组方差分量 |
| $\alpha_i$ | 第 $i$ 个观测的 IGGⅢ 抗差因子 |
| $r_g$ | 第 $g$ 组有效多余度 |

### 3.2 推荐内部单位

建议代码内部统一使用：

- 距离及 range bias：米；
- 时间：现有代码统一的时间尺度；
- 标准差：米；
- 方差：平方米；
- $s_g$、$\alpha_i$：无量纲。

如果输入是双程光行时，必须在数据入口统一转换，避免 Bias 和残差使用不同单位。

### 3.3 日期区间语义

配置文件统一使用左闭右开区间：

$$
[\text{start},\text{end\_exclusive}).
$$

`present` 在程序中表示无上界：

```yaml
end_exclusive: null
```

不得在读取配置时把 `present` 替换成当天日期。

---

## 4. 观测模型

预拟合观测减计算值为

$$
\boldsymbol l
=
\boldsymbol O-\boldsymbol C(\boldsymbol x_0).
$$

线性化观测模型写为

$$
\boldsymbol l
=
\boldsymbol A\Delta\boldsymbol x
+
\boldsymbol B\Delta\boldsymbol b
+
\boldsymbol e.
$$

定义

$$
\boldsymbol M
=
\begin{bmatrix}
\boldsymbol A&\boldsymbol B
\end{bmatrix},
\qquad
\Delta\boldsymbol y
=
\begin{bmatrix}
\Delta\boldsymbol x\\
\Delta\boldsymbol b
\end{bmatrix},
$$

则

$$
\boldsymbol l
=
\boldsymbol M\Delta\boldsymbol y+\boldsymbol e.
$$

残差统一定义为

$$
\boldsymbol v
=
\boldsymbol l-\boldsymbol M\Delta\hat{\boldsymbol y}.
$$

如果现有代码采用相反的残差符号，可以保留原有约定，但必须保证：

- 所有日志和公式说明一致；
- Bias 符号与观测方程一致；
- IGGⅢ 使用 $|t_i|$，不依赖残差正负号。

---

## 5. 随机模型

### 5.1 基础协方差模型

当前版本只估计分组尺度：

$$
\boxed{
\boldsymbol\Sigma
=
\sum_{g=1}^{G}
s_g^2\boldsymbol Q_g
}
$$

其中

$$
\boldsymbol Q_g
=
\operatorname{diag}
\left(
\delta_{ig}\sigma_{i,\mathrm{NP}}^2
\right),
$$

且

$$
\delta_{ig}
=
\begin{cases}
1,&i\in g,\\
0,&i\notin g.
\end{cases}
$$

对观测 $i\in g$：

$$
\operatorname{Var}(e_i)
=
s_g^2\sigma_{i,\mathrm{NP}}^2.
$$

基础标准差为

$$
\sigma_{i,\mathrm{base}}
=
s_g\sigma_{i,\mathrm{NP}},
$$

基础权为

$$
\boxed{
p_i
=
\frac{1}{s_g^2\sigma_{i,\mathrm{NP}}^2}
}
$$

### 5.2 模型含义

- $\sigma_{i,\mathrm{NP}}$ 保留同一 VCE 组内的相对精度；
- $s_g^2$ 修正该组整体正式方差的尺度；
- VCE 不改变同一组内两个观测的形式权比；
- 本版本不包含 $\sigma_{0,g}^2\boldsymbol I$；
- 本版本不包含观测夜或标定周期的相关协方差块。

### 5.3 数据校验

每个 NP 必须满足：

```text
sigma_np > 0
sigma_np is finite
恰好命中一个 VCE 组
```

无法分组的观测不得静默使用默认权。

---

## 6. VCE 分组

### 6.1 完整分组表

| VCE 组 ID | 测站/系统 | 时间范围 | 程序区间 |
|---|---|---|---|
| `MCDONALD_1969_1985` | McDonald | 1969–1985 | `[1969-01-01, 1986-01-01)` |
| `MLRS1_1983_1988` | MLRS1 | 1983–1988 | `[1983-01-01, 1989-01-01)` |
| `MLRS2_1988_2015` | MLRS2 | 1988–2015 | `[1988-01-01, 2016-01-01)` |
| `HALEAKALA_1984_1990` | Haleakala | 1984–1990 | `[1984-01-01, 1991-01-01)` |
| `CERGA_RUBY_1984_1986` | CERGA Ruby | 1984–1986 | `[1984-01-01, 1987-01-01)` |
| `CERGA_YAG_1987_2005` | CERGA YAG | 1987–2005 | `[1987-01-01, 2006-01-01)` |
| `CERGA_MEO_2009_PRESENT` | CERGA MeO | 2009–present | `[2009-01-01, +∞)` |
| `CERGA_IR_2015_PRESENT` | CERGA IR | 2015–present | `[2015-01-01, +∞)` |
| `APACHE_2006_PRESENT` | Apache | 2006–present | `[2006-01-01, +∞)` |
| `MATERA_2003_PRESENT` | Matera | 2003–present | `[2003-01-01, +∞)` |
| `WETTZELL_2018_PRESENT` | Wettzell | 2018–present | `[2018-01-01, +∞)` |

### 6.2 系统识别要求

VCE 分组不能只依据日期。

以下分组存在年份重叠：

- McDonald 与 MLRS1；
- MLRS1 与 MLRS2 的 1988 年；
- CERGA MeO 与 CERGA IR 的 2015 年至今。

分组键必须优先使用：

$$
\boxed{
\text{station identifier}
+
\text{system/configuration identifier}
+
\text{epoch}
}
$$

推荐读取字段：

```text
station_id
station_name
system_config_id
system_name
wavelength
observation_mode
epoch
```

CERGA MeO 和 CERGA IR 必须依靠系统配置、波长或数据产品中的明确模式字段区分，禁止只按日期猜测。

### 6.3 分组配置示例

```yaml
vce_groups:
  - id: MCDONALD_1969_1985
    station_system: MCDONALD
    start: 1969-01-01
    end_exclusive: 1986-01-01

  - id: MLRS1_1983_1988
    station_system: MLRS1
    start: 1983-01-01
    end_exclusive: 1989-01-01

  - id: MLRS2_1988_2015
    station_system: MLRS2
    start: 1988-01-01
    end_exclusive: 2016-01-01

  - id: HALEAKALA_1984_1990
    station_system: HALEAKALA
    start: 1984-01-01
    end_exclusive: 1991-01-01

  - id: CERGA_RUBY_1984_1986
    station_system: CERGA_RUBY
    start: 1984-01-01
    end_exclusive: 1987-01-01

  - id: CERGA_YAG_1987_2005
    station_system: CERGA_YAG
    start: 1987-01-01
    end_exclusive: 2006-01-01

  - id: CERGA_MEO_2009_PRESENT
    station_system: CERGA_MEO
    start: 2009-01-01
    end_exclusive: null

  - id: CERGA_IR_2015_PRESENT
    station_system: CERGA_IR
    start: 2015-01-01
    end_exclusive: null

  - id: APACHE_2006_PRESENT
    station_system: APACHE
    start: 2006-01-01
    end_exclusive: null

  - id: MATERA_2003_PRESENT
    station_system: MATERA
    start: 2003-01-01
    end_exclusive: null

  - id: WETTZELL_2018_PRESENT
    station_system: WETTZELL
    start: 2018-01-01
    end_exclusive: null
```

### 6.4 全观测覆盖

McDonald VCE 组从 1969-01-01 开始，与其长期 Bias 区间保持一致，因此当前输入中的 1969 年观测参与 `MCDONALD_1969_1985` 组。

当前输入还包含 2023 年后的 532.1 nm CERGA 观测。它们由波长明确识别为 MeO，因此 `CERGA_MEO_2009_PRESENT` 保持开放结束端点；CERGA IR 同样保持开放端点。两组仍必须按系统配置、波长或明确模式字段区分，禁止只按日期猜测。

### 6.5 未分组处理

默认策略：

```yaml
unassigned_observation_policy: error
multiple_match_policy: error
```

每个观测必须恰好命中一个 VCE 组。

---

## 7. Bias 模型

### 7.1 Bias 指示矩阵

对第 $j$ 个 Bias：

$$
B_{ij}
=
\begin{cases}
1,&\text{观测 }i\text{ 命中 Bias 区间 }j,\\
0,&\text{其他}.
\end{cases}
$$

一个观测可以命中多个 Bias 区间。

如果长期 Bias 与短期增量 Bias 重叠：

$$
l_i
=
\boldsymbol a_i^T\Delta\boldsymbol x
+
b_{\mathrm{long}}
+
b_{\mathrm{short}}
+
e_i.
$$

Bias 区间与 VCE 分组相互独立：

- Bias 区间可以重叠；
- 每条观测只能属于一个 VCE 组；
- Bias 区间不自动切分 VCE 组。

### 7.2 完整 Bias 区间表

日期按原始表中的 DD.MM.YYYY 转换。

| ID | Station | From | To |
|---:|---|---|---|
| 1 | Apache | 2006-04-07 | 2010-11-01 |
| 2 | Apache | 2007-12-15 | 2008-06-30 |
| 3 | Apache | 2008-09-20 | 2009-06-20 |
| 4 | Apache | 2010-11-01 | 2012-04-07 |
| 5 | Apache | 2012-04-07 | 2013-09-02 |
| 6 | CERGA | 1984-06-01 | 1986-06-13 |
| 7 | CERGA | 1987-10-01 | 2005-08-01 |
| 8 | CERGA | 1996-12-10 | 1997-01-18 |
| 9 | CERGA | 1997-02-08 | 1998-06-24 |
| 10 | CERGA | 2004-12-04 | 2004-12-07 |
| 11 | CERGA | 2005-01-03 | 2005-01-06 |
| 12 | CERGA | 2009-11-01 | 2014-01-01 |
| 13 | Haleakala | 1984-11-01 | 1990-09-01 |
| 14 | Haleakala | 1984-11-01 | 1986-04-01 |
| 15 | Haleakala | 1986-04-02 | 1987-07-30 |
| 16 | Haleakala | 1987-07-31 | 1987-08-14 |
| 17 | Haleakala | 1985-06-09 | 1985-06-10 |
| 18 | Haleakala | 1989-01-28 | 1989-01-29 |
| 19 | Haleakala | 1989-08-23 | 1989-08-24 |
| 20 | Haleakala | 1990-02-06 | 1990-09-01 |
| 21 | McDonald | 1969-01-01 | 1985-07-01 |
| 22 | McDonald | 1971-12-01 | 1972-12-05 |
| 23 | McDonald | 1972-04-21 | 1972-04-27 |
| 24 | McDonald | 1974-08-18 | 1974-10-16 |
| 25 | McDonald | 1975-10-05 | 1976-03-01 |
| 26 | McDonald | 1983-12-01 | 1984-01-17 |
| 27 | Matera | 2003-01-01 | 2016-01-01 |
| 28 | MLRS1 | 1983-08-01 | 1988-01-28 |
| 29 | Wettzell | 2018-01-01 | present |

### 7.3 Bias 日期端点

原表中的 `From` 和 `To` 按日期闭区间解释：

$$
\text{from}\le\text{epoch date}\le\text{to}.
$$

加载配置时转换为：

$$
[\text{from},\text{to}+1\text{ day}).
$$

Wettzell 的 `present` 使用：

```yaml
end_exclusive: null
```

### 7.4 Bias 配置示例

```yaml
bias_intervals:
  - id: 1
    name: APACHE_20060407_20101101
    station: APACHE
    start: 2006-04-07
    end_exclusive: 2010-11-02

  # 中间区间按完整表配置

  - id: 29
    name: WETTZELL_2018_PRESENT
    station: WETTZELL
    start: 2018-01-01
    end_exclusive: null
```

### 7.5 全时段 Bias 的相关性

Wettzell 2018 至今的全时段 Bias，以及其他覆盖测站主要运行期的 Bias，可能与以下参数高度相关：

- 测站径向坐标；
- 反射器径向坐标；
- 地月距离尺度；
- 月球轨道径向初值；
- 常数型大气模型偏差。

必须输出

$$
\operatorname{corr}
\left(
b_j,h_{\mathrm{station}}
\right),
$$

并检查：

- 法方程秩；
- 条件数；
- Bias 形式误差；
- Bias 与测站坐标的相关系数；
- Bias 与反射器径向坐标的相关系数。

如果秩亏或高度相关，应使用外部坐标约束、固定参考 Bias 或其他既有基准策略处理。Bias 初值不能解决可辨识性问题。

---

## 8. Bias 初值

### 8.1 基本原则

Bias 表示残差中心位置，MAD 表示残差离散尺度，二者不能混用。

Bias 是线性参数，理论上可以设置：

$$
\boldsymbol b^{(0)}=\boldsymbol0.
$$

但是在 NP 正式误差很小、真实 Bias 达到厘米级时，零初值可能导致第一次抗差判断前大量正常观测产生很大的标准化残差。

因此推荐使用稳健 Bias 初值。

### 8.2 稳健回归初始化

预拟合残差：

$$
\boldsymbol l
=
\boldsymbol O-\boldsymbol C(\boldsymbol x_0).
$$

求解：

$$
\boxed{
\boldsymbol b^{(0)}
=
\arg\min_{\boldsymbol b}
\sum_i
\rho
\left(
\frac{
l_i-\boldsymbol B_i\boldsymbol b
}{
\widetilde\sigma_i
}
\right)
}
$$

推荐实现：

- Huber IRLS；
- LAD 回归；
- 带权重上限的稳健回归。

启动阶段不要让极小 $\sigma_{i,\mathrm{NP}}$ 获得无限大权重。可使用：

$$
\widetilde p_i
=
\min
\left(
\frac{1}{\sigma_{i,\mathrm{NP}}^2},
p_{\max}
\right).
$$

也可以在 Bias 初始化阶段临时等权。

### 8.3 重叠 Bias

重叠区间由设计矩阵自动处理。

例如一条观测同时命中长期 Bias 1 和短期 Bias 2：

$$
\boldsymbol B_i
=
[1,1,0,\ldots].
$$

禁止分别对每个重叠区间独立取中位数后直接作为绝对 Bias，因为会重复计算公共部分。

### 8.4 回退策略

如果稳健回归失败：

```text
bias_initialization_status = FALLBACK_ZERO
```

并设置：

$$
\boldsymbol b^{(0)}=\boldsymbol0.
$$

必须保留失败原因和诊断日志。

### 8.5 正式解算

初值不能固定：

$$
\hat{\boldsymbol b}
=
\boldsymbol b^{(0)}
+
\Delta\hat{\boldsymbol b}.
$$

Bias 必须在正式联合平差中继续作为未知参数。

---

## 9. MAD 初始化组尺度

### 9.1 去除 Bias 初值

$$
r_i^{(0)}
=
l_i-\boldsymbol B_i\boldsymbol b^{(0)}.
$$

### 9.2 按 NP 正式误差标准化

对第 $g$ 组：

$$
z_i^{(0)}
=
\frac{
r_i^{(0)}
}{
\sigma_{i,\mathrm{NP}}
},
\qquad i\in g.
$$

不能直接对毫米或米单位的残差计算 MAD 后将其作为无量纲 $s_g$。

### 9.3 MAD 启动值

$$
\boxed{
s_g^{(0)}
=
1.4826
\operatorname{median}_{i\in g}
\left|
z_i^{(0)}
-
\operatorname{median}_{j\in g}
z_j^{(0)}
\right|
}
$$

启动阶段使用：

$$
\boxed{
s_g^{(0)}
\leftarrow
\max(1,s_g^{(0)})
}
$$

即：

- 初始阶段允许放大过于乐观的正式误差；
- 初始阶段不提高某组权重；
- 正式 VCE 收敛结果允许 $s_g<1$。

方差分量初值：

$$
\theta_g^{(0)}
=
\left(s_g^{(0)}\right)^2.
$$

### 9.4 小样本和异常值

建议：

```yaml
minimum_mad_count: 10
minimum_initial_scale: 1.0
```

以下情况回退到 $s_g^{(0)}=1$：

- 组内有效观测不足；
- MAD 为零；
- MAD 非有限；
- 数据分组异常。

MAD 只用于启动，不是最终 VCE 结果。

---

## 10. IGGⅢ 抗差估计

### 10.1 标准化残差

正式抗差统计量必须无量纲：

$$
\boxed{
t_i
=
\frac{
v_i
}{
\sqrt{C_{v,ii}}
}
}
$$

不得正式使用 $v_i/\sigma_{i,\mathrm{NP}}$ 进行严格抗差判断。

### 10.2 基础残差方差

对当前尺度：

$$
\Sigma_{ii}
=
s_{g(i)}^2\sigma_{i,\mathrm{NP}}^2.
$$

基础权阵：

$$
\boldsymbol P
=
\operatorname{diag}
\left(
1/\Sigma_{ii}
\right).
$$

完整设计矩阵包括所有物理参数和 Bias：

$$
\boldsymbol M=[\boldsymbol A\ \boldsymbol B].
$$

基础法方程：

$$
\boldsymbol N_{\mathrm{base}}
=
\boldsymbol M^T
\boldsymbol P
\boldsymbol M.
$$

基础杠杆值：

$$
h_i
=
p_i
\boldsymbol m_i^T
\boldsymbol N_{\mathrm{base}}^{-1}
\boldsymbol m_i.
$$

对角协方差模型下：

$$
\boxed{
C_{v,ii}
=
\Sigma_{ii}(1-h_i)
}
$$

数值保护：

```yaml
minimum_one_minus_leverage: 1.0e-8
```

即：

$$
1-h_i
\leftarrow
\max(1-h_i,\varepsilon_h).
$$

### 10.3 为什么标准化使用基础随机模型

标准化残差分母采用当前 VCE 基础随机模型，而不是已经被 IGGⅢ 放大的等价方差。

如果异常点被降权后分母也同步增大，异常残差可能在下一轮被掩盖。

因此区分两个法方程：

1. `N_base`：由当前 $s_g^2$ 和 NP 正式精度构成，用于标准化残差；
2. `N_equivalent`：加入 IGGⅢ 因子，用于参数解算和稳健 VCE 多余度。

### 10.4 IGGⅢ 等价权函数

定义抗差因子：

$$
\boxed{
\alpha_i
=
\begin{cases}
1,
&|t_i|\le k_0,\\[1.2ex]
\dfrac{k_0}{|t_i|}
\left(
\dfrac{k_1-|t_i|}{k_1-k_0}
\right)^2,
&k_0<|t_i|\le k_1,\\[3ex]
0,
&|t_i|>k_1.
\end{cases}
}
$$

固定：

$$
\boxed{
k_0=1.5,\qquad k_1=6.0.
}
$$

等价权：

$$
\boxed{
\bar p_i
=
\alpha_i p_i
=
\frac{
\alpha_i
}{
s_{g(i)}^2\sigma_{i,\mathrm{NP}}^2
}
}
$$

### 10.5 状态分类

| 条件 | 状态 |
|---|---|
| $|t_i|\le1.5$ | `FULL_WEIGHT` |
| $1.5<|t_i|\le6.0$ | `DOWNWEIGHTED` |
| $|t_i|>6.0$ | `REJECTED` |

`REJECTED` 表示本轮权为零，但观测记录必须保留在输出和日志中。

### 10.6 边界测试

```text
|t| = 0       -> alpha = 1
|t| = 1.5     -> alpha = 1
|t| just >1.5 -> 0 < alpha < 1
|t| = 6.0     -> alpha = 0
|t| > 6.0     -> alpha = 0
alpha(-t)     = alpha(t)
0 <= alpha <= 1
```

---

## 11. 参数联合求解

固定当前 $s_g^2$ 和 $\alpha_i$：

$$
\bar{\boldsymbol P}
=
\operatorname{diag}(\bar p_i).
$$

法方程：

$$
\boldsymbol N_{\mathrm{eq}}
=
\boldsymbol M^T
\bar{\boldsymbol P}
\boldsymbol M,
$$

$$
\boldsymbol w_{\mathrm{eq}}
=
\boldsymbol M^T
\bar{\boldsymbol P}
\boldsymbol l.
$$

参数改正：

$$
\Delta\hat{\boldsymbol y}
=
\boldsymbol N_{\mathrm{eq}}^{-1}
\boldsymbol w_{\mathrm{eq}}.
$$

如果现有程序采用局部参数消元、预消元、分块法方程或稀疏求解，IGGⅢ 只改变观测权，不要求改变原有参数消元结构。

---

## 12. IGGⅢ 等价权下的分组 Helmert VCE

### 12.1 基本思想

IGGⅢ 负责把异常观测转换成当前迭代的等价权。

在该等价权系统中，仍使用 Helmert 迹公式计算每组有效多余度。

本版本不再使用：

$$
\sum_{i\in g}
\alpha_i(1-h_i)
$$

作为组多余度。

### 12.2 当前组法方程

对第 $g$ 组：

$$
\bar{\boldsymbol P}_g
=
\operatorname{diag}
\left[
\frac{
\alpha_i
}{
s_g^2\sigma_{i,\mathrm{NP}}^2
}
\right],
\qquad i\in g.
$$

组法方程贡献：

$$
\boxed{
\bar{\boldsymbol N}_g
=
\boldsymbol M_g^T
\bar{\boldsymbol P}_g
\boldsymbol M_g
}
$$

总法方程：

$$
\boxed{
\bar{\boldsymbol N}
=
\sum_{g=1}^{G}
\bar{\boldsymbol N}_g
}
$$

### 12.3 当前有效观测数

IGGⅢ 严格零权观测不进入当前法方程。

定义：

$$
n_g^+
=
\#\left\{
i\in g:\alpha_i>0
\right\}.
$$

数值保护：

```yaml
minimum_nonzero_robust_factor: 1.0e-12
```

当 $\alpha_i\le10^{-12}$ 时按零权处理。

### 12.4 组有效多余度

$$
\boxed{
r_g
=
n_g^+
-
\operatorname{tr}
\left(
\bar{\boldsymbol N}^{-1}
\bar{\boldsymbol N}_g
\right)
}
$$

含义：

$$
\text{组有效多余度}
=
\text{组非零权观测数}
-
\text{该组承担的有效参数自由度}.
$$

迹项一般不是整数，因此 $r_g$ 可以不是整数。

### 12.5 多余度总和检查

若法方程满秩，当前可估参数秩为 $u$，则：

$$
\boxed{
\sum_g r_g
=
n^+-u
}
$$

其中

$$
n^+
=
\sum_g n_g^+.
$$

程序必须验证：

$$
\left|
\sum_g r_g-(n^+-u)
\right|
<
\varepsilon_r.
$$

建议使用相对和绝对混合容差：

```yaml
redundancy_absolute_tolerance: 1.0e-8
redundancy_relative_tolerance: 1.0e-10
```

### 12.6 组残差二次型

当前等价权下：

$$
q_g
=
\boldsymbol v_g^T
\bar{\boldsymbol P}_g
\boldsymbol v_g.
$$

展开为：

$$
q_g
=
\sum_{i\in g}
\frac{
\alpha_i v_i^2
}{
s_g^2\sigma_{i,\mathrm{NP}}^2
}.
$$

如果当前组尺度合适，期望：

$$
q_g\approx r_g.
$$

### 12.7 方差分量原始更新

乘性写法：

$$
\boxed{
s_{g,\mathrm{raw}}^{2,\mathrm{new}}
=
s_g^2
\frac{
\boldsymbol v_g^T
\bar{\boldsymbol P}_g
\boldsymbol v_g
}{
r_g
}
}
$$

展开并约去旧 $s_g^2$：

$$
\boxed{
s_{g,\mathrm{raw}}^{2,\mathrm{new}}
=
\frac{
\displaystyle
\sum_{i\in g}
\alpha_i
\frac{v_i^2}{\sigma_{i,\mathrm{NP}}^2}
}{
r_g
}
}
$$

分子表示 IGGⅢ 抑制粗差后，相对于 NP 正式方差的残差能量；分母表示该组有效多余度。

### 12.8 与普通简化 Helmert VCE 的关系

当所有观测均未降权：

$$
\alpha_i=1,
$$

则

$$
n_g^+=n_g,
$$

$$
\bar{\boldsymbol P}_g=\boldsymbol P_g,
$$

$$
r_g
=
n_g-
\operatorname{tr}
\left(
\boldsymbol N^{-1}\boldsymbol N_g
\right),
$$

且

$$
s_{g,\mathrm{raw}}^2
=
\frac{
\displaystyle
\sum_{i\in g}
v_i^2/\sigma_{i,\mathrm{NP}}^2
}{
r_g
}.
$$

算法严格退化为当前分组随机模型下的简化 Helmert VCE。

### 12.9 小权观测的处理

本版本采用：

- $\alpha_i=0$：不计入 $n_g^+$；
- $0<\alpha_i\le1$：计为一条非零权观测；
- 很小但非零的权仍通过 $\bar P_g$ 对法方程和残差二次型产生很小贡献。

这属于 IGGⅢ 等价权下的 Helmert 工程实现，不把 $\alpha_i$ 解释成“分数观测数量”。

### 12.10 最低多余度

建议：

```yaml
minimum_effective_redundancy: 20.0
```

当 $r_g<r_{\min}$ 时：

- 不更新该组尺度；
- 保留上一轮 $s_g^2$；
- 标记 `INSUFFICIENT_REDUNDANCY`；
- 不将该组方差分量解释为可靠估计。

---

## 13. VCE 阻尼更新

IGGⅢ 权与 VCE 尺度相互影响：

$$
s_g
\rightarrow
t_i
\rightarrow
\alpha_i
\rightarrow
\bar P
\rightarrow
v_i
\rightarrow
s_g.
$$

为避免震荡，采用对数尺度阻尼：

$$
\boxed{
\ln s_g^{2(k+1)}
=
(1-\lambda)\ln s_g^{2(k)}
+
\lambda
\ln s_{g,\mathrm{raw}}^{2(k+1)}
}
$$

默认：

$$
\lambda=0.5.
$$

当 $\lambda=0.5$：

$$
s_g^{2(k+1)}
=
\sqrt{
s_g^{2(k)}
s_{g,\mathrm{raw}}^{2(k+1)}
}.
$$

阻尼只影响迭代路径，不改变稳定收敛点。

建议提供单轮变化保护：

```yaml
minimum_variance_ratio_per_iteration: 0.25
maximum_variance_ratio_per_iteration: 4.0
```

该限制只用于数值稳定，不是最终尺度上下限。

---

## 14. 完整联合迭代流程

### 14.1 两层迭代

#### 内层：函数模型迭代

固定当前：

- $s_g^2$；
- $\alpha_i$。

迭代轨道、月球转动、测站、反射器、Bias 等参数，直到函数模型线性化收敛。

#### 外层：随机模型迭代

在内层收敛后：

1. 计算 post-fit residual；
2. 计算基础标准化残差；
3. 更新 IGGⅢ 因子；
4. 重新联合平差；
5. 在等价权法方程下计算 Helmert 组多余度；
6. 更新 $s_g^2$；
7. 阻尼；
8. 进入下一轮。

### 14.2 推荐顺序

$$
\boxed{
\begin{aligned}
&\text{数据读取和校验}\\
&\rightarrow\text{VCE 唯一分组}\\
&\rightarrow\text{建立可重叠 Bias 矩阵}\\
&\rightarrow\text{稳健 Bias 初值}\\
&\rightarrow\text{去 Bias 后的 MAD 组尺度}\\
&\rightarrow\text{第一次基础权平差}\\
&\rightarrow\text{基础随机模型标准化残差}\\
&\rightarrow\text{IGGⅢ 更新}\\
&\rightarrow\text{等价权参数联合平差}\\
&\rightarrow\text{Helmert 迹公式计算组多余度}\\
&\rightarrow\text{稳健组尺度更新和阻尼}\\
&\rightarrow\text{循环至收敛}\\
&\rightarrow\text{固定最终随机模型再解算一次}.
\end{aligned}
}
$$

### 14.3 第一次平差

初始化：

$$
\alpha_i^{(0)}=1,
$$

$$
p_i^{(0)}
=
\frac{1}{(s_{g(i)}^{(0)})^2\sigma_{i,\mathrm{NP}}^2}.
$$

第一次平差不使用尚未由 post-fit residual 计算的严格 IGGⅢ 剔除名单。

### 14.4 外层第 $k$ 轮

1. 由当前 $s_g^{2(k)}$ 构造基础权；
2. 以内层非线性迭代求得参数和 residual；
3. 用基础随机模型计算 $t_i^{(k)}$；
4. 计算 $\alpha_i^{(k+1)}$；
5. 用新等价权重新求解参数；
6. 由等价权法方程计算 $r_g^{(k+1)}$；
7. 计算 $s_{g,\mathrm{raw}}^{2(k+1)}$；
8. 阻尼得到 $s_g^{2(k+1)}$；
9. 检查收敛。

### 14.5 收敛条件

同时满足：

$$
\max_g
\left|
\ln
\frac{s_g^{2(k+1)}}{s_g^{2(k)}}
\right|
<
\varepsilon_s,
$$

$$
\max_i
\left|
\alpha_i^{(k+1)}-\alpha_i^{(k)}
\right|
<
\varepsilon_\alpha,
$$

且函数模型参数改正满足现有代码的收敛标准。

建议：

```yaml
scale_log_tolerance: 1.0e-3
robust_weight_tolerance: 1.0e-3
maximum_stochastic_iterations: 20
```

达到最大迭代数仍未收敛时：

- 返回非成功状态；
- 保留最后一轮结果；
- 输出未收敛的组、权和参数诊断。

---

## 15. 核心伪代码

```python
def solve_llr_grouped_vce_igg3(
    observations,
    initial_state,
    config,
):
    validate_observations(observations)

    assign_unique_vce_groups(
        observations=observations,
        group_config=config.vce_groups,
        strict=True,
    )

    bias_matrix = build_bias_design_matrix(
        observations=observations,
        bias_intervals=config.bias_intervals,
        overlap_policy="additive",
    )

    prefit = compute_prefit_residuals(
        observations=observations,
        state=initial_state,
    )

    bias_initial = robust_initialize_bias(
        residuals=prefit,
        bias_matrix=bias_matrix,
        sigma_np=observations.sigma_np,
        config=config.bias_initialization,
    )

    residual_after_bias = (
        prefit - bias_matrix @ bias_initial
    )

    scales = initialize_group_scales_with_mad(
        residuals=residual_after_bias,
        sigma_np=observations.sigma_np,
        group_ids=observations.vce_group_id,
        consistency_factor=1.4826,
        minimum_initial_scale=1.0,
        minimum_count=config.minimum_mad_count,
    )

    robust_factors = np.ones(observations.count)
    state = initial_state
    bias = bias_initial

    for outer_iteration in range(
        config.maximum_stochastic_iterations
    ):
        base_variance = (
            scales[observations.vce_group_id] ** 2
            * observations.sigma_np ** 2
        )
        base_weight = 1.0 / base_variance

        equivalent_weight = (
            robust_factors * base_weight
        )

        solution = solve_nonlinear_model(
            observations=observations,
            initial_state=state,
            initial_bias=bias,
            bias_matrix=bias_matrix,
            weights=equivalent_weight,
            convergence=config.function_model_convergence,
        )

        state = solution.state
        bias = solution.bias
        residuals = solution.residuals
        design_matrix = solution.full_design_matrix

        standardized_residuals = (
            compute_base_standardized_residuals(
                residuals=residuals,
                design_matrix=design_matrix,
                base_variance=base_variance,
                base_weight=base_weight,
                minimum_one_minus_leverage=(
                    config.minimum_one_minus_leverage
                ),
            )
        )

        new_robust_factors = igg3_factors(
            standardized_residuals,
            k0=1.5,
            k1=6.0,
        )

        new_equivalent_weight = (
            new_robust_factors * base_weight
        )

        robust_solution = solve_nonlinear_model(
            observations=observations,
            initial_state=state,
            initial_bias=bias,
            bias_matrix=bias_matrix,
            weights=new_equivalent_weight,
            convergence=config.function_model_convergence,
        )

        state = robust_solution.state
        bias = robust_solution.bias
        residuals = robust_solution.residuals
        design_matrix = robust_solution.full_design_matrix

        raw_scales, vce_diagnostics = (
            estimate_group_scales_equivalent_helmert(
                residuals=residuals,
                design_matrix=design_matrix,
                sigma_np=observations.sigma_np,
                group_ids=observations.vce_group_id,
                current_scales=scales,
                robust_factors=new_robust_factors,
                minimum_redundancy=(
                    config.minimum_effective_redundancy
                ),
                alpha_zero_threshold=(
                    config.minimum_nonzero_robust_factor
                ),
            )
        )

        new_scales = damp_scale_update(
            old_scales=scales,
            raw_scales=raw_scales,
            damping=config.vce_damping,
            minimum_ratio=(
                config.minimum_variance_ratio_per_iteration
            ),
            maximum_ratio=(
                config.maximum_variance_ratio_per_iteration
            ),
        )

        write_iteration_diagnostics(
            iteration=outer_iteration,
            state=state,
            bias=bias,
            residuals=residuals,
            standardized_residuals=standardized_residuals,
            robust_factors=new_robust_factors,
            scales=new_scales,
            vce_diagnostics=vce_diagnostics,
        )

        if stochastic_model_converged(
            old_scales=scales,
            new_scales=new_scales,
            old_factors=robust_factors,
            new_factors=new_robust_factors,
            scale_tolerance=config.scale_log_tolerance,
            weight_tolerance=config.robust_weight_tolerance,
        ):
            scales = new_scales
            robust_factors = new_robust_factors
            break

        scales = new_scales
        robust_factors = new_robust_factors

    final_variance = (
        scales[observations.vce_group_id] ** 2
        * observations.sigma_np ** 2
    )
    final_weight = robust_factors / final_variance

    final_solution = solve_nonlinear_model(
        observations=observations,
        initial_state=state,
        initial_bias=bias,
        bias_matrix=bias_matrix,
        weights=final_weight,
        convergence=config.function_model_convergence,
    )

    return build_final_output(
        solution=final_solution,
        scales=scales,
        robust_factors=robust_factors,
        observations=observations,
    )
```

---

## 16. VCE 更新函数伪代码

```python
def estimate_group_scales_equivalent_helmert(
    residuals,
    design_matrix,
    sigma_np,
    group_ids,
    current_scales,
    robust_factors,
    minimum_redundancy=20.0,
    alpha_zero_threshold=1.0e-12,
):
    alpha = robust_factors.copy()
    alpha[alpha <= alpha_zero_threshold] = 0.0

    base_variance = (
        current_scales[group_ids] ** 2
        * sigma_np ** 2
    )
    equivalent_weight = alpha / base_variance
    active = equivalent_weight > 0.0

    M_active = design_matrix[active, :]
    w_active = equivalent_weight[active]

    normal_matrix = (
        M_active.T
        @ (w_active[:, None] * M_active)
    )

    factorization = factorize_normal_matrix(
        normal_matrix
    )
    parameter_rank = numerical_rank(
        normal_matrix
    )

    raw_scales = current_scales.copy()
    diagnostics = {}
    total_redundancy = 0.0

    for group_id in unique(group_ids):
        group_mask = (
            (group_ids == group_id)
            & active
        )

        n_group_active = int(group_mask.sum())

        if n_group_active == 0:
            diagnostics[group_id] = {
                "status": "NO_ACTIVE_OBSERVATIONS",
                "active_count": 0,
            }
            continue

        M_g = design_matrix[group_mask, :]
        w_g = equivalent_weight[group_mask]

        normal_group = (
            M_g.T
            @ (w_g[:, None] * M_g)
        )

        # 求解 N X = N_g，不显式求逆
        X_g = solve_factorized(
            factorization,
            normal_group,
        )
        consumed_dof = np.trace(X_g)

        redundancy = (
            n_group_active - consumed_dof
        )
        total_redundancy += redundancy

        if redundancy < minimum_redundancy:
            diagnostics[group_id] = {
                "status": "INSUFFICIENT_REDUNDANCY",
                "active_count": n_group_active,
                "consumed_dof": consumed_dof,
                "redundancy": redundancy,
            }
            continue

        v_g = residuals[group_mask]
        sigma_g = sigma_np[group_mask]
        alpha_g = alpha[group_mask]

        raw_variance_scale = np.sum(
            alpha_g * (v_g / sigma_g) ** 2
        ) / redundancy

        if (
            raw_variance_scale <= 0.0
            or not np.isfinite(raw_variance_scale)
        ):
            diagnostics[group_id] = {
                "status": "INVALID_SCALE_UPDATE",
                "active_count": n_group_active,
                "consumed_dof": consumed_dof,
                "redundancy": redundancy,
            }
            continue

        raw_scales[group_id] = np.sqrt(
            raw_variance_scale
        )

        diagnostics[group_id] = {
            "status": "UPDATED",
            "active_count": n_group_active,
            "consumed_dof": consumed_dof,
            "redundancy": redundancy,
            "raw_variance_scale": raw_variance_scale,
            "raw_scale": raw_scales[group_id],
        }

    total_active = int(active.sum())
    expected_total_redundancy = (
        total_active - parameter_rank
    )

    assert_close_mixed(
        total_redundancy,
        expected_total_redundancy,
        absolute_tolerance=1.0e-8,
        relative_tolerance=1.0e-10,
    )

    return raw_scales, diagnostics
```

---

## 17. 推荐配置文件

```yaml
stochastic_model:
  type: grouped_np_scale
  estimate_additive_floor: false
  estimate_night_common_error: false

robust_estimation:
  method: IGG3
  k0: 1.5
  k1: 6.0
  residual_type: standardized_postfit
  leverage_correction: true
  minimum_one_minus_leverage: 1.0e-8
  minimum_nonzero_robust_factor: 1.0e-12
  zero_weight_is_rejected: true

initialization:
  bias_method: robust_regression
  scale_method: normalized_mad
  mad_consistency_factor: 1.4826
  minimum_initial_scale: 1.0
  minimum_mad_count: 10

vce:
  method: equivalent_weight_grouped_helmert
  damping: 0.5
  minimum_effective_redundancy: 20.0
  scale_log_tolerance: 1.0e-3
  minimum_variance_ratio_per_iteration: 0.25
  maximum_variance_ratio_per_iteration: 4.0
  redundancy_absolute_tolerance: 1.0e-8
  redundancy_relative_tolerance: 1.0e-10
  maximum_iterations: 20
  allow_final_scale_below_one: true

group_assignment:
  strict: true
  unassigned_observation_policy: error
  multiple_match_policy: error

bias:
  estimate_in_main_solution: true
  overlap_policy: additive
  endpoint_policy: inclusive_dates
  automatic_bias_detection: false
```

---

## 18. 建议的代码模块

```text
llr/
  stochastic/
    group_config.py
    group_assignment.py
    bias_intervals.py
    bias_initialization.py
    mad_scale.py
    igg3.py
    standardized_residuals.py
    grouped_helmert_vce.py
    convergence.py
    diagnostics.py

  solver/
    weighted_normal_equations.py
    nonlinear_iteration.py
    final_solution.py

  config/
    llr_vce_groups.yaml
    llr_bias_intervals.yaml
    llr_robust_vce.yaml

  tests/
    test_group_assignment.py
    test_bias_overlap.py
    test_bias_initialization.py
    test_mad_scale.py
    test_igg3.py
    test_standardized_residuals.py
    test_grouped_helmert_vce.py
    test_redundancy_sum.py
    test_joint_iteration.py
```

---

## 19. 输出与诊断

### 19.1 每个 VCE 组

| 字段 | 含义 |
|---|---|
| `group_id` | VCE 组 ID |
| `configured_start` | 配置起始时间 |
| `configured_end` | 配置结束时间或 `present` |
| `actual_start_epoch` | 实际数据最早时间 |
| `actual_end_epoch` | 实际数据最晚时间 |
| `observation_count` | 输入观测数 |
| `active_count` | 当前非零权观测数 |
| `full_weight_count` | $\alpha=1$ 数量 |
| `downweighted_count` | $0<\alpha<1$ 数量 |
| `rejected_count` | $\alpha=0$ 数量 |
| `consumed_dof` | $\operatorname{tr}(\bar N^{-1}\bar N_g)$ |
| `effective_redundancy` | $r_g$ |
| `initial_scale` | MAD 启动 $s_g^{(0)}$ |
| `final_scale` | 最终 $s_g$ |
| `variance_component` | 最终 $s_g^2$ |
| `residual_rms` | 残差 RMS |
| `residual_wrms` | 最终等价权 WRMS |
| `standardized_rms` | 标准化残差 RMS |
| `median_standardized_residual` | 标准化残差中位数 |
| `mad_standardized_residual` | 标准化残差 MAD |
| `update_status` | VCE 更新状态 |

### 19.2 每个观测

```text
observation_id
epoch
station_id
station_system
vce_group_id
sigma_np
base_scale
base_variance
base_weight
postfit_residual
residual_sigma
standardized_residual
igg3_factor
equivalent_weight
robust_status
matched_bias_ids
```

### 19.3 每个 Bias

```text
bias_id
station
configured_from
configured_to
initial_value
estimated_value
formal_sigma
observation_count
correlation_with_station_radial
maximum_parameter_correlation
```

### 19.4 每轮迭代

```text
iteration
parameter_correction_norm
maximum_scale_log_change
maximum_robust_factor_change
active_observation_count
rejected_observation_count
total_effective_redundancy
expected_total_redundancy
normal_matrix_condition
```

### 19.5 必须生成的图

1. 各 VCE 组残差时间序列；
2. 各组标准化残差时间序列；
3. $|t_i|$ 与 $\alpha_i$ 的关系；
4. 各组 $s_g$ 迭代曲线；
5. 残差绝对值与 $\sigma_{\mathrm{NP}}$ 的关系；
6. 标准化残差直方图；
7. Bias 估计前后残差对比；
8. 拒绝观测在时间、反射器、月相和高度角上的分布；
9. Bias 与测站径向坐标相关系数图；
10. 每组 consumed DOF 与有效多余度。

---

## 20. 质量诊断

### 20.1 单尺度模型充分性

检查最终

$$
t_i
=
\frac{v_i}{\sqrt{C_{v,ii}}}
$$

的散布是否仍随 $\sigma_{i,\mathrm{NP}}$ 系统变化。

如果小 $\sigma_{i,\mathrm{NP}}$ 观测持续集中在降权或拒绝区域，而其绝对残差并不明显更大，应标记：

```text
POSSIBLE_WITHIN_GROUP_FORMAL_ERROR_DISTORTION
```

这说明组内正式误差相对关系可能不可信。当前版本只报告，不自动加入噪声底。

### 20.2 Bias 未充分建模

如果某组：

- $s_g$ 很大；
- 残差中位数显著非零；
- 残差存在阶跃或长期漂移；

应优先检查 Bias 和函数模型，而不是仅解释为随机噪声增大。

### 20.3 抗差掩盖

标准化残差必须由基础随机模型计算，不能使用已被 IGGⅢ 放大的等价方差，以避免异常点被自身降权掩盖。

### 20.4 VCE 多余度检查

必须满足：

$$
\sum_g r_g=n^+-u.
$$

如果不满足，重点检查：

- 组法方程是否完整求和；
- 零权观测计数是否一致；
- 参数秩是否使用数值秩而不是矩阵列数；
- 局部参数消元后组法方程贡献是否正确回代；
- 约束和先验是否纳入总法方程但未分配到组。

### 20.5 约束和先验的处理

如果法方程含有参数先验或约束：

$$
\boldsymbol N
=
\sum_g\boldsymbol N_g
+
\boldsymbol N_{\mathrm{prior}},
$$

则简单的

$$
\sum_g r_g=n^+-u
$$

可能不再直接成立。

实现时必须二选一：

1. 在多余度测试中显式纳入先验分量；
2. 只对数据法方程使用广义多余度关系，并单独记录约束贡献。

代码不得在存在先验时无条件断言普通恒等式。

### 20.6 最终参数协方差

固定最终权后，可按现有加权最小二乘方式计算形式协方差。

但应注明：

- IGGⅢ 权是数据依赖权；
- 普通形式协方差没有完整传播抗差权估计的不确定性；
- VCE 方差分量本身的不确定性也未完全进入参数协方差。

---

## 21. 测试要求

### 21.1 IGGⅢ 单元测试

- 阈值边界；
- 正负对称；
- 单调降权；
- 权因子范围；
- NaN 和 Inf 输入处理；
- $k_1>k_0>0$ 配置校验。

### 21.2 VCE 分组测试

- CERGA MeO/IR 重叠年份正确区分；
- MLRS1/MLRS2 的 1988 年正确区分；
- Apache、Matera、CERGA MeO/IR、Wettzell 的 `present` 无上界；
- 一条观测零命中时报错；
- 一条观测多命中时报错；
- 1969 McDonald 数据正确命中 `MCDONALD_1969_1985`。

### 21.3 Bias 测试

- 无重叠区间；
- 长短区间嵌套；
- 一条观测命中多个 Bias；
- Wettzell 2018 至今正确命中；
- Bias 稳健初值与零初值最终收敛到近似结果；
- Bias 与测站坐标秩亏检测。

### 21.4 MAD 测试

- 正态数据尺度恢复；
- 少量大粗差下保持稳定；
- MAD 为零时回退；
- 小样本回退；
- 非均匀 $\sigma_{\mathrm{NP}}$ 下必须先标准化。

### 21.5 Helmert VCE 测试

构造无抗差数据：

$$
\alpha_i=1.
$$

验证算法退化为：

$$
r_g
=
n_g-
\operatorname{tr}(N^{-1}N_g).
$$

并验证：

$$
\sum_g r_g=n-u.
$$

### 21.6 等价权测试

构造部分 $\alpha_i=0$、部分 $0<\alpha_i<1$：

- 零权观测不计入 $n_g^+$；
- 非零权观测计入 $n_g^+$；
- 组法方程使用等价权；
- 总多余度满足无先验条件下的恒等式。

### 21.7 合成联合测试

构造至少两个组：

$$
s_1=2,\qquad s_2=5.
$$

加入：

- 已知分段 Bias；
- 重叠 Bias；
- 2%–5% 大粗差；
- 不同 $\sigma_{\mathrm{NP}}$；
- 一个全时段 Bias。

验证：

- Bias 恢复；
- $s_g$ 恢复；
- 粗差被降权或拒绝；
- 正常小正式误差观测不会在尺度标定前批量误删；
- 参数结果无明显偏差。

### 21.8 回归测试模式

保留：

```text
ordinary_wls
vce_only
igg3_only
vce_plus_igg3
```

比较：

- 参数；
- Bias；
- WRMS；
- 有效观测数；
- 拒绝数；
- 组尺度；
- 法方程条件数；
- 运行时间。

---

## 22. 性能与数值实现

### 22.1 禁止显式求逆

计算

$$
\operatorname{tr}
\left(
\bar N^{-1}\bar N_g
\right)
$$

时，不应显式构造 $\bar N^{-1}$。

推荐：

1. 对 $\bar N$ 做 Cholesky、LDLT 或适用的稀疏分解；
2. 求解
   $$
   \bar N X_g=\bar N_g;
   $$
3. 计算
   $$
   \operatorname{tr}(X_g).
   $$

### 22.2 大规模参数问题

如果全局参数很多，逐组求解完整矩阵右端可能昂贵。可考虑：

- 已有 Schur complement 结构；
- 稀疏多右端求解；
- Hutchinson 随机迹估计作为可选加速；
- 对局部参数先消元后在约化法方程中计算组贡献。

第一版优先实现精确迹计算，并建立性能基线。

### 22.3 数值秩

总多余度中的 $u$ 必须使用数值秩：

```python
u = numerical_rank(normal_matrix)
```

不能无条件使用参数列数。

### 22.4 观测权上下限

不建议人为给最终 $s_g$ 设置固定上下限。

允许设置单轮更新比例限制和数值权上限，但必须在日志中标明，不得悄悄改变最终模型。

---

## 23. 当前实现边界

当前版本明确不包括：

1. 自动检测新 Bias；
2. 自动拆分新的设备时期；
3. 加性噪声底；
4. 观测夜公共误差；
5. 非对角时间相关模型；
6. 方差分量先验和平滑；
7. 根据残差自动改变 CERGA MeO/IR 分类；
8. 把论文表中的 WRMS 直接作为先验权；
9. 完整稳健 M 估计协方差理论；
10. VCE 方差分量不确定度向最终物理参数协方差的完整传播。

---

## 24. 接入现有代码库的实施清单

### 阶段 1：配置和数据映射

- [ ] 添加 11 个 VCE 组；
- [ ] 支持 `end_exclusive: null`；
- [ ] 添加完整 29 个 Bias；
- [ ] 建立 station/system alias 映射；
- [ ] 校验 CERGA MeO/IR；
- [ ] 校验 MLRS1/MLRS2；
- [ ] 输出未分组报告。

### 阶段 2：初始化

- [ ] 实现 Bias 指示矩阵；
- [ ] 实现稳健 Bias 初值；
- [ ] 实现标准化 MAD；
- [ ] 添加小样本回退；
- [ ] 添加初始化诊断。

### 阶段 3：抗差

- [ ] 实现基础杠杆值；
- [ ] 实现标准化 residual；
- [ ] 实现 IGGⅢ；
- [ ] 实现抗差状态分类；
- [ ] 添加边界单元测试。

### 阶段 4：VCE

- [ ] 实现组法方程贡献；
- [ ] 实现 Helmert 迹多余度；
- [ ] 实现总多余度检查；
- [ ] 实现组尺度更新；
- [ ] 实现对数阻尼；
- [ ] 处理低多余度组。

### 阶段 5：联合迭代

- [ ] 连接现有非线性求解器；
- [ ] 实现外层随机模型迭代；
- [ ] 实现收敛判断；
- [ ] 实现最终固定权解算；
- [ ] 保存迭代历史。

### 阶段 6：诊断和回归

- [ ] 输出观测级结果；
- [ ] 输出组级结果；
- [ ] 输出 Bias 相关性；
- [ ] 生成诊断图；
- [ ] 完成四种处理模式回归对比。

---

## 25. 算法摘要

最终算法为：

$$
\boxed{
\begin{aligned}
&\text{读取并校验 LLR NP}\\
&\rightarrow\text{按测站、系统和时间分配唯一 VCE 组}\\
&\rightarrow\text{按 29 个区间建立可重叠 Bias 矩阵}\\
&\rightarrow\text{稳健估计 Bias 初值}\\
&\rightarrow\text{对去 Bias 的 }(O-C)/\sigma_{\mathrm{NP}}\text{ 做 MAD}\\
&\rightarrow\text{得到 }s_g^{(0)}\\
&\rightarrow\text{参数和 Bias 联合平差}\\
&\rightarrow\text{用基础随机模型计算标准化 residual}\\
&\rightarrow\text{IGGⅢ，}k_0=1.5,\ k_1=6.0\\
&\rightarrow\text{使用等价权重新平差}\\
&\rightarrow\text{Helmert 迹公式计算组有效多余度}\\
&\rightarrow\text{更新并阻尼 }s_g^2\\
&\rightarrow\text{循环至随机模型和参数收敛}\\
&\rightarrow\text{固定最终权重新解算}\\
&\rightarrow\text{输出参数、Bias、尺度、权和诊断}.
\end{aligned}
}
$$

三个模块的职责必须保持清晰：

$$
\boxed{
\text{Bias 负责残差中心，VCE 负责组尺度，IGGⅢ 负责异常观测。}
}
$$

不得用增大方差代替 Bias，也不得在组尺度标定前使用原始 NP 正式误差执行严格抗差剔除。

---

## 26. 参考依据

1. 秦显平、杨元喜，《随机模型对解算西安流动 SLR 站坐标的影响》，武汉大学学报·信息科学版，2019。本文采用其中的 IGGⅢ 等价权函数形式以及 $k_0=1.5,\ k_1=6.0$ 的参数设置。
2. EPM2023a LLR 处理结果表。本文采用其中的测站、设备或系统时期作为 VCE 分组基础，并为覆盖当前输入将 Apache、CERGA MeO/IR、Matera 和 Wettzell 分组扩展至 `present`。
3. LLR Bias 区间表。本文保留原 28 个区间，并新增 Wettzell 2018-01-01 至今的全时段 Bias。
