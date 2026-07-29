# IERS 2010 Fortran 源码导览

本目录保存 LLROPS 使用的 IERS Conventions (2010) Fortran 例程、f2py
接口定义及完整许可证。所有例程被编译进一个私有扩展模块
`llrops._iers2010`，而不是分别生成多个扩展。

本文只说明 vendored 源码的组成、调用关系和维护边界。项目级构建方式、
上游版本固定、生产模型约定和验证要求见
[`../../docs/DEVELOPMENT.md`](../../docs/DEVELOPMENT.md)。

目录内容：

- `src/`：参与原生扩展编译的 Fortran 源文件。
- `bindings/iers2010.pyf`：唯一的 Python 原生接口白名单及数组/参数声明。
- `LICENSE`：随源码及二进制分发的完整 IERS Conventions Software License。
- `../../meson.build`：f2py 代码生成、Fortran 编译、扩展链接和安装规则。

这里的“主要接口”和 IERS 源文件头部的 Class 1、Class 2、Class 3 或
Canonical model 分类不是一回事：

- **主要接口**：出现在 `bindings/iers2010.pyf` 中，可以从
  `llrops._iers2010` 调用。
- **辅助例程**：参与同一扩展的编译和链接，但没有在 `.pyf` 中暴露，
  只能由其他 Fortran 例程内部调用。

当前共有 8 个主要接口文件和 22 个仅内部使用的辅助文件。
`FUNDARG` 是主要接口，同时也被其他地球自转例程内部复用。

## 主要接口

### 光学大气延迟

| Fortran 文件/符号 | Python 名称 | 功能 | 主要输入和输出 |
|---|---|---|---|
| `FCUL_A.F` / `FCUL_A` | `fcul_a` | 计算 Mendes FCULa 总大气映射函数。 | 输入纬度、平均海平面以上高度、温度和高度角；返回无量纲映射因子。 |
| `FCUL_ZD_HPA.F` / `FCULZD_HPA` | `fculzd_hpa` | 计算光学波段天顶总延迟，并分解为静力和非静力部分。 | 输入纬度、椭球高、总气压、水汽压和波长；返回三个以米为单位的延迟。 |

这两个例程没有依赖本目录中的其他 Fortran 辅助文件。相对湿度转换、
最低高度角策略和输入校验由 Python 层负责。

### 地球自转和高频 EOP

| Fortran 文件/符号 | Python 名称 | 功能 | 主要输入和输出 |
|---|---|---|---|
| `ORTHO_EOP.F` / `ORTHO_EOP` | `ortho_eop` | 计算海潮引起的日周期和半日周期极移及 UT1 变化。 | 输入用于潮汐相位的 UT1 MJD；返回 `dx`、`dy`（微角秒）和 `dUT1`（微秒）。 |
| `PMSDNUT2.F` / `PMSDNUT2` | `pmsdnut2` | 计算非刚性地球的亚日尺度极移/章动模型。 | 输入 TT 近似的 MJD；返回 `dx`、`dy`（微角秒）。 |
| `UTLIBR.F` / `UTLIBR` | `utlibr` | 计算地球轴向自转的亚日尺度天平动。 | 输入 TT 近似的 MJD；返回 `dUT1`（微秒）和 `dLOD`（微秒/日）。 |
| `FUNDARG.F` / `FUNDARG` | `fundarg` | 计算 IERS 推荐的日月基本角。 | 输入 J2000.0 起算的儒略世纪；返回五个弧度角。 |

### 固体地球潮汐

| Fortran 文件/符号 | Python 名称 | 功能 | 主要输入和输出 |
|---|---|---|---|
| `DEHANTTIDEINEL.F` / `DEHANTTIDEINEL` | `dehanttideinel` | 计算太阳和月球引力造成的 IERS 2010 固体地球潮汐测站位移。 | 输入测站、太阳和月球的地心 ITRF/ECEF 米制向量及 UTC 日期；返回 ITRF 米制位移向量。 |

### 海潮负荷

| Fortran 文件/符号 | Python 名称 | 功能 | 主要输入和输出 |
|---|---|---|---|
| `HARDISP_WRAP.F` / `HARDISP_WRAP` | `hardisp` | 将上游独立程序 `HARDISP.F` 改造成无文件 I/O 的可调用数组接口，展开 BLQ 潮汐系数并生成规则时间序列。 | 输入 UTC 起始时刻、样本数、步长及 `(3, 11)` BLQ 振幅/相位；返回 Up、South、West 米制位移序列。 |

`HARDISP_WRAP.F` 是 LLROPS 派生适配，不是 IERS Conventions Center 发布的
原始例程。生产观测历元通常不规则，因此当前 Python 生产路径使用单点
`N=1`；规则序列接口仍保留给离线产品。

## 主要调用关系

```text
FCUL_A                         （无本地依赖）
FCULZD_HPA                     （无本地依赖）

ORTHO_EOP
└── CNMTX

PMSDNUT2
└── FUNDARG

UTLIBR
└── FUNDARG

DEHANTTIDEINEL
├── SPROD, ZERO_VEC8
├── ST1IDIU, ST1ISEM, ST1L1
│   └── NORM8
├── STEP2DIU, STEP2LON
├── CAL2JD
└── DAT
    └── CAL2JD

HARDISP_WRAP
├── MDAY
├── ADMINT
│   ├── TDFRPH
│   │   ├── JULDAT, LEAP, TOYMD
│   │   └── ETUTC
│   ├── SHELLS
│   └── SPLINE, EVAL
└── RECURS
```

## 辅助例程

### 地球自转辅助

| 文件/符号 | 被谁使用 | 功能 |
|---|---|---|
| `CNMTX.F` / `CNMTX` | `ORTHO_EOP` | 根据 Cartwright-Tayler-Edden 主谱线计算二阶日周期和半日周期潮汐势的时间相关偏导。 |

`FUNDARG` 不是辅助专用符号，但 `PMSDNUT2` 和 `UTLIBR` 都使用它生成日月
基本角，因此修改它会同时影响三个公开入口。

### 固体地球潮汐辅助

| 文件/符号 | 功能 |
|---|---|
| `SPROD.F` / `SPROD` | 计算两个三维向量的点积及各自模长。 |
| `NORM8.F` / `NORM8` | 返回三维向量的欧几里得模长。 |
| `ZERO_VEC8.F` / `ZERO_VEC8` | 将三维双精度向量清零。 |
| `ST1IDIU.F` / `ST1IDIU` | 计算地幔非弹性导致的日周频带异相 Love 数改正。 |
| `ST1ISEM.F` / `ST1ISEM` | 计算半日周频带异相 Love 数改正。 |
| `ST1L1.F` / `ST1L1` | 计算 Love 数纬度依赖项的位移改正。 |
| `STEP2DIU.F` / `STEP2DIU` | 计算日周频带的同相和异相位移改正。 |
| `STEP2LON.F` / `STEP2LON` | 计算长周期频带的同相和异相位移改正。 |
| `CAL2JD.F` / `CAL2JD` | 将公历日期转换为两部分儒略日/MJD。它是为匹配 IERS 调用名而重命名的 SOFA 派生文件。 |
| `DAT.F` / `DAT` | 根据内置闰秒表计算 `TAI-UTC`。它是为匹配 IERS 调用名而重命名的 SOFA 派生文件，并调用 `CAL2JD`。 |

`DAT.F` 的闰秒表固定在所选上游版本中；不能把它当作自动更新的现代闰秒
数据源。

### 海潮负荷辅助

| 文件/符号 | 功能 |
|---|---|
| `ADMINT.F` / `ADMINT` | 从输入的主要 BLQ 分潮构造导纳，通过实部/虚部样条插值扩展出完整潮汐分量的振幅、频率和相位。 |
| `TDFRPH.F` / `TDFRPH` | 根据 Doodson 数和 `/DATE/` 公共块中的 UTC 时刻计算分潮频率与相位。 |
| `RECURS.F` / `RECURS` | 用正弦/余弦递推高效生成多分潮规则位移序列。 |
| `SPLINE.F` / `SPLINE` | 为非等间隔三次样条插值计算二阶导数数组。 |
| `EVAL.F` / `EVAL` | 使用 `SPLINE` 生成的数组求取样条插值值。 |
| `SHELLS.F` / `SHELLS` | 使用 Shell sort 对频率数组排序并保留索引映射。 |
| `ETUTC.F` / `ETUTC` | 计算 Ephemeris Time 与 UTC 的差，用于潮汐相位；其最后一个闰秒表项在 UTC-TAI 保持 -37 秒期间继续有效。IERS Bulletin C 72 已确认 2026 年 12 月末不增加闰秒，应用层据此接受 2027 年 7 月前的历元。 |
| `JULDAT.F` / `JULDAT` | 将公历年月日转换为整数儒略日。 |
| `LEAP.F` / `LEAP` | 判断公历年份是否为闰年。 |
| `MDAY.F` / `MDAY` | 返回指定月份开始前已经过去的年内日数，供 `HARDISP_WRAP` 构造年积日。 |
| `TOYMD.F` / `TOYMD` | 将“年份 + 年积日”转换为“年、月、日”。 |

## 接口和源码维护规则

1. 不要因为某个辅助符号已经被链接进扩展，就假定它属于 Python API；
   `.pyf` 是公开原生入口的唯一依据。
2. 新增主要接口时，需要同时更新 `bindings/iers2010.pyf`、根目录
   `meson.build`、原生测试和本文档。
3. 新增辅助例程时，需要把源文件加入 `meson.build`，并在本文档相应功能组
   中记录调用者和职责。
4. 不要直接修改固定的上游 IERS 源文件。确需适配时，必须遵守许可证，
   使用新的文件/例程名并写明派生关系。
5. Python 外层负责时间尺度、单位、坐标系、形状和有限值校验；Fortran
   入口的原始单位和约定不能仅凭变量名推断。

上游版本固定、构建方式、生产调用约定和验证要求见
[`../../docs/DEVELOPMENT.md`](../../docs/DEVELOPMENT.md)。
