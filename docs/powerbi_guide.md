# Power BI 看板制作说明

在 Windows 电脑上用 Power BI Desktop 制作。数据来自 `output/powerbi/` 里的 7 个 CSV 文件，先把这个文件夹整个拷到 Windows 电脑上。

预计用时 3 小时左右。

## 1. 导入数据

"获取数据" → "文本/CSV"，把 7 个文件逐个导入。导入时检查两点：

- `year`、`start_year`、`end_year` 的类型是"整数"，不要被识别成日期。
- `iso3` 是"文本"。

| 表 | 内容 | 行数 |
|---|---|---|
| `dim_country` | 国家：代码、名称、地区、收入组 | 163 |
| `dim_year` | 年份 | 44 |
| `dim_variable` | 变量：英文名称、类别、对应的 IMF 变量、重要性排名 | 20 |
| `fact_risk` | 每个国家每年的风险分数和是否临近危机 | 6,058 |
| `fact_contribution` | 每个国家每年每个变量的取值和贡献 | 121,160 |
| `fact_crisis` | 危机的起止年份 | 164 |
| `fact_metrics` | 各模型在三种检验方式下的 AUC | 30 |

## 2. 建立表关系

切到"模型视图"，按下表把字段拖在一起。全部是"一对多"，方向从维度表到事实表。

| 维度表（一） | 事实表（多） | 字段 |
|---|---|---|
| `dim_country` | `fact_risk` | `iso3` |
| `dim_country` | `fact_contribution` | `iso3` |
| `dim_country` | `fact_crisis` | `iso3` |
| `dim_year` | `fact_risk` | `year` |
| `dim_year` | `fact_contribution` | `year` |
| `dim_variable` | `fact_contribution` | `variable` |

`fact_metrics` 不用连任何表。

这就是数据库里说的"星型结构"：中间是事实表，周围是维度表。筛选器加在维度表上，会自动传到事实表。

## 3. 新建三个度量值

在 `fact_risk` 表上右键 → "新建度量值"，逐个输入：

```
Fitted score = AVERAGE(fact_risk[score_fitted])
```

```
Out-of-sample score = AVERAGE(fact_risk[score_block_cv])
```

```
Total contribution = SUM(fact_contribution[contribution])
```

## 4. 三个页面

### 第一页：风险总览

目的：某一年，哪些国家的风险分数最高。

- **切片器**：`dim_year[year]`，样式选"下拉"，设为单选。
- **地图**（"着色地图"）：位置用 `dim_country[country]`，颜色饱和度用 `Fitted score`。颜色用单一色系，从浅到深。
- **条形图**：轴用 `dim_country[country]`，值用 `Fitted score`，按值降序，"筛选器"里设"前 N 个"= 15。
- **表格**：`country`、`Fitted score`、`Out-of-sample score`、`crisis_start_ahead`。这样能直接看到两种分数的差别。
- **文本框**（放在页面顶部）：写明"拟合分数来自用全部样本训练的模型，是描述性的；样本外分数来自按时间分块的交叉验证"。

### 第二页：单个国家的风险时间线

目的：一个国家的风险分数怎么随时间变化，危机之前有没有升高。

- **切片器**：`dim_country[country]`，单选。
- **折线图**：X 轴用 `dim_year[year]`，Y 轴放 `Fitted score` 和 `Out-of-sample score` 两条线。
- **危机标记**：在折线图的"分析"窗格里加"X 轴恒定线"，位置填该国的危机起始年份。或者加一张卡片显示 `fact_crisis[start_year]`。
- **卡片**：显示所选国家的危机次数，用 `COUNTROWS(fact_crisis)`。

建议默认选爱尔兰：拟合分数在 2006–2007 年升到很高，而样本外分数一直很低。这一页最能说明"样本内看起来有效、样本外无效"。

### 第三页：风险分解

目的：某个国家某一年的分数是由哪些变量推高或拉低的。

- **两个切片器**：`dim_country[country]` 和 `dim_year[year]`，都设单选。
- **条形图**：轴用 `dim_variable[label]`，值用 `Total contribution`，按值排序。在"格式" → "数据颜色"里打开"条件格式"，规则设为：大于 0 用红色，小于 0 用蓝色。
- **表格**：`label`、`value`（变量的实际取值）、`Total contribution`、`imf_variable`。
- **卡片**：`Fitted score`。

各变量的贡献加起来，再加上样本平均分 0.50，正好等于拟合分数。可以在页面上写一句说明。

### 可选的第四页：三种检验方式的对比

- **簇状柱形图**：轴用 `fact_metrics[evaluation]`，图例用 `fact_metrics[model]`，值用 `auc`，筛选 `version = v2`。
- 加一条 Y = 0.5 的恒定线，标注"随机猜测"。

这一页对应 README 里的主图，是整个项目的核心结论。

## 5. 做完之后

- 每页截一张图，放到 `output/dashboard/` 文件夹，README 里会引用。
- `.pbix` 文件也放进项目文件夹，一起上传 GitHub。

## 面试时可能被问到的两点

- **为什么用星型结构**：维度表里每个国家、每个变量只出现一次，筛选和改名只需要改一处；事实表只存数字和键，体积小。
- **拟合分数和样本外分数为什么差这么多**：拟合分数的模型见过那一年的数据，包括同一年其他国家的危机；样本外分数的模型没有见过前后几年的任何数据。
