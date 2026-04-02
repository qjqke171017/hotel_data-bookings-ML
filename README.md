# hotel_data-bookings-ML
酒店预订数据集研究
Please run the code with the following order: HotelData -> datacleaning -> function_show -> machinelearning.

HotelData类封装的函数如下：
1.数据集上传。接收一个.csv类型文件。如果输入不符合规定或没能成功上传，则报错。
2.简易的数据清洗。打印每个属性的缺失值数量，填充NA。事实上这样的数据清洗是不充分的，我们还需要进一步进行清洗。
3.添加新的酒店预订记录。输入为列表或字典，且键必须为已上传数据集中的属性。若输入不符合规定则报错。成功输入的话，将把新的预订记录加入数据集，并打印提示何种属性没有被赋值，系统将自动填充NA
4.简易的字段信息研究。输入为数据集中的一个属性。若输入不符合规定将报错。如果输入是字符型的属性，则返回数值特征，包括最大值，最小值，平均数，中位数，众数，分位数，偏度峰度等。若输入为分类型变量，则打印不同取值的计数。
5.简易的数据可视化。输入为数据集中的一个属性。若输入不符合规定则报错。如果输入是字符型的属性，则绘制直方图。若输入是分类型的属性（因子型），则输出条形图
6.数据集划分
7.随机森林方法以研究预定取消的影响因素
8.神经网络方法以研究预订取消的影响因素
9.两种方法的对比，打印两种机器学习方法的预测准确率，并输出准确率更高的方法。
10.总结，绘制条形图，ROC曲线等
11.对于新的预订记录的退订概率预测。使用训练好的两种模型对新记录的预测。输入为一条预订记录，可以有缺失的属性。返回为两条打印信息：随机森林和神经网络预测的退订概率。

datacleaning&visulization.py 数据清洗示例和可视化
function_show.py 展示类中封装函数的使用
hotel_Machine_Learning.py 对预订是否取消的机器学习部分

所需要的包及其版本
pandas 1.1.5
numpy 1.21.5
matplotlib 3.5.3
seaborn 0.9.0
tensorflow 2.8.0
plotly 5.18.0




# 液冷代码解读

![python](https://img.shields.io/badge/python-3.8%2B-blue  )
![license](https://img.shields.io/badge/license-MIT-green  )
![status](https://img.shields.io/badge/status-research-orange  )

目标：对 `code/MLRL_Model` 内所有脚本进行模块化说明，给出清晰的输入/输出与关键技术点，便于复现与二次开发。

---

## 一、总体流程概览

本目录包含一个完整的 CDU（Cooling Distribution Unit）异常检测链路：

1. 数据校验与清洗（`0_Model_Check.py`）
2. 工况切分与预处理（`1_Model_Preprocessing.py`）
3. 特征提取（`2_Model_Feature.py`）
4. 可视化分析（`3_Model_Visualization.py`）
5. 模型训练（`4_Model_Train.py`）
6. 模型测试与报告（`5_Model_Test.py`）

目录结构：

```
code/MLRL_Model/
├─ 0_Model_Check.py               # 原始数据校验 + 异常切片
├─ 1_Model_Preprocessing.py        # 稳态/非稳态工况切分
├─ 2_Model_Feature.py              # 特征提取（9特征+1目标）
├─ 3_Model_Visualization.py        # 可视化分析（相关性/残差/机理模型）
├─ 4_Model_Train.py                # 训练机理+ML+MLRL 模型
├─ 5_Model_Test.py                 # 测试预测 + 交互式异常报告
└─ README.md                       # 本说明
```

---

## 二、模块化详细解读

以下按脚本逐一说明，每个模块均提供：功能定位、输入输出、关键技术点与公式说明。

---

### 模块 0：数据校验与异常切片

文件：`0_Model_Check.py`

功能定位：
- 读取原始多传感器 CSV 数据（压力、阀门、转速）
- 统一时间轴（1 min）并插值对齐
- 依据物理规则标注异常
- 按时间连续性切分，生成设备级合并数据与指标级单文件

输入格式：
- 目录层级：`Root/Site/标准化数据/System/设备/传感器CSV`
- 单个 CSV 必须包含：时间列（`time/date/t` 任一）与 `value` 数值列

输入示例：
```
D:/【原始数据】华为云/全网设备导出数据集/标准化数据/系统A/回路3CDU/设备001/一次侧过滤器入口压力.csv
```

输出内容：
- 设备级合并宽表（每类异常一个 CSV）
- 指标级单变量 CSV（每个传感器一个 CSV）
- 切分明细表：`测试结果/CDU_分段校验明细.csv`

输出示例：
```
治理数据设备级/01_数据正常/SystemA/设备001/设备001.csv
治理数据指标级/01_数据正常/SystemA/设备001/一次侧过滤器入口压力.csv
```

关键技术点：
- 数据量纲修正：阀门与转速异常量纲自动 ×10 或 ÷10。
- 物理规则异常检测：多条规则组合为错误码（压差、逆压、停机等）。
- 片段合并策略：同类状态片段合并保存，过滤短片段。
- 分类映射：异常编号映射到不同文件夹（如 `05_一次侧过滤器异常`）。

---

### 模块 1：工况切分与稳态提取

文件：`1_Model_Preprocessing.py`

功能定位：
- 对 `0_Model_Check.py` 输出的指标级数据做工况识别
- 识别稳态/非稳态/多工况
- 基于转速/阀门行为完成 Pump / Valve / Hybrid 模式分类

输入格式：
- 输入目录通常为：`治理数据指标级`
- 每个设备文件夹下包含多指标 CSV（中文名）

输出内容：
- 输出到 `治理数据_分段处理`，结构如下：

```
治理数据_分段处理/
└─ Pump|Valve|Hybrid/
   └─ 原始分类（如 01_数据正常）/
      └─ System/
         └─ 设备ID_片段名/
            └─ 各指标CSV
```

输出示例：
```
治理数据_分段处理/Pump/01_数据正常/SystemA/设备001_Single/二次侧泵转速.csv
```

关键技术点：
- 稳态检测：基于 1h 聚合趋势 + 二分分割（Binary Segmentation）。
- 非稳态检测：阀门/转速滚动变化 MAD（绝对偏差均值）。
- 模式判定：
  - Pump 模式：高转速 + 小阀门
  - Valve 模式：转速接近常值 + 阀门变动
- 切片保存：每段作为独立设备子目录保存。

---

### 模块 2：特征提取

文件：`2_Model_Feature.py`

功能定位：
- 读取分段后的设备数据
- 计算 9 个特征 + 1 个目标变量（`dP_pump`）
- 汇总成设备级特征表

输入格式：
- 设备目录结构：每个设备子文件夹下包含必要 CSV
- 必需文件：二次侧泵转速、二次侧泵入口/出口压力、二次侧入口/出口压力

输出内容：
- 输出 `CDU_Features_xxx.csv`
- 列包含：
  - `speed, n_squared, dP_HEX, dP_filter, resistance_ratio, phi_filter, phi_HEX, virtual_flow, system_impedance, dP_pump`

输出示例：
```
CDU_Features_治理数据_分段处理.csv
```

关键技术点与公式：
- 转速归一化平方：
  - $$
    n^2 = \left(\frac{\mathrm{speed}}{100}\right)^2
    $$
- 泵压差：
  - $$
    dP_{\mathrm{pump}} = P_{\mathrm{pump\_out}} - P_{\mathrm{pump\_in}}
    $$
- 板换压差：
  - $$
    dP_{\mathrm{HEX}} = P_{\mathrm{sys\_in}} - P_{\mathrm{pump\_in}}
    $$
- 过滤器压差：
  - $$
    dP_{\mathrm{filter}} = P_{\mathrm{pump\_out}} - P_{\mathrm{sys\_out}}
    $$
- 被动压降：
  - $$
    dP_{\mathrm{passive}} = dP_{\mathrm{HEX}} + dP_{\mathrm{filter}}
    $$
- 阻力比与占比：
  - $$
    \mathrm{resistance\_ratio} = \frac{dP_{\mathrm{HEX}}}{dP_{\mathrm{filter}}}
    $$
  - $$
    \phi_{\mathrm{filter}} = \frac{dP_{\mathrm{filter}}}{dP_{\mathrm{passive}}}
    $$
  - $$
    \phi_{\mathrm{HEX}} = \frac{dP_{\mathrm{HEX}}}{dP_{\mathrm{passive}}}
    $$
- 虚拟流量：
  - $$
    Q_{\mathrm{virtual}} \propto \sqrt{dP_{\mathrm{passive}}}
    $$
- 系统阻抗：
  - $$
    K_{\mathrm{sys}} \approx \frac{dP_{\mathrm{passive}}}{n^2}
    $$

---

### 模块 3：特征可视化

文件：`3_Model_Visualization.py`

功能定位：
- 特征相关性分析 + 机理模型全景分析
- 输出 Nature 风格图表（PNG）

输入格式：
- 预处理后的特征 CSV（由模块 2 生成）

输出内容：
- 输出目录：`Visualization_Output/`
- 图表包括：
  1. PairGrid 特征相关性
  2. 机理模型全景图
  3. 特征-残差相关矩阵
  4. 相关性热力图

关键技术点与公式：
- 机理模型采用稳健回归：Huber 回归
  - 目标：
    $$
    dP_{\mathrm{pump}} = \alpha n^2 + \beta
    $$
- 残差定义：
  - $$
    \varepsilon = dP_{\mathrm{pump}} - \hat{dP}_{\mathrm{pump}}
    $$
- 残差 3σ 清洗：
  - 仅保留
    $$
    |\varepsilon| \le 3\sigma
    $$
- 分区间分析：按 \(n^2\) 区间分段计算相关性并绘制矩阵。

---

### 模块 4：模型训练（重点）

文件：`4_Model_Train.py`

功能定位：
- 训练三类模型：
  - 机理模型（Huber）
  - ML 模型（RandomForest / Ridge / KNN / SVR）
  - MLRL（机理残差 + ML）
- 按 \(n^2\) 区间分 Bin 独立训练
- 基于训练残差计算异常阈值（P99.75）
- 导出训练预测结果、离群点与可视化图表

输入格式：
- 模块 2 输出的特征 CSV

输出内容：
- 模型文件：`CDU_Models.pkl`
- 训练结果目录：`Training_Results/`
  - `Training_Predictions.csv`
  - `Outliers_ML/*.csv`、`Outliers_MLRL/*.csv`
  - `Train_Fit_RandomForest.png`
  - `Train_PumpCurve_RandomForest.png`
  - `Residual_Distribution.png`

核心机理与公式：
1. 机理模型（全局 Huber 回归）

$$
\hat{dP}_{\mathrm{pump}} = \alpha n^2 + \beta
$$

2. ML 模型（按 bin 训练）

$$
\hat{dP}_{\mathrm{pump}}^{\mathrm{ML}} = f_{\mathrm{ML}}(\mathbf{x})
$$

其中 \(\mathbf{x}\) 为 9 维特征向量。

3. MLRL 模型（机理 + 残差学习）

$$
\hat{r} = f_{\mathrm{ML}}(\mathbf{x}),\quad r = dP_{\mathrm{pump}} - \hat{dP}_{\mathrm{pump}}
$$

$$
\hat{dP}_{\mathrm{pump}}^{\mathrm{MLRL}} = \hat{dP}_{\mathrm{pump}}^{\mathrm{Mech}} + \hat{r}
$$

4. 阈值判定（异常检测）

训练集残差分布的 P99.75 分位作为阈值：

$$
\tau = P_{99.75}\left(|dP_{\mathrm{pump}}-\hat{dP}_{\mathrm{pump}}|\right)
$$

当 \(|dP_{\mathrm{pump}}-\hat{dP}_{\mathrm{pump}}| > \tau\) 时判为异常。

---

### 模块 5：模型测试与报告（重点）

文件：`5_Model_Test.py`

功能定位：
- 加载训练好的 PKL 模型
- 对测试数据执行预测与异常判断
- 生成检测报告与交互式可视化

输入格式：
- 模型文件：`CDU_Models.pkl`
- 测试特征 CSV（与训练格式一致）

输出内容：
- `Test_Results/Detection_Report.csv`
- `Residual_Interactive_Combined.html`（Plotly 交互残差图）

测试过程关键逻辑：
1. 机理模型预测：
   - 计算
     $$
     \hat{dP}_{\mathrm{pump}}^{\mathrm{Mech}}
     $$
     与残差
     $$
     |\varepsilon|
     $$
2. ML 预测：
   - 对每个 bin 使用对应模型预测
     $$
     \hat{dP}_{\mathrm{pump}}^{\mathrm{ML}}
     $$
3. MLRL 预测：
   - 预测残差
     $$
     \hat{r}
     $$
     再叠加机理预测
4. 异常判定：
   - 依据训练阶段保存的阈值
     $$
     \tau_{\mathrm{bin}}
     $$
     判定是否异常

输出报告字段示例：
```
设备ID, n_squared, Pred_Mech, Resid_Mech, Is_Anomaly_Mech,
Pred_ML_RandomForest, Resid_ML_RandomForest, Is_Anomaly_ML_RandomForest,
Pred_MLRL_RandomForest, Resid_MLRL_RandomForest, Is_Anomaly_MLRL_RandomForest
```

---

## 三、输入输出格式示例（汇总）

单指标 CSV 标准格式：
```
Time,value
2024-01-01 00:00:00,4.23
2024-01-01 00:01:00,4.25
```

特征 CSV 示例：
```
device_id,speed,n_squared,dP_HEX,dP_filter,resistance_ratio,phi_filter,phi_HEX,virtual_flow,system_impedance,dP_pump
CDU_001,78.3,0.613,1.23,0.92,1.34,0.43,0.57,1.48,2.01,2.15
```

---

## 四、建议使用顺序

1. `0_Model_Check.py` → 生成治理数据（校验 + 异常切分）
2. `1_Model_Preprocessing.py` → 稳态切片与模式分类
3. `2_Model_Feature.py` → 生成训练特征表
4. `3_Model_Visualization.py` → 分析与可视化
5. `4_Model_Train.py` → 模型训练与保存
6. `5_Model_Test.py` → 测试与异常报告

---

## 五、运行环境建议

- Python 3.8+
- 主要依赖：
  - pandas / numpy / matplotlib / scikit-learn / scipy
  - tqdm / tkinter
  - plotly（可选，测试交互图）

如需补充参数调优、结构调整或新增异常类型，建议从 `1_Model_Preprocessing.py` 与 `4_Model_Train.py` 两处开始。
