# IMU 数据处理详解

本文档详细讲解 `IMU dataprocessing/` 文件夹下所有代码的数据处理逻辑。

---

## 一、文件清单与职责

| 文件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `data_process.m` | 核心处理：去重力+滤波 | Xsens .txt 原始数据 | .xlsx 滤波特征 |
| `fengge.m` | STS动作周期分割 | 连续时间序列 | 单次STS片段 |
| `XSENS_Adapter.m` | 串口实时数据接收 | USB串口 | 实时数据流 |
| `XsensDataView.m` | 数据可视化查看 | .txt 数据文件 | 波形图 |
| `quaternion_library/` | 四元数运算工具库 | 四元数数组 | 旋转/转换结果 |
| `ximu_matlab_library/` | x-IMU数据类库 | 数据文件 | 结构化数据对象 |
| `AHRS.m` | Mahony姿态融合算法 | 加速度+陀螺仪 | 四元数方向 |

---

## 二、data_process.m — 核心处理流程

这是最重要的文件，完成从原始传感器数据到可用特征的全部转换。

### 2.1 完整代码逻辑（逐行解析）

```matlab
clear; close all; clc;
addpath('quaternion_library');   % 加载四元数运算库
addpath('ximu_matlab_library');  % 加载x-IMU数据类库
```
**作用**：清空工作区，添加依赖库路径。

---

```matlab
XSENS_path = 'C:\Users\...\';   % 原始数据文件夹路径
XSENS_name = '*.txt';            % 匹配所有txt文件
XSENSPath = dir([XSENS_path, XSENS_name]);  % 获取文件列表
```
**作用**：扫描指定文件夹下的所有 `.txt` 数据文件。

---

```matlab
for i = 1:Length   % 遍历每个文件
    SamplePeriod = 1/100;  % 采样周期 = 0.01秒（100Hz）
    temp = importdata(strcat(XSENS_path, XSENSPath(i).name), '\t', 6);
    XSENSDATA = temp.data;
```
**作用**：逐个读取 `.txt` 文件，跳过前6行头信息，用Tab分隔符解析数据矩阵。

---

```matlab
    % 提取原始传感器数据
    Accelerometer = XSENSDATA(:,2:4);      % 3轴加速度 [m/s²]
    Gyroscope = XSENSDATA(:,5:7);          % 3轴角速度 [rad/s]
    raw_quaternion = XSENSDATA(:,8:11);    % 传感器输出的四元数
```
**作用**：从数据矩阵中提取三类信号。四元数是Xsens传感器内置融合算法的输出，表示传感器相对于地球的方向。

---

### 2.2 关键步骤：四元数去重力

```matlab
    Ref_quaternion = quaternConj(raw_quaternion);  % 计算四元数共轭
```
**原理**：四元数共轭 q* = [w, -x, -y, -z]，表示反向旋转。由于Xsens输出的四元数描述"地球→传感器"的旋转，共轭后得到"传感器→地球"的旋转。

---

```matlab
    raw_g_acc = quaternRotate(Accelerometer/9.8, quaternConj(Ref_quaternion));
```
**原理**：这一步执行的是论文公式(1)的四元数旋转：
```
a_W = q ⊗ (0, a_I) ⊗ q*
```
将传感器坐标系下的加速度旋转到世界（地球）坐标系。

`quaternRotate` 函数的实现（`quaternion_library/quaternRotate.m`）：
```matlab
function v = quaternRotate(v, q)
    v0XYZ = quaternProd(quaternProd(q, [zeros(row,1) v]), quaternConj(q));
    v = v0XYZ(:, 2:4);
end
```
即：先把三维向量扩展为纯四元数 `[0, vx, vy, vz]`，然后执行 `q × v × q*` 双重四元数乘法，取结果的虚部。

---

```matlab
    raw_linAcc_2 = raw_g_acc - [zeros(row, 2) ones(row, 1)];
```
**原理**：这是论文公式(2)：`a = a_W - a_g`

在世界坐标系下减去重力分量 `[0, 0, 1]`（注意：因为之前除以了9.8归一化，所以重力是1g=[0,0,1]）。

```matlab
    raw_linAcc_3 = (raw_g_acc - [zeros(row, 2) ones(row, 1)]) * 9.8;
```
再乘回9.8，得到单位为 m/s² 的线性加速度。

---

### 2.3 关键步骤：巴特沃斯低通滤波

```matlab
    filtCutOff = 1.8;  % 截止频率 1.8 Hz
    [b, a] = butter(12, (2*filtCutOff)/(1/SamplePeriod), 'low');
    acc_Filt = filtfilt(b, a, raw_linAcc_2_z);
```

**原理**：
- `butter(12, ...)` — 设计12阶巴特沃斯低通滤波器
- 归一化截止频率 = `2 × 1.8 / 100 = 0.036`（相对于奈奎斯特频率）
- `filtfilt` — 零相位数字滤波（前后各滤一次，消除相位延迟）

**为什么截止1.8Hz？**
- STS动作是低频运动（主要在0-2Hz）
- 高于2Hz的成分主要是噪声和肌肉颤动
- 论文引用Day et al.[21]关于低通滤波截止频率的研究

---

### 2.4 角速度处理

```matlab
    gyr_y = Gyroscope_Y * 180/pi;  % 从 rad/s 转换为 °/s
    
    filtCutOff = 1.8;
    [b, a] = butter(12, (2*filtCutOff)/(1/SamplePeriod), 'low');
    gyr_magFilt = filtfilt(b, a, gyr_y);
```
**作用**：提取Y轴角速度（矢状面内的前后摆动角速度），转换单位并滤波。

---

### 2.5 输出保存

```matlab
    all_1 = [raw_linAcc_2_z'; gyr_y'; acc_Filt'; gyr_magFilt'];
    all_2 = all_1';
    xlswrite(x2, all_2);  % 保存为Excel文件
```
**输出**：4列数据——原始垂直加速度、原始Y轴角速度、滤波后加速度、滤波后角速度。

---

## 三、fengge.m — 动作周期分割

### 3.1 功能
将连续的多次STS动作数据分割成单独的动作周期。

### 3.2 分割原理
```
连续信号: ___/\___/\___/\___
                ↓ 峰值检测
单独动作:   [/\]  [/\]  [/\]
```

使用MATLAB的 `findpeaks` 或手动阈值检测，基于垂直加速度的峰值特征将连续信号切割为独立的STS动作片段。

### 3.3 关键逻辑
1. 检测加速度信号中的显著峰值（STS过渡期间的最大加速度）
2. 在相邻峰值之间的谷值处切割
3. 每个切割后的片段对应一次完整的STS动作

---

## 四、XSENS_Adapter.m — 串口数据接收

### 4.1 功能
实时从Xsens Awinda Station接收传感器数据的串口适配器。

### 4.2 工作流程
```
Xsens MTw传感器 → 2.4GHz无线 → Awinda Station → USB → PC串口
                                                          ↓
                                              XSENS_Adapter.m (解析)
                                                          ↓
                                              实时数据（加速度/角速度/四元数）
```

### 4.3 关键步骤
1. 打开串口连接（波特率设置）
2. 读取数据帧
3. 解析帧头、数据长度、校验和
4. 提取各传感器通道数据
5. 组装为标准格式输出

---

## 五、XsensDataView.m — 数据可视化

### 5.1 功能
快速查看和验证IMU数据质量的可视化工具。

### 5.2 显示内容
- 3轴原始加速度波形
- 3轴角速度波形
- 欧拉角变化曲线
- 四元数分量变化

### 5.3 用途
- 确认传感器数据采集正常
- 检查是否有数据丢失或异常
- 初步观察STS动作模式

---

## 六、quaternion_library/ — 四元数工具库

### 6.1 核心函数

| 函数 | 输入 | 输出 | 公式 |
|------|------|------|------|
| `quaternProd(a, b)` | 两个四元数 | 四元数乘积 | a ⊗ b |
| `quaternConj(q)` | 四元数 | 共轭四元数 | q* = [w,-x,-y,-z] |
| `quaternRotate(v, q)` | 向量+四元数 | 旋转后的向量 | q⊗v⊗q* |
| `quatern2euler(q)` | 四元数 | 欧拉角[φ,θ,ψ] | 四元数→RPY |
| `quatern2rotMat(q)` | 四元数 | 3×3旋转矩阵 | 四元数→DCM |
| `euler2rotMat(φ,θ,ψ)` | 欧拉角 | 旋转矩阵 | RPY→DCM |
| `axisAngle2quatern(axis,angle)` | 轴+角 | 四元数 | 轴角→四元数 |

### 6.2 四元数乘法详解（`quaternProd.m`）

```matlab
function ab = quaternProd(a, b)
    % 四元数汉密尔顿乘法
    ab(:,1) = a(:,1).*b(:,1) - a(:,2).*b(:,2) - a(:,3).*b(:,3) - a(:,4).*b(:,4);
    ab(:,2) = a(:,1).*b(:,2) + a(:,2).*b(:,1) + a(:,3).*b(:,4) - a(:,4).*b(:,3);
    ab(:,3) = a(:,1).*b(:,3) - a(:,2).*b(:,4) + a(:,3).*b(:,1) + a(:,4).*b(:,2);
    ab(:,4) = a(:,1).*b(:,4) + a(:,2).*b(:,3) - a(:,3).*b(:,2) + a(:,4).*b(:,1);
end
```

支持向量化操作（每行一个四元数），可批量处理整个时间序列。

### 6.3 四元数旋转详解（`quaternRotate.m`）

```matlab
function v = quaternRotate(v, q)
    [row col] = size(v);
    v0XYZ = quaternProd(quaternProd(q, [zeros(row,1) v]), quaternConj(q));
    v = v0XYZ(:, 2:4);
end
```

数学等价于：
```
v' = R(q) × v
```
其中 R(q) 是四元数对应的3×3旋转矩阵。但使用四元数乘法更高效、数值更稳定。

---

## 七、AHRS.m — 姿态融合算法

### 7.1 算法类型
Mahony 互补滤波器（非Madgwick）。

### 7.2 原理
```
陀螺仪积分（快速但有漂移）
         ↓
    + 加速度校正（消除漂移）
         ↓
    = 融合后的姿态四元数
```

### 7.3 参数
| 参数 | 默认值 | 含义 |
|------|--------|------|
| SamplePeriod | 1/256 | 采样周期 |
| Kp | 2 | 比例增益（加速度校正强度） |
| Ki | 0 | 积分增益（消除恒定偏移） |
| KpInit | 200 | 初始化阶段的高增益 |
| InitPeriod | 5 | 初始化持续时间（秒） |

### 7.4 在本项目中的角色
**注意**：`data_process.m` 实际上**没有使用** AHRS.m 进行姿态融合。代码直接使用 Xsens 传感器输出的四元数（XSENSDATA 列8-11），这些四元数由传感器内置的融合算法计算。AHRS.m 作为备选方案保留在项目中。

---

## 八、数据处理完整流水线图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Xsens MTw 传感器                               │
│         采集 100Hz：3轴加速度 + 3轴角速度 + 四元数               │
└────────────────────────────┬────────────────────────────────────┘
                             │ 导出 .txt（tab分隔，6行头信息）
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  data_process.m                                   │
│                                                                   │
│  ① 读取原始数据（importdata）                                    │
│       ↓                                                           │
│  ② 提取加速度 [列2-4] + 四元数 [列8-11]                        │
│       ↓                                                           │
│  ③ 四元数共轭（quaternConj）                                     │
│       ↓                                                           │
│  ④ 四元数旋转（quaternRotate）→ 世界坐标系加速度                │
│       ↓                                                           │
│  ⑤ 减去重力 [0,0,9.8] → 线性加速度                             │
│       ↓                                                           │
│  ⑥ 提取垂直分量（Z轴线性加速度）                                │
│       ↓                                                           │
│  ⑦ 12阶巴特沃斯低通滤波（1.8Hz）→ 去噪                        │
│       ↓                                                           │
│  ⑧ 提取Y轴角速度，rad/s → °/s 转换，滤波                      │
│       ↓                                                           │
│  ⑨ 输出4列数据 → .xlsx                                          │
│     [原始加速度Z, 原始角速度Y, 滤波加速度Z, 滤波角速度Y]        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    fengge.m                                        │
│  基于峰值检测，分割连续信号为单次STS动作片段                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  后续数据集构建流程
                  （phase segmentaion and build dataset/）
```

---

## 九、关键数学概念总结

### 9.1 坐标系转换
- **传感器坐标系（Body Frame）**：随传感器运动，X/Y/Z轴固定在传感器上
- **世界坐标系（World Frame）**：固定于地球，Z轴指向天空
- **转换方式**：通过四元数旋转实现两个坐标系之间的变换

### 9.2 为什么要去重力？
传感器测量的是"比力"（specific force），包含重力分量：
```
a_measured = a_motion + g（在传感器坐标系下）
```
要获得纯运动加速度：
1. 先将加速度旋转到世界坐标系（重力方向固定为 [0,0,g]）
2. 再减去重力分量

### 9.3 为什么用低通滤波？
- STS动作频率：0.5-2 Hz
- 传感器噪声和肌肉颤动：>3 Hz
- 1.8Hz 截止频率保留运动信号，去除噪声
