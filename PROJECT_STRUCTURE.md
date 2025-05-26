# 多级零信任IoT检测系统 - 项目结构

## 📁 项目目录结构

```
iotdetection/
├── 📁 src/                                    # 核心源代码
│   ├── enhanced_multi_stage_detector.py       # 🎯 增强型多级检测器 (主检测引擎)
│   ├── multi_stage_zero_trust_detector.py     # 🛡️ 多级零信任检测器
│   ├── advanced_feature_optimizer.py          # ⚡ 高级特征优化器
│   └── cic_iot_data_processor.py              # 📊 CIC-IoT数据处理器
│
├── 📁 threat_intelligence/                    # 威胁情报模块
│   ├── demo_threat_intelligence.py            # 🔍 威胁情报演示系统
│   ├── threat_intelligence_updater.py         # 🔄 威胁情报更新器
│   ├── threat_intel_demo.json                 # 📋 威胁情报演示数据
│   ├── threat_intel_config.json               # ⚙️ 威胁情报配置
│   ├── threat_intelligence_db.json            # 🗄️ 威胁情报数据库
│   └── feature_patterns_db.json               # 🎨 特征模式数据库
│
├── 📁 utils/                                  # 工具和辅助模块
│   ├── performance_evaluation.py              # 📈 性能评估工具
│   └── focal_loss.py                          # 🎯 焦点损失函数
│
├── 📁 data/                                   # 数据目录
│   └── cic_iot_2023/                         # CIC-IoT-2023数据集
│
├── 📁 models/                                 # 模型存储
│   └── cache/                                 # 模型缓存
│
├── 📁 docs/                                   # 文档
│   ├── PAPER_FRAMEWORK.md                     # 📄 英文论文框架
│   ├── PAPER_FRAMEWORK_CN.md                  # 📄 中文论文框架
│   └── paper.txt                              # 📝 论文草稿
│
├── 📁 reports/                                # 项目报告
│   ├── PROJECT_COMPLETION_SUMMARY.md          # ✅ 项目完成总结
│   ├── TRAINING_PERFORMANCE_REPORT.md         # 🏃 训练性能报告
│   ├── FINAL_SYSTEM_STATUS.md                 # 🎯 最终系统状态
│   └── SYSTEM_STATUS_REPORT.md                # 📊 系统状态报告
│
├── 📁 results/                                # 实验结果
│   ├── performance_evaluation_charts.png      # 📊 性能评估图表
│   ├── detailed_performance_results.json      # 📋 详细性能结果
│   └── performance_summary.csv                # 📈 性能摘要
│
├── 📁 venv/                                   # Python虚拟环境
├── README.md                                  # 📖 项目说明
├── requirements.txt                           # 📦 依赖包列表
└── PROJECT_STRUCTURE.md                       # 📁 项目结构说明 (本文件)
```

## 🎯 核心模块说明

### 1. 主检测引擎 (`src/enhanced_multi_stage_detector.py`)
- **功能**: 四级检测架构的核心实现
- **特性**: 
  - 边界检测器 (Isolation Forest)
  - 行为分析器 (Random Forest)
  - 异常检测器 (One-Class SVM)
  - 威胁情报分析器 (Neural Network)
- **创新**: Mirai变种智能检测、实时威胁情报同步

### 2. 特征优化器 (`src/advanced_feature_optimizer.py`)
- **功能**: 智能特征工程和选择
- **特性**: 
  - 多算法特征选择 (F统计量、互信息、随机森林)
  - NaN值处理和数据清洗
  - 特征重要性分析
- **优化**: 从80个特征优化到37个最佳特征

### 3. 威胁情报系统 (`threat_intelligence/`)
- **功能**: 威胁情报收集、更新和分析
- **数据源**: 
  - Mirai变种数据库
  - 攻击特征模式库
  - 实时威胁情报更新
- **自动化**: 威胁情报与检测器自动同步

### 4. 性能评估 (`utils/performance_evaluation.py`)
- **功能**: 全面的性能测试和评估
- **指标**: 准确率、召回率、F1分数、混淆矩阵
- **可视化**: 性能图表和详细报告生成

## 🚀 系统性能指标

### 最新测试结果 (500,000样本训练)
- **分类准确率**: 99.939%
- **威胁检测率**: 99.609%
- **误报率**: 0.021%
- **处理速度**: 550,533样本/秒

### 检测能力
- **Mirai变种检测**: 支持6种变种 (包括数据集实际类型)
- **DDoS攻击检测**: 高精度识别
- **零日攻击检测**: 基于异常检测的未知威胁识别

## 🔧 使用说明

### 1. 环境设置
```bash
cd iotdetection
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. 运行主检测器
```bash
python src/enhanced_multi_stage_detector.py
```

### 3. 威胁情报更新
```bash
python threat_intelligence/threat_intelligence_updater.py
```

### 4. 性能评估
```bash
python utils/performance_evaluation.py
```

## 📊 项目特色

### 🌟 创新点
1. **多级零信任架构**: 四级检测器协同工作
2. **智能特征工程**: 自动化特征选择和优化
3. **实时威胁情报**: 动态更新和同步机制
4. **Mirai变种检测**: 专门针对IoT僵尸网络
5. **高性能处理**: 大规模数据实时处理能力

### 🎯 技术栈
- **机器学习**: Scikit-learn, TensorFlow
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **威胁情报**: 自研威胁情报框架

### 📈 应用场景
- IoT网络安全监控
- 企业网络威胁检测
- 智能家居安全防护
- 工业IoT安全监控

---

**项目状态**: ✅ 完成开发和测试  
**最后更新**: 2024年12月  
**版本**: v2.0 (多级零信任架构) 