# 多级零信任IoT检测系统

## 🎯 项目概述

基于CIC-IoT-2023数据集的多级零信任IoT网络安全威胁检测系统，采用四阶段检测架构，实现高精度IoT威胁检测和实时防护。

## 🚀 核心功能

### 1. 🛡️ 多阶段零信任检测
- **四阶段架构**: 边界检测 → 行为分析 → 异常检测 → 威胁情报
- **零信任神经网络**: 双头输出(分类+信任评估)
- **准确率**: 99.7%

### 2. 🔄 威胁情报实时更新
- **自动收集**: 外部威胁源 + 本地攻击日志
- **实时学习**: 自动学习新攻击模式
- **多线程监控**: 24/7不间断威胁监控

### 3. 🦠 Mirai变种专项检测
- **精准识别**: Echobot、Satori、Okiru等变种
- **100%置信度**: 已知变种检测
- **自适应学习**: 未知变种自动学习

### 4. 📊 性能评估系统
- **大规模测试**: 支持10万+样本评估
- **全面指标**: 准确率、检测率、误报率、处理速度
- **可视化报告**: 自动生成性能图表

## 📁 项目结构

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
├── 📁 data/                                   # 数据目录
├── 📁 models/                                 # 模型存储
├── 📁 venv/                                   # Python虚拟环境
├── README.md                                  # 📖 项目说明
├── requirements.txt                           # 📦 依赖包列表
└── PROJECT_STRUCTURE.md                       # 📁 项目结构说明
```

## ⚡ 快速开始

### 1. 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 准备数据集
# 将CIC-IoT-2023数据集放入 data/cic_iot_2023/merged_csv/ 目录
```

### 2. 核心功能测试
```bash
# 威胁情报和Mirai检测演示
python3 threat_intelligence/demo_threat_intelligence.py

# 多阶段零信任检测
python3 src/multi_stage_zero_trust_detector.py

# 性能评估 (选择测试模式)
python3 utils/performance_evaluation.py

# 增强版集成系统演示
python3 src/enhanced_multi_stage_detector.py --demo

# 增强版集成系统训练
python3 src/enhanced_multi_stage_detector.py
```

## 📊 性能指标

| 指标 | 数值 | 评级 |
|------|------|------|
| 分类准确率 | 99.939% | 🏆 优秀 |
| 威胁检测率 | 99.609% | 🏆 优秀 |
| 误报率 | 0.021% | 🏆 优秀 |
| 处理速度 | 550K样本/秒 | 🚀 极快 |
| Mirai检测 | 6种变种支持 | 🎯 精准 |

## 🔧 技术特色

- **零信任架构**: 永不信任，始终验证
- **四阶段检测**: 多层防护，全面覆盖
- **实时更新**: 自动威胁情报收集和学习
- **专项检测**: Mirai僵尸网络变种精准识别
- **高性能**: 大规模并发处理能力
- **自适应**: 自动学习新攻击模式

## 🎯 应用场景

- **企业IoT安全**: 保护智能设备和网络
- **运营商网络**: 大规模IoT设备监控
- **关键基础设施**: 工控系统和智慧城市
- **网络安全研究**: 威胁检测算法研究

## 📈 系统优势

1. **高精度检测**: 99.939%的分类准确率
2. **低误报率**: 仅0.021%的误报率
3. **实时防护**: 毫秒级威胁检测响应
4. **智能学习**: 自动适应新威胁
5. **易于部署**: 即插即用，快速部署

## 🔗 相关资源

- **数据集**: CIC-IoT-Dataset 2023
- **论文参考**: `paper.txt`
- **系统状态**: `SYSTEM_STATUS_REPORT.md`
- **配置文件**: `threat_intel_config.json`

## 📞 技术支持

如有问题或建议，请查看系统状态报告或联系开发团队。

---

**版本**: v2.0 增强版  
**状态**: 🟢 全功能正常运行  
**更新**: 2024年5月25日 