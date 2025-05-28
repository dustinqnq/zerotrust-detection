#!/usr/bin/env python3
"""
增强零信任IDS训练脚本
使用IoT-23数据集训练改进的三阶段模型
"""

import sys
import os
sys.path.append('../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from enhanced_autoencoder_ids import EnhancedZeroTrustIDS
from processors.iot23_processor import IoT23Processor
import torch
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import itertools
from sklearn.preprocessing import LabelEncoder

def prepare_iot23_data():
    """准备IoT-23数据集并处理数据不平衡问题"""
    print("\n=== 开始加载IoT-23数据集 ===")
    print("1. 查找数据文件...")
    
    # 查找数据文件
    data_dir = Path('../data/iot-23')  # 修改数据目录路径
    print(f"搜索目录: {data_dir.absolute()}")
    
    # 查找所有可用的捕获文件
    capture_dirs = list(data_dir.glob('CTU-IoT-Malware-Capture-*-*'))
    data_files = []
    for capture_dir in capture_dirs:
        labeled_files = list(capture_dir.glob('**/*.labeled'))
        data_files.extend(labeled_files)
    
    if not data_files:
        print("❌ 未找到IoT-23数据文件，请确保数据文件在正确的目录下")
        print(f"当前搜索目录: {data_dir.absolute()}")
        return None, None, None
    
    print(f"✓ 找到 {len(data_files)} 个数据文件:")
    for f in data_files:
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
    
    # 处理多个文件并合并
    all_features = []
    all_labels = []
    all_detailed_labels = []
    
    for file_idx, file_path in enumerate(data_files[:3]):  # 限制处理前3个文件以避免内存问题
        print(f"\n2.{file_idx+1} 正在处理文件: {file_path.name}")
        print("   开始读取数据...")
        
        # 读取和解析数据
        features_list = []
        labels_list = []
        detailed_labels_list = []
        processed_lines = 0
        total_lines = 20000  # 每个文件限制处理的行数
        
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i > total_lines:
                        break
                        
                    if i % 2000 == 0 and i > 0:  # 每处理2000行打印一次进度
                        print(f"   已处理 {i} 行...")
                        
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            fields = line.split('\t')  # 使用tab分隔符
                            
                            if len(fields) >= 21:  # 确保有足够的字段
                                # 提取数值特征（跳过前面的非数值字段，从时间戳等开始）
                                numeric_features = []
                                
                                # 从第1列开始（时间戳），选择一些数值字段
                                numeric_indices = [0, 3, 4, 5, 8, 9, 14, 16, 17, 18, 19]  # 选择数值字段的索引
                                
                                for idx in numeric_indices:
                                    if idx < len(fields):
                                        try:
                                            # 处理特殊值
                                            val_str = fields[idx].strip()
                                            if val_str == '-' or val_str == '(empty)' or val_str == '':
                                                val = 0.0
                                            else:
                                                val = float(val_str)
                                            numeric_features.append(val)
                                        except:
                                            numeric_features.append(0.0)
                                    else:
                                        numeric_features.append(0.0)
                                
                                # 确保特征维度一致（扩展到20维）
                                while len(numeric_features) < 20:
                                    numeric_features.append(0.0)
                                
                                # 限制到20维
                                numeric_features = numeric_features[:20]
                                    
                                # 获取标签 - 最后一列包含复合标签信息
                                if len(fields) >= 21:
                                    label_info = fields[-1].strip()
                                    # 解析复合标签："(empty)   Malicious   PartOfAHorizontalPortScan"
                                    label_parts = label_info.split()
                                    
                                    if len(label_parts) >= 2:
                                        basic_label = label_parts[1].strip().lower()  # Malicious/Benign
                                        detailed_label = label_parts[2] if len(label_parts) > 2 else 'Unknown'
                                        
                                        # 只保留有效的标签
                                        if basic_label in ['benign', 'malicious']:
                                            features_list.append(numeric_features)
                                            labels_list.append(basic_label)
                                            detailed_labels_list.append(detailed_label)
                                            processed_lines += 1
                                
                        except Exception as e:
                            continue  # 静默跳过解析失败的行
                            
        except Exception as e:
            print(f"❌ 文件读取出错: {str(e)}")
            continue
        
        print(f"   文件 {file_path.name} 处理完成: {processed_lines} 行")
        
        # 合并到总列表
        all_features.extend(features_list)
        all_labels.extend(labels_list)
        all_detailed_labels.extend(detailed_labels_list)
    
    print(f"\n3. 多文件数据合并完成:")
    print(f"   - 总有效样本数: {len(all_features)}")
    
    if not all_features:
        print("❌ 未能提取到有效数据")
        return None, None, None
    
    # 转换为DataFrame和数组
    print("\n4. 转换数据格式...")
    features_df = pd.DataFrame(all_features)
    labels_array = np.array(all_labels)
    detailed_labels_array = np.array(all_detailed_labels)
    
    # 创建完整的数据集DataFrame
    dataset_df = features_df.copy()
    dataset_df['basic_label'] = labels_array
    dataset_df['detailed_label'] = detailed_labels_array
    
    print(f"\n5. 原始数据统计:")
    print(f"   特征形状: {features_df.shape}")
    print(f"   基本标签分布:")
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = count / len(labels_array) * 100
        print(f"   - {label}: {count} ({percentage:.1f}%)")
    
    print(f"\n   详细标签分布 (Top 10):")
    detailed_counts = pd.Series(detailed_labels_array).value_counts()
    for label, count in detailed_counts.head(10).items():
        percentage = count / len(detailed_labels_array) * 100
        print(f"   - {label}: {count} ({percentage:.1f}%)")
    
    # 实现分层采样
    print("\n6. 实施分层采样策略...")
    
    # 分离良性和恶意样本
    benign_mask = dataset_df['basic_label'] == 'benign'
    malicious_mask = dataset_df['basic_label'] == 'malicious'
    
    benign_data = dataset_df[benign_mask]
    malicious_data = dataset_df[malicious_mask]
    
    print(f"   良性样本数量: {len(benign_data)}")
    print(f"   恶意样本数量: {len(malicious_data)}")
    
    # 对恶意样本按详细标签进行分层采样
    malicious_groups = malicious_data.groupby('detailed_label')
    stratified_malicious = []
    
    print("\n   恶意样本分层采样:")
    for detailed_label, group in malicious_groups:
        if len(group) >= 50:  # 如果样本充足，采样50个
            sampled = group.sample(n=50, random_state=42)
            print(f"   - {detailed_label}: 采样 50 个 (原有 {len(group)} 个)")
        elif len(group) >= 10:  # 如果样本较少但不是太少，全部使用
            sampled = group
            print(f"   - {detailed_label}: 使用全部 {len(group)} 个")
        else:  # 样本太少，使用过采样
            sampled = group.sample(n=10, replace=True, random_state=42)
            print(f"   - {detailed_label}: 过采样到 10 个 (原有 {len(group)} 个)")
        
        stratified_malicious.append(sampled)
    
    # 合并分层采样的恶意样本
    stratified_malicious_df = pd.concat(stratified_malicious, ignore_index=True)
    
    # 对良性样本进行采样，使数据相对平衡
    target_benign_size = min(len(stratified_malicious_df), len(benign_data), 500)
    if len(benign_data) >= target_benign_size:
        sampled_benign_df = benign_data.sample(n=target_benign_size, random_state=42)
    else:
        sampled_benign_df = benign_data.sample(n=target_benign_size, replace=True, random_state=42)
    
    print(f"   良性样本采样: {len(sampled_benign_df)} 个")
    
    # 合并最终数据集
    final_dataset = pd.concat([sampled_benign_df, stratified_malicious_df], ignore_index=True)
    
    # 随机打乱数据
    final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n7. 分层采样后数据统计:")
    print(f"   总样本数: {len(final_dataset)}")
    print(f"   基本标签分布:")
    basic_counts = final_dataset['basic_label'].value_counts()
    for label, count in basic_counts.items():
        percentage = count / len(final_dataset) * 100
        print(f"   - {label}: {count} ({percentage:.1f}%)")
    
    print(f"\n   详细标签分布:")
    detailed_counts = final_dataset['detailed_label'].value_counts()
    for label, count in detailed_counts.items():
        percentage = count / len(final_dataset) * 100
        print(f"   - {label}: {count} ({percentage:.1f}%)")
    
    # 提取特征和标签
    balanced_features = final_dataset.iloc[:, :-2]  # 除了最后两列标签外的所有特征
    balanced_labels = final_dataset['basic_label'].values
    
    # 特征预处理
    print("\n8. 特征预处理...")
    balanced_features = balanced_features.fillna(0)
    balanced_features = balanced_features.replace([np.inf, -np.inf], 0)
    
    print("✅ 数据准备完成!")
    print(f"   最终特征形状: {balanced_features.shape}")
    print(f"   包含的攻击变种数量: {final_dataset['detailed_label'].nunique()}")
    
    return balanced_features, balanced_labels, final_dataset

def create_synthetic_attack_labels(labels, features, detailed_labels_df=None):
    """创建合成的攻击标签用于训练"""
    print("=== 创建训练标签 ===")
    
    # 基础二分类标签 (0: 正常, 1: 攻击)
    binary_labels = np.array(['Benign' if 'benign' in str(label).lower() else 'Attack' for label in labels])
    
    # 如果有详细标签信息，使用它来创建更准确的分类
    if detailed_labels_df is not None and 'detailed_label' in detailed_labels_df.columns:
        detailed_labels = detailed_labels_df['detailed_label'].values
        print(f"使用详细标签信息，共 {len(np.unique(detailed_labels))} 种攻击变种")
        
        # 创建主攻击类型标签（基于详细标签）
        attack_types = []
        for detail_label in detailed_labels:
            detail_str = str(detail_label).lower()
            if 'benign' in detail_str or detail_str == '-':
                attack_types.append('Benign')
            elif any(keyword in detail_str for keyword in ['ddos', 'flood']):
                attack_types.append('DDoS')
            elif any(keyword in detail_str for keyword in ['mirai', 'okiru']):
                attack_types.append('Mirai')
            elif any(keyword in detail_str for keyword in ['scan', 'portscan', 'horizontalportscan']):
                attack_types.append('PortScan')
            elif any(keyword in detail_str for keyword in ['bot', 'torii', 'muhstik']):
                attack_types.append('Botnet')
            elif any(keyword in detail_str for keyword in ['c&c', 'cc', 'heartbeat']):
                attack_types.append('C&C')
            elif any(keyword in detail_str for keyword in ['attack', 'malicious']):
                attack_types.append('Attack')
            else:
                attack_types.append('Other')
        
        # 创建子类型标签（基于详细标签的具体类型）
        sub_types = []
        for detail_label in detailed_labels:
            detail_str = str(detail_label).lower()
            if 'benign' in detail_str or detail_str == '-':
                sub_types.append('Normal')
            elif 'tcp' in detail_str:
                sub_types.append('TCP_Attack')
            elif 'udp' in detail_str:
                sub_types.append('UDP_Attack')
            elif 'http' in detail_str:
                sub_types.append('HTTP_Attack')
            elif any(keyword in detail_str for keyword in ['c&c', 'cc', 'heartbeat']):
                sub_types.append('CC_Attack')
            elif any(keyword in detail_str for keyword in ['scan', 'portscan']):
                sub_types.append('Scan_Attack')
            elif 'partofhorizontalportscan' in detail_str.replace(' ', ''):
                sub_types.append('Horizontal_Scan')
            elif 'filedownload' in detail_str:
                sub_types.append('FileDownload_Attack')
            elif any(keyword in detail_str for keyword in ['mirai', 'okiru', 'muhstik']):
                sub_types.append('Botnet_Variant')
            else:
                sub_types.append('Generic_Attack')
                
    else:
        # 传统的标签创建方法（如果没有详细标签信息）
        print("使用基础标签信息创建攻击分类")
        
        # 创建主攻击类型标签
        attack_types = []
        for label in labels:
            label_str = str(label).lower()
            if 'benign' in label_str:
                attack_types.append('Benign')
            elif 'ddos' in label_str:
                attack_types.append('DDoS')
            elif 'mirai' in label_str or 'okiru' in label_str:
                attack_types.append('Mirai')
            elif 'scan' in label_str:
                attack_types.append('PortScan')
            elif 'bot' in label_str:
                attack_types.append('Botnet')
            elif 'c&c' in label_str:
                attack_types.append('C&C')
            else:
                attack_types.append('Other')
        
        # 创建子类型标签
        sub_types = []
        for label in labels:
            label_str = str(label).lower()
            if 'benign' in label_str:
                sub_types.append('Normal')
            elif 'tcp' in label_str:
                sub_types.append('TCP_Attack')
            elif 'udp' in label_str:
                sub_types.append('UDP_Attack')
            elif 'http' in label_str:
                sub_types.append('HTTP_Attack')
            elif 'c&c' in label_str:
                sub_types.append('CC_Attack')
            elif 'scan' in label_str:
                sub_types.append('Scan_Attack')
            else:
                sub_types.append('Generic_Attack')
    
    # 转换为数值标签
    # 二分类编码
    le_binary = LabelEncoder()
    binary_encoded = le_binary.fit_transform(binary_labels)
    
    # 主类型编码
    le_attack = LabelEncoder()
    attack_encoded = le_attack.fit_transform(attack_types)
    
    # 子类型编码
    le_sub = LabelEncoder()
    sub_encoded = le_sub.fit_transform(sub_types)
    
    print(f"标签创建完成:")
    print(f"  二分类标签: {len(np.unique(binary_labels))} 类")
    print(f"  主攻击类型: {len(np.unique(attack_types))} 类 - {np.unique(attack_types)}")
    print(f"  子攻击类型: {len(np.unique(sub_types))} 类 - {np.unique(sub_types)}")
    
    # 计算分布统计
    print(f"\n标签分布统计:")
    print("主攻击类型分布:")
    for attack_type in np.unique(attack_types):
        count = np.sum(np.array(attack_types) == attack_type)
        percentage = count / len(attack_types) * 100
        print(f"  - {attack_type}: {count} ({percentage:.1f}%)")
    
    print("子攻击类型分布:")
    for sub_type in np.unique(sub_types):
        count = np.sum(np.array(sub_types) == sub_type)
        percentage = count / len(sub_types) * 100
        print(f"  - {sub_type}: {count} ({percentage:.1f}%)")
    
    return {
        'binary': binary_encoded,
        'attack_type': attack_encoded,
        'sub_type': sub_encoded,
        'binary_classes': le_binary.classes_,
        'attack_classes': le_attack.classes_,
        'sub_classes': le_sub.classes_
    }

def train_enhanced_model():
    """训练增强模型"""
    print("\n🚀 开始训练增强零信任IDS模型")
    print("=" * 50)
    
    # 1. 加载数据
    print("\n[1/10] 加载数据...")
    features, labels, dataset_df = prepare_iot23_data()
    if features is None:
        print("❌ 数据加载失败，退出训练")
        return
    
    # 2. 创建标签
    print("\n[2/10] 创建训练标签...")
    attack_labels = create_synthetic_attack_labels(labels, features, dataset_df)
    
    # 3. 分割数据集
    print("\n[3/10] 分割数据集...")
    X_train, X_test, y_bin_train, y_bin_test, y_att_train, y_att_test, y_sub_train, y_sub_test = train_test_split(
        features, attack_labels['binary'], attack_labels['attack_type'], attack_labels['sub_type'], 
        test_size=0.2, random_state=42, stratify=attack_labels['binary']
    )
    
    X_train, X_val, y_bin_train, y_bin_val, y_att_train, y_att_val, y_sub_train, y_sub_val = train_test_split(
        X_train, y_bin_train, y_att_train, y_sub_train,
        test_size=0.2, random_state=42, stratify=y_bin_train
    )
    
    print(f"   训练集: {X_train.shape}")
    print(f"   验证集: {X_val.shape}")
    print(f"   测试集: {X_test.shape}")
    
    # 4. 初始化模型
    print("\n[4/10] 初始化增强模型...")
    input_dim = features.shape[1]
    
    # 计算类别数量
    num_binary_classes = len(attack_labels['binary_classes'])
    num_attack_classes = len(attack_labels['attack_classes'])
    num_sub_classes = len(attack_labels['sub_classes'])
    
    print(f"   二分类类别数: {num_binary_classes}")
    print(f"   主攻击类别数: {num_attack_classes}")
    print(f"   子攻击类别数: {num_sub_classes}")
    
    model = EnhancedZeroTrustIDS(input_dim=input_dim)
    
    # 5. 数据预处理
    print("\n[5/10] 数据预处理...")
    print("   - 标准化特征...")
    
    # 将编码后的标签转换为字符串形式以便模型处理
    y_bin_train_str = attack_labels['binary_classes'][y_bin_train]
    y_att_train_str = attack_labels['attack_classes'][y_att_train]
    y_sub_train_str = attack_labels['sub_classes'][y_sub_train]
    
    X_train_scaled, y_bin_train_enc, y_att_train_enc, y_sub_train_enc = model.prepare_data(
        X_train, y_bin_train_str, y_att_train_str, y_sub_train_str
    )
    
    print("   - 处理验证集...")
    X_val_scaled = model.scaler.transform(X_val)
    
    # 对验证集标签进行相同的转换
    y_bin_val_str = attack_labels['binary_classes'][y_bin_val]
    y_att_val_str = attack_labels['attack_classes'][y_att_val]
    y_sub_val_str = attack_labels['sub_classes'][y_sub_val]
    
    y_bin_val_enc = model.label_encoders['binary'].transform(y_bin_val_str)
    y_att_val_enc = model.label_encoders['multi_stage1'].transform(y_att_val_str)
    y_sub_val_enc = model.label_encoders['multi_stage2'].transform(y_sub_val_str)
    
    print("   - 处理测试集...")
    X_test_scaled = model.scaler.transform(X_test)
    
    # 对测试集标签进行相同的转换
    y_bin_test_str = attack_labels['binary_classes'][y_bin_test]
    y_bin_test_enc = model.label_encoders['binary'].transform(y_bin_test_str)
    
    # 6. 训练阶段1
    print("\n[6/10] 🔥 Training Stage 1 - Shallow Classifier")
    print("   Starting training...")
    model.train_stage1(
        X_train_scaled, y_bin_train_enc, y_att_train_enc,
        X_val_scaled, y_bin_val_enc, y_att_val_enc,
        epochs=30  # 增加训练轮数
    )
    
    # 7. 训练阶段2
    print("\n[7/10] 🔥 训练阶段2 - 深层分类器")
    print("   开始训练...")
    model.train_stage2(
        X_train_scaled, y_bin_train_enc, y_sub_train_enc,
        X_val_scaled, y_bin_val_enc, y_sub_val_enc,
        epochs=30  # 增加训练轮数
    )
    
    # 8. 准备自编码器训练数据
    print("\n[8/10] Preparing Autoencoder Training Data...")
    normal_mask = y_bin_train_enc == 0
    X_normal = X_train_scaled[normal_mask]
    
    print(f"   Normal traffic samples: {len(X_normal)} / {len(X_train_scaled)}")
    print(f"   Ratio: {len(X_normal)/len(X_train_scaled)*100:.1f}%")
    
    if len(X_normal) < 100:
        print("\n⚠️  Warning: Too few normal samples, using mixed training strategy")
        print("   - Using all samples with weighted normal samples")
        sample_weights = np.ones(len(X_train_scaled))
        sample_weights[normal_mask] = 3.0
        X_normal = X_train_scaled
        print(f"   - Normal sample weight=3.0, anomaly sample weight=1.0")
    elif len(X_normal) < 500:
        print("\n⚠️  Limited normal samples, might affect performance")
    else:
        print("\n✓ Sufficient normal samples")
    
    # 9. 训练阶段3 - 自编码器
    print("\n[9/10] 🔥 Training Stage 3 - Autoencoder")
    if len(X_normal) >= 100:
        print("   Starting training...")
        ae_losses = model.train_autoencoder(X_normal, epochs=50)  # 增加训练轮数
    else:
        print("❌ Skipping autoencoder training: insufficient normal samples")
        ae_losses = ([], [])
    
    # 10. 模型评估
    print("\n[10/10] 📊 模型评估")
    print("   计算性能指标...")
    results = model.evaluate(X_test_scaled, y_bin_test_enc)
    
    # 数据平衡性分析
    print("\n=== 数据集平衡性分析 ===")
    test_benign_ratio = np.mean(y_bin_test_enc == 0) * 100
    test_malicious_ratio = np.mean(y_bin_test_enc == 1) * 100
    
    print(f"测试集分布:")
    print(f"- 良性样本: {test_benign_ratio:.1f}%")
    print(f"- 恶意样本: {test_malicious_ratio:.1f}%")
    
    # 评估数据平衡性
    imbalance_ratio = max(test_benign_ratio, test_malicious_ratio) / min(test_benign_ratio, test_malicious_ratio)
    
    if imbalance_ratio > 10:
        print(f"\n⚠️  极度不平衡 (比例 {imbalance_ratio:.1f}:1)")
    elif imbalance_ratio > 3:
        print(f"\n⚠️  轻度不平衡 (比例 {imbalance_ratio:.1f}:1)")
    else:
        print(f"\n✓ 数据相对平衡 (比例 {imbalance_ratio:.1f}:1)")
    
    # 详细性能指标
    predictions = model.predict_binary(X_test_scaled)
    precision, recall, f1, _ = precision_recall_fscore_support(y_bin_test_enc, predictions, average=None)
    
    print("\n=== 详细性能指标 ===")
    print("良性类:")
    print(f"- 精确度: {precision[0]:.3f}")
    print(f"- 召回率: {recall[0]:.3f}")
    print(f"- F1分数: {f1[0]:.3f}")
    
    print("\n恶意类:")
    print(f"- 精确度: {precision[1]:.3f}")
    print(f"- 召回率: {recall[1]:.3f}")
    print(f"- F1分数: {f1[1]:.3f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_bin_test_enc, predictions)
    print("\n混淆矩阵:")
    print("          预测")
    print("实际    良性  恶意")
    print(f"良性    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"恶意    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # 更新results字典
    results.update({
        'detailed_precision': precision.tolist(),
        'detailed_recall': recall.tolist(),
        'detailed_f1': f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'data_balance_ratio': float(imbalance_ratio),
        'test_benign_ratio': float(test_benign_ratio),
        'test_malicious_ratio': float(test_malicious_ratio)
    })
    
    # 11. 保存模型
    print("\n💾 保存模型...")
    model.save_models()
    
    # 12. 生成可视化
    print("\n📈 生成训练报告...")
    generate_training_report(ae_losses, results)
    
    return model, results

def generate_training_report(ae_losses, results):
    """Generate training report with visualizations"""
    print("\n📈 Generating training report...")
    
    # Create save directory
    save_dir = Path('./reports')
    save_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    
    # Plot autoencoder loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    train_losses, val_losses = ae_losses
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.7)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.7)
    ax1.set_title('Autoencoder Training Progress', pad=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = np.array(results['confusion_matrix'])
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.set_title('Confusion Matrix')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2)
    
    # Add labels
    classes = ['Benign', 'Malicious']
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax2.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    ax2.set_ylabel('True label')
    ax2.set_xlabel('Predicted label')
    
    # Plot detection stage distribution
    ax3 = fig.add_subplot(gs[1, 0])
    stages = ['Stage 1', 'Stage 2', 'Stage 3']
    stage_counts = [results['stage_stats']['stage1'],
                   results['stage_stats']['stage2'],
                   results['stage_stats']['stage3']]
    
    colors = ['#3498db', '#e67e22', '#9b59b6']
    bars = ax3.bar(stages, stage_counts, color=colors)
    ax3.set_title('Detection Stage Distribution')
    ax3.set_ylabel('Number of Samples')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Plot performance metrics
    ax4 = fig.add_subplot(gs[1, 1])
    metrics = ['Precision', 'Recall', 'F1-Score']
    scores = [results['precision'], results['recall'], results['f1']]
    colors = ['#2ecc71', '#e74c3c', '#f1c40f']
    
    bars = ax4.bar(metrics, scores, color=colors)
    ax4.set_title('Model Performance Metrics')
    ax4.set_ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Save the figure
    plt.savefig(save_dir / 'training_report.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Training report saved to {save_dir}/training_report.png")

def main():
    """主函数"""
    print("🛡️  增强零信任IoT入侵检测系统训练")
    print("=" * 50)
    
    # 检查PyTorch和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
    print()
    
    try:
        # 开始训练
        model, results = train_enhanced_model()
        
        print("\n🎉 训练完成!")
        print(f"最终精确度: {results['precision']:.4f}")
        print(f"最终召回率: {results['recall']:.4f}")
        print(f"最终F1分数: {results['f1']:.4f}")
        
        print("\n📁 生成的文件:")
        print("- improved_zero_trust_ids/enhanced_ids_*.pth (模型文件)")
        print("- improved_zero_trust_ids/training_report.png (训练报告)")
        print("- improved_zero_trust_ids/training_results.txt (详细结果)")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 