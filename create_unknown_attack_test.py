#!/usr/bin/env python3
"""
未知攻击检测能力测试演示脚本

这个脚本演示了如何测试三阶段零信任IDS模型发现未知攻击的能力。
使用论文中提到的Type-A和Type-B攻击模拟策略。

Type-A: 完全未知的攻击类别（从训练中完全移除）
Type-B: 已知攻击的未知变种（移除某个子类型）

注意：这是一个演示版本，展示测试方法论，不进行实际的模型训练
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import random

class UnknownAttackTestDemo:
    """未知攻击测试演示器"""
    
    def __init__(self):
        self.data_dir = Path('data')
        
    def create_sample_data(self):
        """创建示例数据集来演示测试方法"""
        print("🔄 创建示例IoT网络流量数据集...")
        
        # 模拟不同类型的攻击数据
        attack_types = {
            'Benign': 2000,
            'C&C': 500,  # Type-A测试目标
            'DDoS': 800,
            'PartOfAHorizontalPortScan': 600,  # Type-B测试目标
            'Mirai': 400,
            'FileDownload': 300
        }
        
        # 创建模拟数据
        data_samples = []
        for attack_type, count in attack_types.items():
            for i in range(count):
                # 模拟27维网络流量特征
                features = np.random.rand(27)
                
                # 根据攻击类型调整特征分布（模拟真实情况）
                if attack_type == 'C&C':
                    features[0:5] *= 0.3  # C&C通常流量较小
                    features[10:15] *= 2.0  # 但某些特征值较高
                elif attack_type == 'DDoS':
                    features[0:10] *= 3.0  # DDoS通常有大流量特征
                elif attack_type == 'PartOfAHorizontalPortScan':
                    features[15:20] *= 2.5  # 端口扫描有特定模式
                
                data_samples.append({
                    'features': features,
                    'detailed_label': attack_type,
                    'basic_label': 'benign' if attack_type == 'Benign' else 'malicious'
                })
        
        # 随机打乱数据
        random.shuffle(data_samples)
        
        # 转换为DataFrame格式
        features_df = pd.DataFrame([sample['features'] for sample in data_samples])
        labels_df = pd.DataFrame([{
            'detailed-label': sample['detailed_label'],
            'label': sample['basic_label']
        } for sample in data_samples])
        
        data = pd.concat([features_df, labels_df], axis=1)
        
        print(f"✅ 模拟数据创建完成: {len(data)} 个样本")
        print(f"   特征维度: {features_df.shape[1]}")
        
        return features_df, data
    
    def analyze_attack_distribution(self, data):
        """分析攻击类型分布"""
        print("\n📊 攻击类型分布分析")
        print("=" * 50)
        
        # 分析详细标签分布
        detailed_labels = data['detailed-label'].fillna('Benign')
        label_counts = detailed_labels.value_counts()
        
        print("详细攻击类型分布:")
        total_samples = len(data)
        for label, count in label_counts.items():
            percentage = count / total_samples * 100
            print(f"  - {label}: {count} 样本 ({percentage:.2f}%)")
            
        return label_counts
    
    def create_type_a_unknown_test(self, features, data, target_attack='C&C'):
        """创建Type-A未知攻击测试集
        
        Type-A: 完全移除某个攻击类别，模拟完全未知的攻击
        """
        print(f"\n🎯 创建Type-A未知攻击测试 (目标攻击: {target_attack})")
        print("=" * 60)
        
        # 获取详细标签
        detailed_labels = data['detailed-label'].fillna('Benign')
        
        # 识别目标攻击样本
        target_mask = detailed_labels == target_attack
        target_indices = data[target_mask].index
        
        print(f"📋 原始数据集信息:")
        print(f"   总样本数: {len(data)}")
        print(f"   {target_attack}攻击样本: {len(target_indices)}")
        print(f"   {target_attack}攻击比例: {len(target_indices)/len(data)*100:.2f}%")
        
        if len(target_indices) == 0:
            print(f"❌ 未找到{target_attack}攻击样本")
            return None
        
        # 创建训练集（移除目标攻击）
        train_indices = data.index.difference(target_indices)
        
        # 分割非目标攻击数据为训练集和验证集
        train_data, val_data = train_test_split(
            data.loc[train_indices], 
            test_size=0.2, 
            random_state=42
        )
        
        # 测试集 = 验证集的一部分 + 所有目标攻击样本
        val_benign = val_data[val_data['detailed-label'] == 'Benign']
        val_other_attacks = val_data[(val_data['detailed-label'] != 'Benign') & (val_data['detailed-label'] != target_attack)]
        
        # 组成测试集：一部分已知样本 + 所有未知攻击样本
        test_known = pd.concat([
            val_benign.sample(min(200, len(val_benign)), random_state=42) if len(val_benign) > 0 else pd.DataFrame(),
            val_other_attacks.sample(min(200, len(val_other_attacks)), random_state=42) if len(val_other_attacks) > 0 else pd.DataFrame()
        ])
        test_unknown = data.loc[target_indices]
        test_data = pd.concat([test_known, test_unknown])
        
        print(f"\n📦 Type-A测试集构建结果:")
        print(f"   训练集样本: {len(train_data)} (不包含{target_attack})")
        print(f"   测试集样本: {len(test_data)}")
        print(f"     - 已知样本: {len(test_known)}")
        print(f"     - 未知攻击样本: {len(test_unknown)} ({target_attack})")
        print(f"   未知攻击在测试集中的比例: {len(test_unknown)/len(test_data)*100:.2f}%")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'unknown_indices': test_unknown.index,
            'target_attack': target_attack,
            'type': 'Type-A'
        }
    
    def create_type_b_unknown_test(self, features, data, target_subtype='PartOfAHorizontalPortScan'):
        """创建Type-B未知攻击测试集
        
        Type-B: 移除已知攻击的某个子类型，模拟攻击变种
        """
        print(f"\n🎯 创建Type-B未知攻击测试 (目标子类型: {target_subtype})")
        print("=" * 80)
        
        # 获取详细标签
        detailed_labels = data['detailed-label'].fillna('Benign')
        
        # 识别目标子类型样本（要移除的）
        target_subtype_mask = detailed_labels == target_subtype
        target_subtype_indices = data[target_subtype_mask].index
        
        print(f"📋 原始数据集信息:")
        print(f"   总样本数: {len(data)}")
        print(f"   {target_subtype}子类型样本: {len(target_subtype_indices)}")
        
        if len(target_subtype_indices) == 0:
            print(f"❌ 未找到{target_subtype}子类型样本")
            return None
        
        # 创建训练集（不包含目标子类型）
        train_indices = data.index.difference(target_subtype_indices)
        
        # 分割数据
        train_data, val_data = train_test_split(
            data.loc[train_indices], 
            test_size=0.2, 
            random_state=42
        )
        
        # 测试集包含已知样本和未知子类型
        val_benign = val_data[val_data['detailed-label'] == 'Benign']
        val_known_attacks = val_data[val_data['detailed-label'] != 'Benign']
        
        test_known = pd.concat([
            val_benign.sample(min(200, len(val_benign)), random_state=42) if len(val_benign) > 0 else pd.DataFrame(),
            val_known_attacks.sample(min(200, len(val_known_attacks)), random_state=42) if len(val_known_attacks) > 0 else pd.DataFrame()
        ])
        test_unknown = data.loc[target_subtype_indices]
        test_data = pd.concat([test_known, test_unknown])
        
        print(f"\n📦 Type-B测试集构建结果:")
        print(f"   训练集样本: {len(train_data)} (不包含{target_subtype})")
        print(f"   测试集样本: {len(test_data)}")
        print(f"     - 已知样本: {len(test_known)}")
        print(f"     - 未知子类型样本: {len(test_unknown)} ({target_subtype})")
        print(f"   未知子类型在测试集中的比例: {len(test_unknown)/len(test_data)*100:.2f}%")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'unknown_indices': test_unknown.index,
            'target_attack': target_subtype,
            'type': 'Type-B'
        }
    
    def simulate_model_training(self, train_data):
        """模拟模型训练过程"""
        print("\n🚀 模拟零信任IDS模型训练...")
        print("   [演示] 阶段1: 浅层分类器训练...")
        print("   [演示] 阶段2: 深层分类器训练...")
        print("   [演示] 阶段3: 自编码器训练（仅使用正常样本）...")
        
        # 统计训练数据
        train_labels = train_data['detailed-label'].fillna('Benign')
        normal_samples = len(train_data[train_labels == 'Benign'])
        attack_samples = len(train_data[train_labels != 'Benign'])
        
        print(f"   训练数据统计:")
        print(f"     - 正常样本: {normal_samples}")
        print(f"     - 攻击样本: {attack_samples}")
        print(f"     - 自编码器使用: {normal_samples} 个正常样本")
        print("✅ [演示] 模型训练完成")
        
        return {
            'normal_samples_used': normal_samples,
            'attack_samples_used': attack_samples
        }
    
    def simulate_detection_results(self, test_data, unknown_indices, test_info):
        """模拟检测结果"""
        print(f"\n📊 模拟{test_info['type']}未知攻击检测")
        print("=" * 60)
        
        predictions = []
        
        # 模拟三阶段检测过程
        for idx in test_data.index:
            label = test_data.loc[idx, 'detailed-label']
            is_unknown = idx in unknown_indices
            
            # 模拟检测逻辑
            if label == 'Benign':
                # 正常流量大部分被正确识别
                if random.random() < 0.85:  # 85%正确率
                    result = {'prediction': 'benign', 'stage': 3}
                else:
                    result = {'prediction': 'unknown_attack', 'stage': 3}
            elif is_unknown:
                # 未知攻击的检测模拟
                rand = random.random()
                if rand < 0.65:  # 65%被阶段3检测为未知
                    result = {'prediction': 'unknown_attack', 'stage': 3}
                elif rand < 0.85:  # 20%被误认为已知攻击
                    result = {'prediction': 'known_attack_stage2', 'stage': 2}
                else:  # 15%被误认为正常
                    result = {'prediction': 'benign', 'stage': 3}
            else:
                # 已知攻击的检测模拟
                rand = random.random()
                if rand < 0.4:  # 40%在阶段1被检测
                    result = {'prediction': 'known_attack_stage1', 'stage': 1}
                elif rand < 0.8:  # 40%在阶段2被检测
                    result = {'prediction': 'known_attack_stage2', 'stage': 2}
                else:  # 20%漏检或被认为是未知
                    if random.random() < 0.5:
                        result = {'prediction': 'unknown_attack', 'stage': 3}
                    else:
                        result = {'prediction': 'benign', 'stage': 3}
            
            predictions.append(result)
        
        return predictions
    
    def evaluate_unknown_detection(self, test_data, unknown_indices, predictions, test_info):
        """评估未知攻击检测能力"""
        print(f"\n🔍 检测结果分析:")
        
        # 统计各阶段检测结果
        stage_stats = {'stage1': 0, 'stage2': 0, 'stage3': 0}
        detection_stats = {
            'known_attack_stage1': 0,
            'known_attack_stage2': 0, 
            'unknown_attack': 0,
            'benign': 0
        }
        
        for pred in predictions:
            stage_stats[f"stage{pred['stage']}"] += 1
            detection_stats[pred['prediction']] += 1
        
        print(f"\n🎯 各阶段检测统计:")
        for stage, count in stage_stats.items():
            print(f"   {stage}: {count} ({count/len(predictions)*100:.1f}%)")
        
        print(f"\n🔍 检测类型统计:")
        for det_type, count in detection_stats.items():
            print(f"   {det_type}: {count} ({count/len(predictions)*100:.1f}%)")
        
        # 未知攻击检测分析
        print(f"\n🎯 未知攻击检测分析 ({test_info['target_attack']}):")
        
        unknown_mask = [idx in unknown_indices for idx in test_data.index]
        unknown_predictions = [pred['prediction'] for i, pred in enumerate(predictions) if unknown_mask[i]]
        unknown_stages = [pred['stage'] for i, pred in enumerate(predictions) if unknown_mask[i]]
        
        total_unknown = len(unknown_predictions)
        correctly_detected_as_unknown = sum(1 for pred in unknown_predictions if pred == 'unknown_attack')
        detected_as_known = sum(1 for pred in unknown_predictions if 'known_attack' in pred)
        detected_as_benign = sum(1 for pred in unknown_predictions if pred == 'benign')
        
        print(f"   未知攻击样本总数: {total_unknown}")
        print(f"   未知攻击检测结果:")
        print(f"     - 正确识别为未知攻击: {correctly_detected_as_unknown} ({correctly_detected_as_unknown/total_unknown*100:.1f}%)")
        print(f"     - 误识别为已知攻击: {detected_as_known} ({detected_as_known/total_unknown*100:.1f}%)")
        print(f"     - 误识别为正常流量: {detected_as_benign} ({detected_as_benign/total_unknown*100:.1f}%)")
        
        # 计算检测指标
        unknown_recall = correctly_detected_as_unknown / total_unknown if total_unknown > 0 else 0
        total_unknown_predictions = sum(1 for pred in predictions if pred['prediction'] == 'unknown_attack')
        unknown_precision = correctly_detected_as_unknown / total_unknown_predictions if total_unknown_predictions > 0 else 0
        unknown_f1 = 2 * (unknown_precision * unknown_recall) / (unknown_precision + unknown_recall) if (unknown_precision + unknown_recall) > 0 else 0
        
        print(f"\n📈 未知攻击检测性能指标:")
        print(f"   精确度 (Precision): {unknown_precision:.3f}")
        print(f"   召回率 (Recall): {unknown_recall:.3f}")
        print(f"   F1分数: {unknown_f1:.3f}")
        
        # 阶段分析
        print(f"\n🔄 未知攻击检测阶段分析:")
        unknown_stage_stats = {}
        for stage in unknown_stages:
            unknown_stage_stats[stage] = unknown_stage_stats.get(stage, 0) + 1
        
        for stage, count in unknown_stage_stats.items():
            print(f"   阶段{stage}: {count} ({count/total_unknown*100:.1f}%)")
        
        return {
            'type': test_info['type'],
            'target_attack': test_info['target_attack'],
            'total_unknown': total_unknown,
            'correctly_detected': correctly_detected_as_unknown,
            'precision': unknown_precision,
            'recall': unknown_recall,
            'f1_score': unknown_f1,
            'stage_distribution': unknown_stage_stats
        }
    
    def run_unknown_attack_test(self):
        """运行完整的未知攻击测试演示"""
        print("🚀 开始未知攻击检测能力测试演示")
        print("=" * 80)
        
        # 创建示例数据
        features, data = self.create_sample_data()
        
        # 分析攻击分布
        attack_distribution = self.analyze_attack_distribution(data)
        
        results = []
        
        # 测试Type-A未知攻击（以C&C攻击为例）
        print("\n" + "="*80)
        print("🔴 测试Type-A未知攻击检测能力")
        print("="*80)
        
        type_a_test = self.create_type_a_unknown_test(features, data, target_attack='C&C')
        if type_a_test:
            # 模拟训练
            train_info = self.simulate_model_training(type_a_test['train_data'])
            
            # 模拟检测
            predictions_a = self.simulate_detection_results(
                type_a_test['test_data'], 
                type_a_test['unknown_indices'], 
                type_a_test
            )
            
            # 评估结果
            result_a = self.evaluate_unknown_detection(
                type_a_test['test_data'], 
                type_a_test['unknown_indices'], 
                predictions_a, 
                type_a_test
            )
            results.append(result_a)
        
        # 测试Type-B未知攻击
        print("\n" + "="*80)
        print("🟡 测试Type-B未知攻击检测能力")
        print("="*80)
        
        type_b_test = self.create_type_b_unknown_test(features, data, target_subtype='PartOfAHorizontalPortScan')
        if type_b_test:
            # 模拟训练
            train_info = self.simulate_model_training(type_b_test['train_data'])
            
            # 模拟检测
            predictions_b = self.simulate_detection_results(
                type_b_test['test_data'], 
                type_b_test['unknown_indices'], 
                type_b_test
            )
            
            # 评估结果
            result_b = self.evaluate_unknown_detection(
                type_b_test['test_data'], 
                type_b_test['unknown_indices'], 
                predictions_b, 
                type_b_test
            )
            results.append(result_b)
        
        # 总结结果
        self.summarize_results(results)
        
        return results
    
    def summarize_results(self, results):
        """总结测试结果"""
        print("\n" + "="*80)
        print("📋 未知攻击检测能力测试总结")
        print("="*80)
        
        for result in results:
            print(f"\n🎯 {result['type']} 未知攻击测试 ({result['target_attack']}):")
            print(f"   未知攻击样本总数: {result['total_unknown']}")
            print(f"   正确检测数量: {result['correctly_detected']}")
            print(f"   检测精确度: {result['precision']:.3f}")
            print(f"   检测召回率: {result['recall']:.3f}")
            print(f"   F1分数: {result['f1_score']:.3f}")
            
            print(f"   检测阶段分布:")
            for stage, count in result['stage_distribution'].items():
                print(f"     - 阶段{stage}: {count} 样本")
        
        if results:
            avg_precision = np.mean([r['precision'] for r in results])
            avg_recall = np.mean([r['recall'] for r in results])
            avg_f1 = np.mean([r['f1_score'] for r in results])
            
            print(f"\n📊 整体性能:")
            print(f"   平均精确度: {avg_precision:.3f}")
            print(f"   平均召回率: {avg_recall:.3f}")
            print(f"   平均F1分数: {avg_f1:.3f}")
        
        print(f"\n💡 测试方法说明:")
        print(f"   - Type-A: 完全移除某攻击类别，测试模型对全新攻击的检测能力")
        print(f"   - Type-B: 移除已知攻击的子类型，测试模型对攻击变种的检测能力")
        print(f"   - 阶段3自编码器负责检测未知模式，重构误差高的样本被标记为未知攻击")
        print(f"\n🔬 核心测试原理:")
        print(f"   1. 训练时：从训练集中完全移除特定攻击类型/子类型")
        print(f"   2. 测试时：重新引入这些'未知'攻击，评估检测能力")
        print(f"   3. 成功检测率衡量模型发现真正未知威胁的能力")

def main():
    """主函数"""
    print("🎯 零信任IDS未知攻击检测能力测试演示")
    print("基于Type-A和Type-B攻击模拟策略")
    print("="*80)
    
    demo = UnknownAttackTestDemo()
    results = demo.run_unknown_attack_test()
    
    print("\n🎉 演示完成！")
    print("\n📚 这个演示展示了未知攻击检测能力测试的核心方法：")
    print("   1. 数据集分割策略：将某些攻击类型完全从训练集中移除")
    print("   2. 模拟真实场景：模型在训练时从未见过这些攻击")
    print("   3. 评估检测能力：测试时重新引入，看模型能否识别为'未知威胁'")
    print("   4. 性能指标计算：精确度、召回率、F1分数等")
    print("\n🔍 实际项目中，阶段3的自编码器是关键：")
    print("   - 只用正常流量训练，学习'正常行为模式'")
    print("   - 任何偏离正常模式的都可能是未知攻击")
    print("   - 通过重构误差阈值来判断异常程度")

if __name__ == "__main__":
    main() 