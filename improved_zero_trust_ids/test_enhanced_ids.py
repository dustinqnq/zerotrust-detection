#!/usr/bin/env python3
"""
增强零信任IDS测试脚本
加载训练好的模型并进行推理测试
"""

import sys
import os
sys.path.append('../src')

import numpy as np
import pandas as pd
import torch
from enhanced_autoencoder_ids import EnhancedZeroTrustIDS
from processors.iot23_processor import IoT23Processor
import time
import json

def load_test_data():
    """加载测试数据"""
    print("=== 加载测试数据 ===")
    
    processor = IoT23Processor()
    data_path = "../data/iot23/CTU-IoT-Malware-Capture-3-1"
    
    try:
        df = processor.load_data(data_path)
        df_processed = processor.preprocess(df)
        features = processor.extract_features(df_processed)
        labels = processor.prepare_labels(df_processed)
        
        # 取一小部分数据进行快速测试
        test_size = min(1000, len(features))
        features_test = features[:test_size]
        labels_test = labels[:test_size]
        
        print(f"测试数据形状: {features_test.shape}")
        return features_test, labels_test, df_processed[:test_size]
        
    except Exception as e:
        print(f"测试数据加载失败: {e}")
        return None, None, None

def test_model_performance():
    """测试模型性能"""
    print("🧪 增强零信任IDS模型测试")
    print("=" * 50)
    
    # 1. 加载测试数据
    features, labels, df = load_test_data()
    if features is None:
        print("❌ 无法加载测试数据")
        return
    
    # 2. 初始化模型
    print("=== 初始化模型 ===")
    input_dim = features.shape[1]
    model = EnhancedZeroTrustIDS(input_dim=input_dim)
    
    # 3. 加载训练好的模型
    print("=== 加载训练好的模型 ===")
    try:
        model.load_models()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请先运行训练脚本 train_enhanced_ids.py")
        return
    
    # 4. 数据预处理
    print("=== 数据预处理 ===")
    features_scaled = model.scaler.transform(features)
    
    # 5. 性能测试
    print("=== 性能测试 ===")
    
    # 单样本推理时间测试
    print("📊 单样本推理时间测试:")
    sample = features_scaled[0:1]  # 取一个样本
    
    # 预热
    for _ in range(10):
        _ = model.predict(sample)
    
    # 正式测试
    start_time = time.time()
    for _ in range(100):
        results = model.predict(sample)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
    print(f"平均单样本推理时间: {avg_time:.2f} ms")
    
    # 批量推理测试
    print("\n📊 批量推理测试:")
    batch_sizes = [10, 50, 100, 500]
    
    for batch_size in batch_sizes:
        if batch_size <= len(features_scaled):
            batch_data = features_scaled[:batch_size]
            
            start_time = time.time()
            batch_results = model.predict(batch_data)
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000  # ms
            per_sample_time = total_time / batch_size
            throughput = 1000 / per_sample_time  # samples/second
            
            print(f"批量大小 {batch_size}: 总时间 {total_time:.2f}ms, "
                  f"每样本 {per_sample_time:.2f}ms, 吞吐量 {throughput:.1f} samples/s")
    
    # 6. 检测能力测试
    print("\n=== 检测能力测试 ===")
    test_samples = min(200, len(features_scaled))
    test_features = features_scaled[:test_samples]
    test_labels = labels[:test_samples]
    
    predictions = model.predict(test_features)
    
    # 统计各阶段检测结果
    stage_stats = {'stage1': 0, 'stage2': 0, 'stage3': 0}
    detection_stats = {
        'known_attack_stage1': 0,
        'known_attack_stage2': 0, 
        'unknown_attack': 0,
        'benign': 0
    }
    
    confidence_scores = []
    
    for pred in predictions:
        stage_stats[f"stage{pred['stage']}"] += 1
        detection_stats[pred['prediction']] += 1
        confidence_scores.append(pred['confidence'])
    
    print("各阶段检测统计:")
    for stage, count in stage_stats.items():
        print(f"{stage}: {count} ({count/len(predictions)*100:.1f}%)")
    
    print("\n检测类型统计:")
    for det_type, count in detection_stats.items():
        print(f"{det_type}: {count} ({count/len(predictions)*100:.1f}%)")
    
    print(f"\n平均置信度: {np.mean(confidence_scores):.3f}")
    print(f"置信度标准差: {np.std(confidence_scores):.3f}")
    
    # 7. 详细分析几个样本
    print("\n=== 样本详细分析 ===")
    analyze_samples = min(5, len(predictions))
    
    for i in range(analyze_samples):
        pred = predictions[i]
        actual_label = test_labels[i]
        
        print(f"\n样本 {i+1}:")
        print(f"  实际标签: {actual_label}")
        print(f"  预测结果: {pred['prediction']}")
        print(f"  检测阶段: Stage {pred['stage']}")
        print(f"  置信度: {pred['confidence']:.3f}")
        
        if 'reconstruction_error' in pred:
            print(f"  重构误差: {pred['reconstruction_error']:.6f}")
            print(f"  异常阈值: {model.anomaly_threshold:.6f}")
    
    # 8. 保存测试结果
    print("\n=== 保存测试结果 ===")
    
    test_results = {
        'performance': {
            'avg_inference_time_ms': avg_time,
            'avg_confidence': float(np.mean(confidence_scores)),
            'confidence_std': float(np.std(confidence_scores))
        },
        'stage_distribution': stage_stats,
        'detection_distribution': detection_stats,
        'sample_predictions': predictions[:10]  # 保存前10个预测结果
    }
    
    with open('improved_zero_trust_ids/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print("✅ 测试结果已保存到: improved_zero_trust_ids/test_results.json")
    
    return test_results

def real_time_monitoring_demo():
    """实时监控演示"""
    print("\n🔴 实时监控演示")
    print("=" * 30)
    
    # 加载模型
    features, _, _ = load_test_data()
    if features is None:
        return
    
    input_dim = features.shape[1]
    model = EnhancedZeroTrustIDS(input_dim=input_dim)
    
    try:
        model.load_models()
    except:
        print("❌ 模型未加载，请先训练模型")
        return
    
    features_scaled = model.scaler.transform(features)
    
    print("开始模拟实时监控... (按Ctrl+C停止)")
    print("显示格式: [时间] 样本ID | 检测结果 | 阶段 | 置信度")
    print("-" * 80)
    
    try:
        for i in range(min(50, len(features_scaled))):  # 模拟50个样本
            sample = features_scaled[i:i+1]
            
            start_time = time.time()
            result = model.predict(sample)[0]
            end_time = time.time()
            
            # 格式化输出
            timestamp = time.strftime("%H:%M:%S")
            sample_id = f"Sample_{i+1:03d}"
            prediction = result['prediction']
            stage = f"Stage_{result['stage']}"
            confidence = f"{result['confidence']:.3f}"
            inference_time = f"{(end_time-start_time)*1000:.1f}ms"
            
            # 根据检测结果使用不同颜色（简单的标记）
            if prediction == 'benign':
                status = "✅ BENIGN    "
            elif 'known_attack' in prediction:
                status = "⚠️  KNOWN_ATK "
            else:
                status = "🚨 UNKNOWN_ATK"
            
            print(f"[{timestamp}] {sample_id} | {status} | {stage} | {confidence} | {inference_time}")
            
            # 模拟实时延迟
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  监控演示已停止")

def main():
    """主函数"""
    try:
        # 运行性能测试
        test_results = test_model_performance()
        
        if test_results:
            print("\n" + "="*50)
            print("📋 测试总结:")
            print(f"平均推理时间: {test_results['performance']['avg_inference_time_ms']:.2f} ms")
            print(f"平均置信度: {test_results['performance']['avg_confidence']:.3f}")
            
            # 询问是否运行实时监控演示
            print("\n是否运行实时监控演示? (y/n): ", end="")
            choice = input().lower().strip()
            
            if choice in ['y', 'yes']:
                real_time_monitoring_demo()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 