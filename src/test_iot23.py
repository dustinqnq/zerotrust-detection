import os
import gc
import pandas as pd
from datetime import datetime
from pathlib import Path
from processors.iot23_processor import IoT23Processor
from zero_trust_ids import ZeroTrustIDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf

def main():
    # Configure TensorFlow for memory optimization
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Set TensorFlow to use less CPU memory
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    # Initialize processor
    data_dir = Path('data')
    processor = IoT23Processor(data_dir)
    
    # 使用原始的单一数据集处理
    file_path = data_dir / 'iot-23/CTU-IoT-Malware-Capture-3-1/bro/conn.log.labeled'
    print(f"\n正在处理 {file_path}...")
    
    # 处理数据
    features, binary_labels, type_a_labels, type_b_labels = processor.process(file_path)
    
    print(f"原始数据集: {len(features)}个样本")
    print(f"原始攻击分布: {np.bincount(binary_labels)}")
    
    # 创建平衡的小数据集
    benign_indices = np.where(binary_labels == 0)[0]
    malicious_indices = np.where(binary_labels == 1)[0]
    
    print(f"良性样本数量: {len(benign_indices)}")
    print(f"恶意样本数量: {len(malicious_indices)}")
    
    # 设置每类样本数量（创建平衡数据集）
    samples_per_class = 500  # 每类500个样本，总共1000个样本
    
    # 随机采样良性样本
    if len(benign_indices) >= samples_per_class:
        selected_benign = np.random.choice(benign_indices, samples_per_class, replace=False)
    else:
        # 如果良性样本不够，就全部使用
        selected_benign = benign_indices
        print(f"警告: 良性样本不足{samples_per_class}个，使用全部{len(benign_indices)}个")
    
    # 随机采样恶意样本
    selected_malicious = np.random.choice(malicious_indices, samples_per_class, replace=False)
    
    # 合并选中的样本索引
    selected_indices = np.concatenate([selected_benign, selected_malicious])
    
    # 打乱顺序
    np.random.shuffle(selected_indices)
    
    # 创建平衡数据集
    features = features.iloc[selected_indices]
    binary_labels = binary_labels[selected_indices]
    type_a_labels = type_a_labels[selected_indices]
    type_b_labels = type_b_labels[selected_indices]
    
    print(f"\n平衡后数据集: {len(features)}个样本")
    print(f"特征维度: {features.shape}")
    print(f"平衡后攻击分布: {np.bincount(binary_labels)}")
    print(f"良性样本比例: {np.mean(binary_labels == 0)*100:.1f}%")
    print(f"恶意样本比例: {np.mean(binary_labels == 1)*100:.1f}%")
    
    # 转换为float32以减少内存使用
    features = features.astype(np.float32)
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 从内存中清除原始特征
    del features
    gc.collect()
    
    # 使用分层采样保持训练集和测试集的平衡
    X_train, X_test, y_train, y_test, \
    y_type_a_train, y_type_a_test, \
    y_type_b_train, y_type_b_test = train_test_split(
        features_scaled, binary_labels, type_a_labels, type_b_labels,
        test_size=0.2, random_state=42, stratify=binary_labels  # 使用分层采样
    )
    
    print(f"\n数据分割结果:")
    print(f"训练集: {len(X_train)}个样本")
    print(f"  良性: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
    print(f"  恶意: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
    print(f"测试集: {len(X_test)}个样本")
    print(f"  良性: {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)")
    print(f"  恶意: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)")
    
    # 从内存中清除标准化特征
    del features_scaled
    gc.collect()
    
    # 初始化并训练模型
    print("\n正在训练模型...")
    model = ZeroTrustIDS(input_dim=X_train.shape[1])
    model.fit(
        X_train, y_train,
        y_type_a_train, y_type_b_train,
        epochs=2,  # 快速测试使用2个周期
        batch_size=128,  # 从64增加到128
        validation_split=0.1  # 从0.15减少到0.1
    )
    
    # 进行预测（包含DBSCAN）
    print("\n正在进行预测...")
    type_a_binary, type_a_multi, type_b_binary, type_b_multi, dbscan_labels = model.predict(X_test)
    
    # 评估结果
    print("\n评估结果:")
    
    # 将pandas Series转换为numpy数组以进行比较
    y_test_array = np.array(y_test)
    y_type_a_test_array = np.array(y_type_a_test) if y_type_a_test is not None else None
    y_type_b_test_array = np.array(y_type_b_test) if y_type_b_test is not None else None
    
    print("类型A二分类准确率:", np.mean((type_a_binary.flatten() > 0.5).astype(int) == y_test_array))
    
    if type_a_multi is not None and y_type_a_test_array is not None:
        print("类型A多分类准确率:", np.mean(np.argmax(type_a_multi, axis=1) == np.argmax(y_type_a_test_array, axis=1)))
    else:
        print("类型A多分类: 未训练")
    
    print("类型B二分类准确率:", np.mean((type_b_binary.flatten() > 0.5).astype(int) == y_test_array))
    
    if type_b_multi is not None and y_type_b_test_array is not None:
        print("类型B多分类准确率:", np.mean(np.argmax(type_b_multi, axis=1) == np.argmax(y_type_b_test_array, axis=1)))
    else:
        print("类型B多分类: 未训练")
        
    print("DBSCAN聚类数:", len(np.unique(dbscan_labels)), "个唯一聚类")
    print("DBSCAN噪声点:", np.sum(dbscan_labels == -1), f"({np.mean(dbscan_labels == -1)*100:.1f}%)")
    
    # 详细数据分布分析
    print("\n详细数据分布分析:")
    print(f"总样本数: {len(y_test)}")
    print(f"正常样本: {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)")
    print(f"恶意样本: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)")
    
    # 类型A分类分析
    if y_type_a_test_array is not None:
        type_a_predicted = np.argmax(type_a_multi, axis=1) if type_a_multi is not None else None
        type_a_actual = np.argmax(y_type_a_test_array, axis=1)
        
        print(f"\n类型A攻击类型分布:")
        unique_attacks, counts = np.unique(type_a_actual, return_counts=True)
        for attack_type, count in zip(unique_attacks, counts):
            print(f"  攻击类型 {attack_type}: {count} 样本 ({count/len(type_a_actual)*100:.1f}%)")
    
    # 类型B分类分析  
    if y_type_b_test_array is not None:
        type_b_predicted = np.argmax(type_b_multi, axis=1) if type_b_multi is not None else None
        type_b_actual = np.argmax(y_type_b_test_array, axis=1)
        
        print(f"\n类型B攻击子类型分布:")
        unique_subtypes, counts = np.unique(type_b_actual, return_counts=True)
        for subtype, count in zip(unique_subtypes[:10], counts[:10]):  # 显示前10个
            print(f"  子类型 {subtype}: {count} 样本 ({count/len(type_b_actual)*100:.1f}%)")
    
    # DBSCAN聚类分析
    unique_clusters = np.unique(dbscan_labels)
    print(f"\nDBSCAN聚类分析:")
    print(f"发现的总聚类数: {len(unique_clusters)}")
    for cluster_id in unique_clusters:
        cluster_size = np.sum(dbscan_labels == cluster_id)
        if cluster_id == -1:
            print(f"  噪声点: {cluster_size} ({cluster_size/len(dbscan_labels)*100:.1f}%)")
        else:
            print(f"  聚类 {cluster_id}: {cluster_size} 点 ({cluster_size/len(dbscan_labels)*100:.1f}%)")
    
    # 整体性能总结
    print(f"\n{'='*50}")
    print(f"多攻击类型IoT检测系统性能")
    print(f"{'='*50}")
    print(f"数据集多样性: 处理了多种攻击类型")
    print(f"总训练样本数: {len(X_train)}")
    print(f"总测试样本数: {len(X_test)}")
    print(f"特征维度: {X_train.shape[1]}")
    
    benign_ratio = np.mean(y_test == 0) * 100
    malicious_ratio = np.mean(y_test == 1) * 100
    
    if benign_ratio > 98 or malicious_ratio > 98:
        print(f"⚠️  警告: 检测到高度不平衡的数据集!")
        print(f"   这可能导致准确率分数虚高。")
    else:
        print(f"✅ 具有多样攻击类型的平衡数据集")
    
    # 生成测试结果CSV
    print(f"\n正在生成测试结果...")
    os.makedirs('results', exist_ok=True)
    
    # 收集测试结果数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 计算性能指标
    type_a_binary_acc = np.mean((type_a_binary.flatten() > 0.5).astype(int) == y_test_array)
    type_a_multi_acc = np.mean(np.argmax(type_a_multi, axis=1) == np.argmax(y_type_a_test_array, axis=1)) if type_a_multi is not None and y_type_a_test_array is not None else None
    type_b_binary_acc = np.mean((type_b_binary.flatten() > 0.5).astype(int) == y_test_array)
    type_b_multi_acc = np.mean(np.argmax(type_b_multi, axis=1) == np.argmax(y_type_b_test_array, axis=1)) if type_b_multi is not None and y_type_b_test_array is not None else None
    
    # 聚类分析
    num_clusters = len(np.unique(dbscan_labels))
    noise_ratio = np.mean(dbscan_labels == -1) * 100
    
    # 创建结果记录
    result_data = {
        '测试时间': [timestamp],
        '数据集名称': ['CTU-IoT-Malware-Capture-3-1'],
        '数据集路径': ['iot-23/CTU-IoT-Malware-Capture-3-1/bro/conn.log.labeled'],
        '原始总样本数': [156103],
        '原始良性样本数': [4536],
        '原始恶意样本数': [151567],
        '原始良性比例(%)': [4536/156103*100],
        '原始恶意比例(%)': [151567/156103*100],
        '平衡后总样本数': [len(selected_indices)],
        '平衡后良性样本数': [samples_per_class if len(benign_indices) >= samples_per_class else len(benign_indices)],
        '平衡后恶意样本数': [samples_per_class],
        '平衡后良性比例(%)': [np.mean(binary_labels == 0)*100],
        '平衡后恶意比例(%)': [np.mean(binary_labels == 1)*100],
        '训练集样本数': [len(X_train)],
        '测试集样本数': [len(X_test)],
        '特征维度': [X_train.shape[1]],
        '训练轮数': [2],
        '批次大小': [128],
        '类型A二分类准确率': [type_a_binary_acc],
        '类型A多分类准确率': [type_a_multi_acc],
        '类型B二分类准确率': [type_b_binary_acc],
        '类型B多分类准确率': [type_b_multi_acc],
        'DBSCAN聚类数': [num_clusters],
        'DBSCAN噪声点比例(%)': [noise_ratio],
        '数据集平衡状态': ['平衡' if abs(benign_ratio - malicious_ratio) < 5 else '不平衡'],
        '模型保存路径': ['models/iot23_multi_attack_model'],
        '备注': ['使用平衡采样策略，零信任三阶段架构']
    }
    
    # 创建DataFrame并保存为CSV
    results_df = pd.DataFrame(result_data)
    csv_filename = f'results/test_result_{timestamp}.csv'
    results_df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"测试结果已保存到: {csv_filename}")
    
    # 如果存在汇总文件，则追加记录，否则创建新文件
    summary_file = 'results/all_test_results.csv'
    if os.path.exists(summary_file):
        # 读取现有汇总文件并追加新记录
        existing_df = pd.read_csv(summary_file, encoding='utf-8')
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        updated_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"结果已追加到汇总文件: {summary_file}")
    else:
        # 创建新的汇总文件
        results_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"创建新的汇总文件: {summary_file}")
    
    # 保存模型
    print(f"\n正在保存模型...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/iot23_multi_attack_model')
    print("模型已保存到 models/iot23_multi_attack_model")

if __name__ == '__main__':
    main() 