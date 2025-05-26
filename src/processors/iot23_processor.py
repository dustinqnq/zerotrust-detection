import pandas as pd
import numpy as np
from pathlib import Path
from .base_processor import BaseDataProcessor

class IoT23Processor(BaseDataProcessor):
    """Processor for IoT-23 dataset"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        # Column names for the conn.log format
        self.columns = [
            'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
            'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
            'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
            'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
            'resp_ip_bytes', 'tunnel_parents', 'label', 'detailed-label'
        ]
        
        # Define label mappings
        self.type_a_mapping = {
            'Benign': 0,
            'C&C': 1,
            'DDoS': 2,
            'FileDownload': 3,
            'Attack': 4,
            'Malicious': 4,  # Map generic malicious to Attack
            'unknown': 4     # Map unknown to Attack
        }
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from conn.log.labeled file"""
        file_path = Path(file_path)
        print(f"\n正在读取文件: {file_path}")
        
        # Read the file, skipping comment lines and parsing Zeek log format
        data = []
        line_count = 0
        valid_count = 0
        separator = None
        actual_columns = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line_count += 1
                # Parse separator from header
                if line.startswith('#separator'):
                    separator = bytes(line.strip().split(' ')[1], 'utf-8').decode('unicode_escape')
                    continue
                # Parse column names from header
                if line.startswith('#fields'):
                    fields_line = line.strip().split(separator)[1:]  # Skip '#fields'
                    # Handle the case where the last field contains multiple column names
                    actual_columns = []
                    for field in fields_line[:-1]:
                        actual_columns.append(field)
                    # Split the last field which contains multiple columns
                    last_field_parts = fields_line[-1].split()
                    actual_columns.extend(last_field_parts)
                    print(f"发现 {len(actual_columns)} 个字段: {actual_columns}")
                    continue
                # Skip other header lines
                if line.startswith('#'):
                    continue
                    
                # Process data lines
                raw_fields = line.strip().split(separator)
                
                # Handle the case where the last field contains multiple values
                if actual_columns and len(raw_fields) < len(actual_columns):
                    # Split the last field to extract label and detailed-label
                    fields = raw_fields[:-1]  # All fields except the last
                    last_field_parts = raw_fields[-1].split(None, 2)  # Split into max 3 parts
                    fields.extend(last_field_parts)
                else:
                    fields = raw_fields
                
                # Use actual columns if available
                expected_fields = len(actual_columns) if actual_columns else len(self.columns)
                
                # Handle missing fields
                if len(fields) < expected_fields:
                    # Add empty values for missing fields
                    fields.extend([None] * (expected_fields - len(fields)))
                elif len(fields) > expected_fields:
                    # Truncate extra fields
                    fields = fields[:expected_fields]
                
                # Replace '-' with None for missing values, but keep detailed labels intact
                fields = [None if f == '-' else f for f in fields]
                data.append(fields)
                valid_count += 1
        
        print(f"\n已处理 {line_count} 行")
        print(f"找到 {valid_count} 条有效记录")
        
        if not data:
            raise ValueError("文件中未找到有效的数据记录")
        
        # Convert to DataFrame using actual columns if available
        columns_to_use = actual_columns if actual_columns else self.columns
        df = pd.DataFrame(data, columns=columns_to_use)
        print(f"\n数据框形状: {df.shape}")
        print("\n数据样例:")
        print(df.head())
        return df
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data"""
        print("\n正在预处理数据...")
        print(f"初始形状: {data.shape}")
        
        # Convert numeric columns
        numeric_cols = ['id.orig_p', 'id.resp_p', 'duration', 'orig_bytes',
                       'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes',
                       'resp_pkts', 'resp_ip_bytes']
                       
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        # Fill missing values
        data = data.fillna({
            'duration': 0,
            'orig_bytes': 0,
            'resp_bytes': 0,
            'missed_bytes': 0,
            'orig_pkts': 0,
            'orig_ip_bytes': 0,
            'resp_pkts': 0,
            'resp_ip_bytes': 0,
            'service': 'unknown',
            'proto': 'unknown',
            'conn_state': 'unknown',
            'label': 'unknown',
            'detailed-label': 'unknown'
        })
        
        # Convert timestamp
        data['ts'] = pd.to_datetime(data['ts'].astype(float), unit='s')
        
        print(f"最终形状: {data.shape}")
        return data
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from data"""
        print("\n正在提取特征...")
        features = pd.DataFrame()
        
        # Basic features
        features['duration'] = data['duration']
        features['orig_bytes'] = data['orig_bytes']
        features['resp_bytes'] = data['resp_bytes']
        features['orig_pkts'] = data['orig_pkts']
        features['resp_pkts'] = data['resp_pkts']
        
        # Derived features
        features['bytes_per_pkt'] = (data['orig_bytes'] + data['resp_bytes']) / \
                                  (data['orig_pkts'] + data['resp_pkts']).clip(lower=1)
        features['pkts_per_sec'] = (data['orig_pkts'] + data['resp_pkts']) / \
                                 data['duration'].clip(lower=0.1)
        features['bytes_per_sec'] = (data['orig_bytes'] + data['resp_bytes']) / \
                                  data['duration'].clip(lower=0.1)
        
        # Protocol one-hot encoding
        features = pd.concat([
            features,
            pd.get_dummies(data['proto'], prefix='proto')
        ], axis=1)
        
        # Service one-hot encoding
        features = pd.concat([
            features,
            pd.get_dummies(data['service'], prefix='service')
        ], axis=1)
        
        # Connection state one-hot encoding
        features = pd.concat([
            features,
            pd.get_dummies(data['conn_state'], prefix='state')
        ], axis=1)
        
        print(f"已提取特征形状: {features.shape}")
        print("\n特征列:")
        print(features.columns.tolist())
        return features
        
    def prepare_labels(self, data: pd.DataFrame) -> tuple:
        """Prepare labels for type-A and type-B classification"""
        print("\n正在准备标签...")
        
        # 使用label字段来确定基本分类（Benign/Malicious）
        def extract_ground_truth(label_str):
            if pd.isna(label_str) or label_str in ['-', '(empty)', None]:
                return 'unknown'
            return str(label_str).lower()
        
        # 提取真实标签用于最终评估
        data['ground_truth'] = data['label'].apply(extract_ground_truth)
        
        # 根据零信任原则，所有样本初始都标记为unknown
        # 这些标签将通过算法来预测和分类
        data['initial_label'] = 'unknown'
        
        # 为了训练目的，我们需要创建基于真实标签的训练标签
        # 但在实际预测时，所有数据都从unknown开始
        
        # Binary labels for training (基于真实标签)
        binary_labels = (data['ground_truth'] == 'malicious').astype(int)
        
        # Type-A labels (简化的攻击类型分类)
        self.type_a_mapping = {
            'benign': 0,
            'malicious': 1
        }
        
        # 将真实标签映射为训练标签
        data['attack_type'] = data['ground_truth'].apply(
            lambda x: 'benign' if x == 'benign' else 'malicious'
        )
        
        type_a_labels = pd.get_dummies(data['attack_type']).values
        
        # Type-B labels (使用真实标签作为子类型)
        type_b_labels = pd.get_dummies(data['ground_truth']).values
        
        # 显示真实标签分布（用于验证数据质量）
        print(f"\n真实标签分布（用于最终评估）:")
        label_counts = data['ground_truth'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  {label}: {count} 样本 ({percentage:.1f}%)")
        
        print(f"\n训练标签分布: {np.bincount(binary_labels)}")
        print(f"类型A标签形状: {type_a_labels.shape}")
        print(f"类型B标签形状: {type_b_labels.shape}")
        
        # 保存真实标签到实例变量，供后续评估使用
        self.ground_truth_labels = data['ground_truth'].values
        
        return binary_labels, type_a_labels, type_b_labels 