import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import joblib
import warnings
import pickle
from pathlib import Path
import torch.nn.functional as F

warnings.filterwarnings('ignore')

class DeepAutoencoder(nn.Module):
    """Deep autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, encoding_dims: List[int] = [256, 128, 64]):
        super(DeepAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.4)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        
        # Decoder with skip connections
        self.decoder_layers = nn.ModuleList()
        decoding_dims = encoding_dims[::-1][1:] + [input_dim]
        
        # Add projection layers for residual connections
        self.projections = nn.ModuleList()
        
        prev_dim = encoding_dims[-1]  # Start from bottleneck dimension
        for i, dim in enumerate(decoding_dims):
            # Add decoder layer
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.1) if i < len(decoding_dims)-1 else nn.Tanh(),
                nn.Dropout(0.4) if i < len(decoding_dims)-1 else nn.Identity()
            ))
            # Add projection layer for residual connection
            self.projections.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder with residual connections
        decoded = encoded
        for i, (layer, proj) in enumerate(zip(self.decoder_layers[:-1], self.projections[:-1])):
            # Project the input to match the output dimension
            residual = proj(decoded)
            # Apply decoder layer and add residual
            decoded = layer(decoded) + residual
        
        # Final layer without residual
        decoded = self.decoder_layers[-1](decoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class ShallowClassifier(nn.Module):
    """Shallow classifier - Stage 1"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super(ShallowClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class DeepClassifier(nn.Module):
    """Deep classifier - Stage 2"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super(DeepClassifier, self).__init__()
        
        # 增强特征提取层
        self.features = nn.Sequential(
            # 第一层特征提取
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),  # 降低dropout以保留更多特征
            
            # 第二层特征提取
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            # 第三层特征提取
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
        # 投影层用于残差连接
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        
        # 注意力机制
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # 残差连接
        residual = self.projection(x)
        combined = features + residual
        
        # 分类
        return self.classifier(combined)

class EnhancedZeroTrustIDS:
    """增强的零信任入侵检测系统"""
    
    def __init__(self, input_dim: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.input_dim = input_dim
        self.scaler = StandardScaler()
        
        # 阶段1: 浅层分类器
        self.stage1_binary = ShallowClassifier(input_dim, 2).to(device)
        self.stage1_multi = ShallowClassifier(input_dim, 10).to(device)  # 假设最多10个攻击类别
        
        # 阶段2: 深层分类器  
        self.stage2_binary = DeepClassifier(input_dim, 2).to(device)
        self.stage2_multi = DeepClassifier(input_dim, 20).to(device)  # 假设最多20个子类别
        
        # 阶段3: 自编码器
        self.autoencoder = DeepAutoencoder(input_dim).to(device)
        
        # 优化器
        self.optimizers = {
            'stage1_binary': optim.AdamW(self.stage1_binary.parameters(), lr=0.002, weight_decay=0.01, betas=(0.9, 0.999)),
            'stage1_multi': optim.AdamW(self.stage1_multi.parameters(), lr=0.002, weight_decay=0.01, betas=(0.9, 0.999)),
            'stage2_binary': optim.AdamW(self.stage2_binary.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)),
            'stage2_multi': optim.AdamW(self.stage2_multi.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)),
            'autoencoder': optim.AdamW(self.autoencoder.parameters(), lr=0.0005, weight_decay=0.01, betas=(0.9, 0.999))
        }
        
        # 学习率调度器
        self.schedulers = {
            'stage1_binary': optim.lr_scheduler.ReduceLROnPlateau(self.optimizers['stage1_binary'], mode='min', factor=0.1, patience=5, verbose=True),
            'stage1_multi': optim.lr_scheduler.ReduceLROnPlateau(self.optimizers['stage1_multi'], mode='min', factor=0.1, patience=5, verbose=True),
            'stage2_binary': optim.lr_scheduler.ReduceLROnPlateau(self.optimizers['stage2_binary'], mode='min', factor=0.1, patience=5, verbose=True),
            'stage2_multi': optim.lr_scheduler.ReduceLROnPlateau(self.optimizers['stage2_multi'], mode='min', factor=0.1, patience=5, verbose=True),
            'autoencoder': optim.lr_scheduler.ReduceLROnPlateau(self.optimizers['autoencoder'], mode='min', factor=0.1, patience=5, verbose=True)
        }
        
        # 损失函数
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion_ae = nn.MSELoss()
        
        # 阈值参数
        self.anomaly_threshold = None
        
        # 标签编码器
        self.label_encoders = {
            'binary': LabelEncoder(),
            'multi_stage1': LabelEncoder(),
            'multi_stage2': LabelEncoder()
        }
        
        print(f"模型初始化完成，使用设备: {self.device}")
    
    def prepare_data(self, X: np.ndarray, y_binary: np.ndarray, y_multi: np.ndarray, y_sub: np.ndarray):
        """数据预处理"""
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 标签编码
        y_binary_encoded = self.label_encoders['binary'].fit_transform(y_binary)
        y_multi_encoded = self.label_encoders['multi_stage1'].fit_transform(y_multi)
        y_sub_encoded = self.label_encoders['multi_stage2'].fit_transform(y_sub)
        
        return X_scaled, y_binary_encoded, y_multi_encoded, y_sub_encoded
    
    def train_stage1(self, X_train, y_bin_train, y_multi_train, X_val, y_bin_val, y_multi_val, epochs=10):
        """训练阶段1的分类器"""
        print("开始训练阶段1分类器...")
        
        # 创建保存目录
        save_dir = Path('./models')
        save_dir.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练二分类器
            self.stage1_binary.train()
            binary_loss = self._train_binary_classifier(X_train, y_bin_train)
            
            # 训练多分类器
            self.stage1_multi.train()
            multi_loss = self._train_multi_classifier(X_train, y_multi_train)
            
            # 验证
            val_loss = self._validate_stage1(X_val, y_bin_val, y_multi_val)
            
            print(f"Epoch {epoch}: Binary Loss: {binary_loss:.4f}, Multi Loss: {multi_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.stage1_binary.state_dict(), save_dir / 'stage1_binary_best.pth')
                torch.save(self.stage1_multi.state_dict(), save_dir / 'stage1_multi_best.pth')
                print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
    
    def train_stage2(self, X_train, y_bin_train, y_multi_train, X_val, y_bin_val, y_multi_val, epochs=30):
        """训练阶段2的分类器"""
        print("开始训练阶段2分类器...")
        
        # 创建保存目录
        save_dir = Path('./models')
        save_dir.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        patience = 15  # 增加早停的耐心值
        no_improve = 0
        
        # 重置优化器，使用更大的学习率
        self.optimizers['stage2_binary'] = optim.AdamW(
            self.stage2_binary.parameters(),
            lr=0.001,  # 降低学习率以获得更稳定的训练
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        self.optimizers['stage2_multi'] = optim.AdamW(
            self.stage2_multi.parameters(),
            lr=0.001,  # 降低学习率以获得更稳定的训练
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 使用StepLR调度器，更稳定的学习率衰减
        self.schedulers['stage2_binary'] = optim.lr_scheduler.StepLR(
            self.optimizers['stage2_binary'],
            step_size=10,
            gamma=0.5
        )
        self.schedulers['stage2_multi'] = optim.lr_scheduler.StepLR(
            self.optimizers['stage2_multi'],
            step_size=10,
            gamma=0.5
        )
        
        # 对多分类任务使用focal loss来处理类别不平衡
        num_classes = len(np.unique(y_multi_train))
        focal_loss = FocalLoss(alpha=None, gamma=2.0)  # 简化，不使用alpha权重
        
        for epoch in range(epochs):
            # 训练二分类器
            self.stage2_binary.train()
            binary_loss = self._train_stage2_binary_classifier(X_train, y_bin_train)
            
            # 训练多分类器，使用focal loss和增加权重
            self.stage2_multi.train()
            multi_loss = self._train_stage2_multi_classifier(X_train, y_multi_train, focal_loss) * 3.0  # 进一步增加权重
            
            # 验证
            val_loss = self._validate_stage2(X_val, y_bin_val, y_multi_val)
            
            # 更新学习率
            self.schedulers['stage2_binary'].step()
            self.schedulers['stage2_multi'].step()
            
            # 打印当前学习率
            current_lr = self.schedulers['stage2_binary'].get_last_lr()[0]
            print(f"轮次 {epoch}: 二分类损失: {binary_loss:.4f}, 多分类损失: {multi_loss:.4f}, 验证损失: {val_loss:.4f}, 学习率: {current_lr:.6f}")
            
            # 改进的模型保存策略
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.stage2_binary.state_dict(), save_dir / 'stage2_binary_best.pth')
                torch.save(self.stage2_multi.state_dict(), save_dir / 'stage2_multi_best.pth')
                print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping，但更宽松
            if no_improve >= patience:
                print(f"提前停止训练，{patience} 轮后没有改善")
                break
        
        # 训练结束后，加载最佳模型
        if (save_dir / 'stage2_binary_best.pth').exists():
            self.stage2_binary.load_state_dict(torch.load(save_dir / 'stage2_binary_best.pth'))
            self.stage2_multi.load_state_dict(torch.load(save_dir / 'stage2_multi_best.pth'))
            print("✓ 已加载最佳模型权重")
    
    def train_autoencoder(self, X_train, epochs=20):
        """训练自编码器"""
        print("开始训练自编码器...")
        
        # 创建保存目录
        save_dir = Path('./models')
        save_dir.mkdir(exist_ok=True)
        
        best_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练
            self.autoencoder.train()
            train_loss = self._train_autoencoder_epoch(X_train)
            train_losses.append(train_loss)
            
            # 验证
            self.autoencoder.eval()
            with torch.no_grad():
                val_loss = self._validate_autoencoder(X_train)  # 使用训练集的一部分作为验证集
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.autoencoder.state_dict(), save_dir / 'autoencoder_best.pth')
                print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
        
        # 计算异常检测阈值
        print("\n计算异常检测阈值...")
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            reconstructed = self.autoencoder(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            # 使用95%分位数作为阈值
            self.anomaly_threshold = float(torch.quantile(reconstruction_errors, 0.95))
            print(f"✓ 设置异常检测阈值: {self.anomaly_threshold:.6f}")
        
        return train_losses, val_losses
    
    def predict(self, X):
        """三阶段预测"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        results = []
        
        self.stage1_binary.eval()
        self.stage1_multi.eval()
        self.stage2_binary.eval()
        self.stage2_multi.eval()
        self.autoencoder.eval()
        
        with torch.no_grad():
            for i, sample in enumerate(X_tensor):
                sample = sample.unsqueeze(0)  # 添加batch维度
                
                # 阶段1：检测明显的攻击
                stage1_binary_out = torch.softmax(self.stage1_binary(sample), dim=1)
                stage1_multi_out = torch.softmax(self.stage1_multi(sample), dim=1)
                
                stage1_confidence = stage1_binary_out[0][1].item()  # 攻击的置信度
                
                # 使用较高的置信度阈值检测明显攻击
                if stage1_confidence > 0.8:  # 保持0.8的阈值
                    results.append({
                        'prediction': 'known_attack_stage1',
                        'attack_type': torch.argmax(stage1_multi_out, dim=1).item(),
                        'confidence': stage1_confidence,
                        'stage': 1
                    })
                    continue
                
                # 阶段2：检测攻击变种
                stage2_binary_out = torch.softmax(self.stage2_binary(sample), dim=1)
                stage2_multi_out = torch.softmax(self.stage2_multi(sample), dim=1)
                
                stage2_binary_confidence = stage2_binary_out[0][1].item()
                stage2_multi_confidence = torch.max(stage2_multi_out).item()
                
                # 基于调试分析的新策略：
                # 1. 二分类置信度中等以上(>0.55)，且多分类有一定信心(>0.05)
                # 2. 或者二分类置信度较高(>0.65)，不管多分类表现如何
                # 3. 考虑到多分类器学习效果差，降低对其的依赖
                
                stage2_trigger = (
                    (stage2_binary_confidence > 0.55 and stage2_multi_confidence > 0.05) or  # 宽松条件
                    (stage2_binary_confidence > 0.65) or  # 单纯依赖二分类
                    (stage2_binary_confidence > 0.50 and stage2_multi_confidence > 0.10)  # 平衡条件
                )
                
                if stage2_trigger:
                    # 计算组合置信度，但更依赖二分类结果
                    combined_confidence = stage2_binary_confidence * 0.8 + stage2_multi_confidence * 0.2
                    
                    results.append({
                        'prediction': 'known_attack_stage2',
                        'attack_subtype': torch.argmax(stage2_multi_out, dim=1).item(),
                        'confidence': combined_confidence,
                        'stage': 2
                    })
                    continue
                
                # 阶段3：自编码器异常检测
                reconstructed = self.autoencoder(sample)
                reconstruction_error = torch.mean((sample - reconstructed) ** 2).item()
                
                # 使用动态阈值，基于重构误差的分布
                if reconstruction_error > self.anomaly_threshold * 1.1:  # 保持阈值倍数为1.1
                    results.append({
                        'prediction': 'unknown_attack',
                        'reconstruction_error': reconstruction_error,
                        'confidence': min(reconstruction_error / self.anomaly_threshold, 2.0),
                        'stage': 3
                    })
                else:
                    results.append({
                        'prediction': 'benign',
                        'reconstruction_error': reconstruction_error,
                        'confidence': 1.0 - (reconstruction_error / self.anomaly_threshold),
                        'stage': 3
                    })
        
        return results
    
    def predict_binary(self, X):
        """返回二分类预测结果 (0: 正常, 1: 攻击)"""
        predictions = self.predict(X)
        
        binary_predictions = []
        for pred in predictions:
            if pred['prediction'] == 'benign':
                binary_predictions.append(0)
            else:
                binary_predictions.append(1)
                
        return np.array(binary_predictions)
    
    def evaluate(self, X_test, y_test_binary, y_test_multi=None):
        """评估模型性能"""
        predictions = self.predict(X_test)
        
        # 转换预测结果为二分类
        y_pred_binary = []
        for pred in predictions:
            if pred['prediction'] == 'benign':
                y_pred_binary.append(0)
            else:
                y_pred_binary.append(1)
        
        # 计算性能指标
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_binary, y_pred_binary, average='binary')
        
        print("\n=== 模型评估结果 ===")
        print(f"精确度 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test_binary, y_pred_binary)
        print("\n混淆矩阵:")
        print(cm)
        
        # 各阶段检测统计
        stage_stats = {'stage1': 0, 'stage2': 0, 'stage3': 0}
        for pred in predictions:
            stage_stats[f"stage{pred['stage']}"] += 1
        
        print("\n各阶段检测统计:")
        total = len(predictions)
        for stage, count in stage_stats.items():
            print(f"{stage}: {count} ({count/total*100:.1f}%)")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'stage_stats': stage_stats
        }
    
    def save_models(self):
        """保存所有模型"""
        # 创建保存目录
        save_dir = Path('./models')
        save_dir.mkdir(exist_ok=True)
        
        # 保存各个阶段的模型
        torch.save(self.stage1_binary.state_dict(), save_dir / 'stage1_binary.pth')
        torch.save(self.stage1_multi.state_dict(), save_dir / 'stage1_multi.pth')
        torch.save(self.stage2_binary.state_dict(), save_dir / 'stage2_binary.pth')
        torch.save(self.stage2_multi.state_dict(), save_dir / 'stage2_multi.pth')
        torch.save(self.autoencoder.state_dict(), save_dir / 'autoencoder.pth')
        
        # 保存数据预处理器
        with open(save_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存标签编码器
        with open(save_dir / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        print(f"✓ 所有模型已保存到 {save_dir} 目录")
    
    def load_models(self, path_prefix='improved_zero_trust_ids/enhanced_ids'):
        """加载所有模型"""
        self.stage1_binary.load_state_dict(torch.load(f'{path_prefix}_stage1_binary.pth'))
        self.stage1_multi.load_state_dict(torch.load(f'{path_prefix}_stage1_multi.pth'))
        self.stage2_binary.load_state_dict(torch.load(f'{path_prefix}_stage2_binary.pth'))
        self.stage2_multi.load_state_dict(torch.load(f'{path_prefix}_stage2_multi.pth'))
        self.autoencoder.load_state_dict(torch.load(f'{path_prefix}_autoencoder.pth'))
        self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')
        self.label_encoders = joblib.load(f'{path_prefix}_label_encoders.pkl')
        
        # 加载阈值
        with open(f'{path_prefix}_threshold.txt', 'r') as f:
            self.anomaly_threshold = float(f.read().strip())
        
        print(f"模型已从 {path_prefix}_* 加载")
    
    def _train_binary_classifier(self, X, y):
        """Train binary classifier for one epoch"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.optimizers['stage1_binary'].zero_grad()
        output = self.stage1_binary(X_tensor)
        loss = self.criterion_cls(output, y_tensor)
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in self.stage1_binary.parameters())
        loss = loss + l2_lambda * l2_norm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.stage1_binary.parameters(), max_norm=1.0)
        self.optimizers['stage1_binary'].step()
        
        return loss.item()
    
    def _train_multi_classifier(self, X, y):
        """Train multi-class classifier for one epoch"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.optimizers['stage1_multi'].zero_grad()
        output = self.stage1_multi(X_tensor)
        loss = self.criterion_cls(output, y_tensor)
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in self.stage1_multi.parameters())
        loss = loss + l2_lambda * l2_norm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.stage1_multi.parameters(), max_norm=1.0)
        self.optimizers['stage1_multi'].step()
        
        return loss.item()
    
    def _validate_stage1(self, X_val, y_bin_val, y_multi_val):
        """验证阶段1模型"""
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_bin_val_tensor = torch.LongTensor(y_bin_val).to(self.device)
        y_multi_val_tensor = torch.LongTensor(y_multi_val).to(self.device)
        
        self.stage1_binary.eval()
        self.stage1_multi.eval()
        
        with torch.no_grad():
            val_binary_output = self.stage1_binary(X_val_tensor)
            val_multi_output = self.stage1_multi(X_val_tensor)
            
            val_binary_loss = self.criterion_cls(val_binary_output, y_bin_val_tensor)
            val_multi_loss = self.criterion_cls(val_multi_output, y_multi_val_tensor)
            
            total_val_loss = val_binary_loss + val_multi_loss
        
        return total_val_loss.item()
    
    def _validate_stage2(self, X_val, y_bin_val, y_multi_val):
        """验证阶段2模型"""
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_bin_val_tensor = torch.LongTensor(y_bin_val).to(self.device)
        y_multi_val_tensor = torch.LongTensor(y_multi_val).to(self.device)
        
        self.stage2_binary.eval()
        self.stage2_multi.eval()
        
        with torch.no_grad():
            # 二分类验证
            val_binary_output = self.stage2_binary(X_val_tensor)
            val_binary_loss = self.criterion_cls(val_binary_output, y_bin_val_tensor)
            
            # 多分类验证
            val_multi_output = self.stage2_multi(X_val_tensor)
            val_multi_loss = self.criterion_cls(val_multi_output, y_multi_val_tensor)
            
            # 计算准确率
            binary_acc = (torch.argmax(val_binary_output, dim=1) == y_bin_val_tensor).float().mean()
            multi_acc = (torch.argmax(val_multi_output, dim=1) == y_multi_val_tensor).float().mean()
            
            # 综合损失：同时考虑损失和准确率
            total_val_loss = (val_binary_loss + val_multi_loss) * (2 - (binary_acc + multi_acc) / 2)
        
        return total_val_loss.item()
    
    def _train_autoencoder_epoch(self, X_train):
        """Train autoencoder for one epoch"""
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        
        self.optimizers['autoencoder'].zero_grad()
        reconstructed = self.autoencoder(X_train_tensor)
        
        # Reconstruction loss
        recon_loss = self.criterion_ae(reconstructed, X_train_tensor)
        
        # Add KL divergence loss for regularization
        encoded = self.autoencoder.encode(X_train_tensor)
        kl_loss = -0.5 * torch.mean(1 + torch.log(torch.var(encoded, dim=0) + 1e-10) - torch.mean(encoded, dim=0).pow(2) - torch.var(encoded, dim=0))
        
        # Total loss
        loss = recon_loss + 0.1 * kl_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
        self.optimizers['autoencoder'].step()
        
        return loss.item()
    
    def _validate_autoencoder(self, X_val):
        """验证自编码器"""
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        with torch.no_grad():
            val_reconstructed = self.autoencoder(X_val_tensor)
            val_loss = self.criterion_ae(val_reconstructed, X_val_tensor)
        
        return val_loss.item()
    
    def _train_stage2_binary_classifier(self, X, y):
        """训练阶段2二分类器 - 增强版"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.optimizers['stage2_binary'].zero_grad()
        output = self.stage2_binary(X_tensor)
        
        # 使用label smoothing的交叉熵损失
        loss = self.criterion_cls(output, y_tensor)
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in self.stage2_binary.parameters())
        loss = loss + l2_lambda * l2_norm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.stage2_binary.parameters(), max_norm=1.0)
        self.optimizers['stage2_binary'].step()
        
        return loss.item()
    
    def _train_stage2_multi_classifier(self, X, y, focal_loss):
        """训练阶段2多分类器 - 使用focal loss"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.optimizers['stage2_multi'].zero_grad()
        output = self.stage2_multi(X_tensor)
        
        # 使用focal loss处理类别不平衡
        loss = focal_loss(output, y_tensor)
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in self.stage2_multi.parameters())
        loss = loss + l2_lambda * l2_norm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.stage2_multi.parameters(), max_norm=1.0)
        self.optimizers['stage2_multi'].step()
        
        return loss.item()

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # 简化的focal loss实现
        CE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss 