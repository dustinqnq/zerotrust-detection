#!/usr/bin/env python3
"""
分析Mirai攻击使用的协议类型
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cic_iot_data_processor import MiraiDDoSDataProcessor

def analyze_mirai_protocols():
    """分析Mirai攻击的协议使用情况"""
    print('🔍 分析Mirai攻击的协议使用情况...')
    
    # 加载数据处理器
    processor = MiraiDDoSDataProcessor()
    
    # 加载数据进行协议分析
    data = processor.load_data(max_files=8, sample_per_file=2000)
    
    if data is None:
        print('❌ 数据加载失败')
        return
    
    print(f'📊 总样本数量: {len(data):,}')
    
    # 筛选Mirai相关攻击
    mirai_data = data[data['Label'].str.contains('MIRAI', case=False, na=False)]
    
    print(f'📊 Mirai样本数量: {len(mirai_data):,}')
    
    if len(mirai_data) == 0:
        print('❌ 未找到Mirai样本')
        return
    
    print(f'\n🎯 Mirai攻击类型分布:')
    mirai_types = mirai_data['Label'].value_counts()
    for attack_type, count in mirai_types.items():
        print(f'   {attack_type}: {count:,} 样本')
    
    # 分析协议字段
    if 'Protocol' in mirai_data.columns:
        print(f'\n📡 Mirai使用的协议分布:')
        protocols = mirai_data['Protocol'].value_counts()
        for protocol, count in protocols.items():
            print(f'   协议 {protocol}: {count:,} 样本 ({count/len(mirai_data)*100:.1f}%)')
    
    # 分析所有列名，寻找协议相关信息
    print(f'\n🔍 数据集列名分析:')
    columns = list(mirai_data.columns)
    
    # 寻找协议相关列
    protocol_cols = [col for col in columns if any(keyword in col.lower() 
                    for keyword in ['protocol', 'tcp', 'udp', 'icmp', 'flag'])]
    
    if protocol_cols:
        print(f'📡 协议相关字段: {protocol_cols}')
        
        # 分析TCP标志位
        tcp_flags = [col for col in protocol_cols if 'flag' in col.lower()]
        if tcp_flags:
            print(f'\n🚩 TCP标志位分析:')
            for flag in tcp_flags:
                if flag in mirai_data.columns:
                    avg_flag = mirai_data[flag].mean()
                    max_flag = mirai_data[flag].max()
                    nonzero_count = (mirai_data[flag] > 0).sum()
                    print(f'   {flag}:')
                    print(f'     平均值: {avg_flag:.2f}')
                    print(f'     最大值: {max_flag:.0f}')
                    print(f'     非零样本: {nonzero_count:,} ({nonzero_count/len(mirai_data)*100:.1f}%)')
    
    # 分析端口相关信息
    port_cols = [col for col in columns if any(keyword in col.lower() 
                for keyword in ['port', 'dst', 'src'])]
    
    if port_cols:
        print(f'\n🔌 端口相关字段: {port_cols[:5]}')  # 只显示前5个
    
    # 分析流量特征，推断协议使用
    print(f'\n📊 流量特征分析 (推断协议使用):')
    
    # 检查包大小分布
    pkt_size_cols = [col for col in columns if any(keyword in col.lower() 
                    for keyword in ['pkt', 'packet', 'size', 'len'])]
    
    if pkt_size_cols:
        print(f'📦 包大小相关字段: {pkt_size_cols[:3]}')
        for col in pkt_size_cols[:3]:
            if col in mirai_data.columns:
                avg_size = mirai_data[col].mean()
                print(f'   {col} 平均值: {avg_size:.2f}')
    
    # 分析流量速率
    flow_cols = [col for col in columns if any(keyword in col.lower() 
                for keyword in ['flow', 'byts', 'rate'])]
    
    if flow_cols:
        print(f'🌊 流量速率字段: {flow_cols[:3]}')
        for col in flow_cols[:3]:
            if col in mirai_data.columns:
                avg_flow = mirai_data[col].mean()
                print(f'   {col} 平均值: {avg_flow:.2f}')

if __name__ == "__main__":
    analyze_mirai_protocols() 