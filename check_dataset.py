#!/usr/bin/env python3
"""
检查AirV2X数据集中不同批次的LiDAR数据和图片数据结构
"""

import os
import sys
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
sys.path.append('/home/chubin/suyi/AirV2X-Perception')

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset

def check_dataset_structure(dataloader, num_samples=3):
    """检查数据集结构"""
    print("=" * 80)
    print("检查AirV2X数据集结构")
    print("=" * 80)
    
    sample_count = 0
    for i, batch_data in enumerate(dataloader):
        if sample_count >= num_samples:
            break
        
        print(f"\n--- 样本 {sample_count+1} ---")
        
        try:
            # 检查顶层键
            print(f"顶层键: {list(batch_data.keys())}")
            
            # 检查每个agent的数据
            for agent in ['vehicle', 'rsu', 'drone']:
                if agent in batch_data['ego']:
                    print(f"\n{agent.upper()} 数据:")
                    agent_data = batch_data['ego'][agent]
                    print(f"  键: {list(agent_data.keys())}")
                    
                    # 检查LiDAR数据
                    if 'batch_merged_lidar_features_torch' in agent_data:
                        lidar_data = agent_data['batch_merged_lidar_features_torch']
                        print(f"  LiDAR数据形状: {lidar_data.shape if hasattr(lidar_data, 'shape') else type(lidar_data)}")
                    
                    # 检查相机数据
                    if 'batch_merged_cam_inputs' in agent_data:
                        cam_data = agent_data['batch_merged_cam_inputs']
                        print(f"  相机数据: {type(cam_data)}")
                        if isinstance(cam_data, dict):
                            print(f"    相机数据键: {list(cam_data.keys())}")
                            if 'imgs' in cam_data:
                                imgs = cam_data['imgs']
                                print(f"    图像形状: {imgs.shape if hasattr(imgs, 'shape') else type(imgs)}")
                            else:
                                print("    没有'imgs'键")
                        else:
                            print(f"    相机数据为空: {cam_data}")
                    else:
                        print("  没有相机数据")
                
        except Exception as e:
            print(f"处理样本 {sample_count+1} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        sample_count += 1

def check_origin_lidar_data(dataloader, num_samples=6):
    """检查origin_lidar数据"""
    print("\n" + "=" * 80)
    print("检查origin_lidar数据")
    print("=" * 80)
    
    sample_count = 0
    for i, batch_data in enumerate(dataloader):
        if sample_count >= num_samples:
            break
            
        print(f"\n--- 样本 {sample_count+1} ---")
        
        try:
            # 从batch_data中获取ego数据，注意batch_data是批处理格式
            ego_data = batch_data['ego']
            
            # 检查origin_lidar
            if 'origin_lidar' in ego_data:
                origin_lidar = ego_data['origin_lidar']
                print(f"origin_lidar形状: {origin_lidar.shape if hasattr(origin_lidar, 'shape') else type(origin_lidar)}")
                if hasattr(origin_lidar, 'shape') and len(origin_lidar.shape) > 0:
                    print(f"origin_lidar类型: {origin_lidar.dtype}")
                    print(f"origin_lidar非零元素数: {torch.count_nonzero(origin_lidar).item()}")
            else:
                print("没有origin_lidar")
            
            # 检查各agent的origin_lidar
            for agent in ['vehicle', 'rsu', 'drone']:
                origin_lidar_key = f'origin_lidar_{agent}'
                if origin_lidar_key in ego_data:
                    origin_lidar = ego_data[origin_lidar_key]
                    print(f"{origin_lidar_key}形状: {origin_lidar.shape if hasattr(origin_lidar, 'shape') else type(origin_lidar)}")
                    if hasattr(origin_lidar, 'shape') and len(origin_lidar.shape) > 0:
                        print(f"{origin_lidar_key}类型: {origin_lidar.dtype}")
                        print(f"{origin_lidar_key}非零元素数: {torch.count_nonzero(origin_lidar).item()}")
                else:
                    print(f"没有{origin_lidar_key}")
                    
        except Exception as e:
            print(f"处理样本 {sample_count+1} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        sample_count += 1

def main():
    """主函数"""
    # 加载配置
    config_path = "/home/chubin/suyi/AirV2X-Perception/AirV2X-Perception-Checkpoints/airv2x_intermediate_mambafusion/config.yaml"
    
    try:
        print("加载配置文件...")
        params = load_yaml(config_path)
        print("配置文件加载成功")
        
        # 修改为使用测试数据集
        params['root_dir'] = '/home/chubin/suyi/dataset/AirV2X-Perception/test/test'
        params['validate_dir'] = '/home/chubin/suyi/dataset/AirV2X-Perception/test/test'
        print(f"使用测试数据集: {params['root_dir']}")
        
        # 创建数据集
        print("创建测试数据集...")
        dataset = build_dataset(params, visualize=True, train=False)
        print(f"测试数据集创建成功，总样本数: {len(dataset)}")
        
        # 创建DataLoader
        print("创建DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False
        )
        print("DataLoader创建成功")
        
        # 检查数据集结构
        check_dataset_structure(dataloader, num_samples=3)
        
        # 检查origin_lidar数据
        check_origin_lidar_data(dataloader, num_samples=10)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
