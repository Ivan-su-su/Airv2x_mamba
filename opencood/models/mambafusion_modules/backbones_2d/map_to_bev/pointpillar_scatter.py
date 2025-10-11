import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
import numpy as np
import os

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        import pdb; pdb.set_trace()
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def _generate_bev_features(self, pillar_features, coords):
        """
        为单个agent生成BEV特征的辅助方法
        
        Args:
            pillar_features: [num_pillars, num_features]
            coords: [num_pillars, 4] (batch_idx, z, y, x)
            
        Returns:
            bev_features: [batch_size, num_features, ny, nx]
        """
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()


            dense_feats = scatter_mean(pillars, indices, dim=1)
            
            dense_len = dense_feats.shape[1]
            spatial_feature[:, :dense_len] = dense_feats

            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        
        # 根据网格尺寸进行池化以减少内存占用
        # 修改池化策略：保持长宽比，但限制最大尺寸
            
        
        return batch_spatial_features

    def forward(self, batch_dict, visualize=False, **kwargs):
        # Merge pillar features from multiple agents if present
        if not ('pillar_features' in batch_dict and 'voxel_coords' in batch_dict):
            fused_pillars = []
            fused_coords = []
            agent_names = []
            agent_bev_features = {}  # 存储每个agent的BEV特征
            
            # collect agent sub-dicts that carry pillars/coords
            for k, v in batch_dict.items():
                if isinstance(v, dict) and ('pillar_features' in v) and ('voxel_coords' in v):
                    fused_pillars.append(v['pillar_features'])
                    fused_coords.append(v['voxel_coords'])
                    agent_names.append(k)
                    
                    # 为每个agent单独生成BEV特征（用于可视化）
                    agent_pillars = v['pillar_features']
                    agent_coords = v['voxel_coords']
                    agent_bev = self._generate_bev_features(agent_pillars, agent_coords)
                    agent_bev_features[k] = agent_bev
                    
            if len(fused_pillars) == 0:
                raise KeyError('No pillar_features/voxel_coords found in batch_dict or agent sub-dicts')
            pillar_features = torch.cat(fused_pillars, dim=0)
            coords = torch.cat(fused_coords, dim=0)
            batch_dict['pillar_features'] = pillar_features
            batch_dict['voxel_coords'] = coords
            
            # 将各agent的BEV特征存储到batch_dict中
            batch_dict['agent_bev_features'] = agent_bev_features
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] # [num_of_voxel, 128] [num_of_voxel, 4] batch_idx, z, y, x
            agent_names = None
            agent_bev_features = None
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device) # [128, 1*360*360]

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3] # [num_of_voxel_]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :] # [num_of_voxel_, 128]
            pillars = pillars.t() # [128, num_of_voxel_]

            dense_feats = scatter_mean(pillars,indices,dim=1) # [128, 1*360*360接近] 将pillars按照indices的位置加到dense_feats上并求平均
            dense_len = dense_feats.shape[1] # 1*360*360接近
            spatial_feature[:,:dense_len] = dense_feats # 将dense_feats的值赋给spatial_feature [128, 1*360*360] 相当于转换为一个dense的特征？

            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0) # torch.Size([2, 128, 1*360*360])
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx) # torch.Size([2, 128, 360, 360])
        
        # 根据网格尺寸进行池化以减少内存占用
        # if self.ny > 200 or self.nx > 200:
        #     target_h = min(200, self.ny)
        #     target_w = min(200, self.nx)
        #     batch_spatial_features = F.adaptive_avg_pool2d(batch_spatial_features, (target_h, target_w))
        #     print(f"[PointPillarScatter] 池化后尺寸: {batch_spatial_features.shape}")
        
        batch_dict['spatial_features'] = batch_spatial_features # torch.Size([2, 128, 180, 180])
        # 可视化BEV特征
        visualize = True
        if visualize:
            print("[PointPillarScatter] 开始可视化BEV特征...")
            
            # 如果有多个agent，使用各agent独立的BEV特征进行可视化
            if agent_bev_features is not None:
                print(f"[PointPillarScatter] 可视化{len(agent_bev_features)}个agent的独立BEV特征")
                self.visualize_multi_agent_bev_simple_independent(agent_bev_features, agent_names)
            else:
                # 单agent情况，使用融合后的特征
                self.visualize_bev_simple(batch_dict, agent_names=agent_names)
            
            print("[PointPillarScatter] BEV特征可视化完成!")
        import pdb; pdb.set_trace()
        return batch_dict

    def visualize_bev_simple(self, batch_dict, save_dir="./bev_simple", agent_names=None):
        """
        简单直观的BEV可视化 - 显示网格占用和强度分布
        
        Args:
            batch_dict: 包含spatial_features的字典
            save_dir: 保存图片的目录
            agent_names: agent名称列表
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if 'spatial_features' not in batch_dict:
            print("Warning: No spatial_features found in batch_dict")
            return
            
        spatial_features = batch_dict['spatial_features']  # [B, C, H, W]
        batch_size, num_channels, height, width = spatial_features.shape
        
        print(f"[BEV Simple] Features shape: {spatial_features.shape}")
        
        # 为每个batch样本创建可视化
        for batch_idx in range(batch_size):
            batch_features = spatial_features[batch_idx]  # [C, H, W]
            
            # 计算所有通道的平均强度作为占用强度
            occupancy_map = torch.mean(batch_features, dim=0).detach().cpu().numpy()  # [H, W]
            
            # 创建可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 左图：占用强度热力图
            im1 = ax1.imshow(occupancy_map, cmap='hot', aspect='equal')
            ax1.set_title('BEV Occupancy Intensity')
            ax1.set_xlabel('X (width)')
            ax1.set_ylabel('Y (height)')
            plt.colorbar(im1, ax=ax1, label='Intensity')
            
            # 右图：占用统计
            occupied_cells = np.count_nonzero(occupancy_map)
            total_cells = occupancy_map.size
            occupancy_ratio = occupied_cells / total_cells
            
            # 强度统计
            max_intensity = np.max(occupancy_map)
            mean_intensity = np.mean(occupancy_map[occupancy_map > 0]) if occupied_cells > 0 else 0
            
            # 显示统计信息
            stats_text = f"""BEV Grid Statistics:
            
Grid Size: {height} × {width}
Total Cells: {total_cells:,}
Occupied Cells: {occupied_cells:,}
Occupancy Ratio: {occupancy_ratio:.2%}

Intensity Statistics:
Max Intensity: {max_intensity:.4f}
Mean Intensity: {mean_intensity:.4f}
Non-zero Mean: {mean_intensity:.4f}"""
            
            ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('Statistics')
            
            # 添加总标题
            agent_name = agent_names[batch_idx] if agent_names and batch_idx < len(agent_names) else f"Agent_{batch_idx}"
            fig.suptitle(f'BEV Occupancy Map - {agent_name}', fontsize=14)
            
            # 调整布局以避免重叠 - 增加更多间距
            plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.6)
            
            # 保存图片
            save_path = os.path.join(save_dir, f'bev_simple_{agent_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[BEV Simple] Saved: {save_path}")
            print(f"[BEV Simple] {agent_name}: {occupied_cells:,}/{total_cells:,} cells occupied ({occupancy_ratio:.2%})")
    
    def visualize_multi_agent_bev_simple(self, batch_dict, save_dir="./multi_agent_bev_simple"):
        """
        多Agent简单BEV对比可视化
        
        Args:
            batch_dict: 包含多个agent数据的字典
            save_dir: 保存图片的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 收集所有agent的BEV特征
        agent_bev_features = {}
        agent_names = []
        
        for k, v in batch_dict.items():
            if isinstance(v, dict) and 'spatial_features' in v:
                agent_bev_features[k] = v['spatial_features']
                agent_names.append(k)
        
        if len(agent_bev_features) == 0:
            print("Warning: No agent spatial_features found in batch_dict")
            return
            
        print(f"[Multi-Agent BEV Simple] Found {len(agent_names)} agents: {agent_names}")
        
        # 获取特征维度
        first_agent = list(agent_bev_features.values())[0]
        batch_size, num_channels, height, width = first_agent.shape
        
        # 为每个batch创建对比图
        for batch_idx in range(batch_size):
            fig, axes = plt.subplots(2, len(agent_names), figsize=(4*len(agent_names), 8))
            if len(agent_names) == 1:
                axes = axes.reshape(2, 1)
            
            # 收集所有agent的占用图用于统一颜色范围
            all_occupancy_maps = []
            for agent_name in agent_names:
                features = agent_bev_features[agent_name][batch_idx]  # [C, H, W]
                # 使用L2范数而不是平均值，避免零值问题
                occupancy_map = torch.norm(features, dim=0).detach().cpu().numpy()
                all_occupancy_maps.append(occupancy_map)
            
            # 计算统一的颜色范围
            all_values = np.concatenate([m.flatten() for m in all_occupancy_maps])
            vmin, vmax = np.min(all_values), np.max(all_values)
            
            for i, agent_name in enumerate(agent_names):
                occupancy_map = all_occupancy_maps[i]
                
                # 上排：占用强度热力图
                im1 = axes[0, i].imshow(occupancy_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
                axes[0, i].set_title(f'{agent_name}\nOccupancy Intensity')
                axes[0, i].set_xlabel('X')
                axes[0, i].set_ylabel('Y')
                
                # 下排：占用统计
                occupied_cells = np.count_nonzero(occupancy_map)
                total_cells = occupancy_map.size
                occupancy_ratio = occupied_cells / total_cells
                max_intensity = np.max(occupancy_map)
                mean_intensity = np.mean(occupancy_map[occupancy_map > 0]) if occupied_cells > 0 else 0
                
                stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
Ratio: {occupancy_ratio:.2%}
Max: {max_intensity:.3f}
Mean: {mean_intensity:.3f}"""
                
                axes[1, i].text(0.1, 0.9, stats_text, transform=axes[1, i].transAxes, 
                              fontsize=9, verticalalignment='top', fontfamily='monospace')
                axes[1, i].set_xlim(0, 1)
                axes[1, i].set_ylim(0, 1)
                axes[1, i].axis('off')
                axes[1, i].set_title('Statistics')
            
            # 添加颜色条
            plt.colorbar(im1, ax=axes[0, :], orientation='horizontal', pad=0.1, label='Intensity')
            
            fig.suptitle(f'Multi-Agent BEV Comparison - Batch {batch_idx}', fontsize=14)
            
            # 调整布局以避免重叠 - 增加更多间距
            plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.6)
            
            # 保存对比图
            save_path = os.path.join(save_dir, f'multi_agent_bev_simple_batch_{batch_idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[Multi-Agent BEV Simple] Saved: {save_path}")
            
            # 打印统计信息
            print(f"[Multi-Agent BEV Simple] Batch {batch_idx} Statistics:")
            for agent_name in agent_names:
                occupancy_map = all_occupancy_maps[agent_names.index(agent_name)]
                occupied_cells = np.count_nonzero(occupancy_map)
                total_cells = occupancy_map.size
                occupancy_ratio = occupied_cells / total_cells
                print(f"  {agent_name}: {occupied_cells:,}/{total_cells:,} cells ({occupancy_ratio:.2%})")
    
    def visualize_multi_agent_bev_simple_independent(self, agent_bev_features, agent_names, save_dir="./multi_agent_bev_independent"):
        """
        可视化各agent独立的BEV特征（在合并之前）
        
        Args:
            agent_bev_features: 字典，包含每个agent的BEV特征
            agent_names: agent名称列表
            save_dir: 保存图片的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[Multi-Agent BEV Independent] Found {len(agent_names)} agents: {agent_names}")
        
        # 获取特征维度
        first_agent = list(agent_bev_features.values())[0]
        batch_size, num_channels, height, width = first_agent.shape
        
        # 为每个batch创建对比图
        for batch_idx in range(batch_size):
            fig, axes = plt.subplots(2, len(agent_names), figsize=(4*len(agent_names), 8))
            if len(agent_names) == 1:
                axes = axes.reshape(2, 1)
            
            # 收集所有agent的占用图用于统一颜色范围
            all_occupancy_maps = []
            for agent_name in agent_names:
                features = agent_bev_features[agent_name][batch_idx]  # [C, H, W]
                # 使用L2范数而不是平均值，避免零值问题
                occupancy_map = torch.norm(features, dim=0).detach().cpu().numpy()
                all_occupancy_maps.append(occupancy_map)
            
            # 计算统一的颜色范围
            all_values = np.concatenate([m.flatten() for m in all_occupancy_maps])
            vmin, vmax = np.min(all_values), np.max(all_values)
            
            for i, agent_name in enumerate(agent_names):
                occupancy_map = all_occupancy_maps[i]
                
                # 上排：占用强度热力图
                im1 = axes[0, i].imshow(occupancy_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
                axes[0, i].set_title(f'{agent_name}\nIndependent BEV')
                axes[0, i].set_xlabel('X')
                axes[0, i].set_ylabel('Y')
                
                # 下排：占用统计（只显示占用率）
                occupied_cells = np.count_nonzero(occupancy_map)
                total_cells = occupancy_map.size
                occupancy_ratio = occupied_cells / total_cells
                
                stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
Ratio: {occupancy_ratio:.2%}"""
                
                axes[1, i].text(0.1, 0.9, stats_text, transform=axes[1, i].transAxes, 
                              fontsize=9, verticalalignment='top', fontfamily='monospace')
                axes[1, i].set_xlim(0, 1)
                axes[1, i].set_ylim(0, 1)
                axes[1, i].axis('off')
                axes[1, i].set_title('Statistics')
            
            fig.suptitle(f'Independent Agent BEV Comparison - Batch {batch_idx}', fontsize=14)
            
            # 调整布局以避免重叠 - 去掉颜色条后减少底部空间
            plt.subplots_adjust(top=0.8, bottom=0.1, hspace=0.6)
            
            # 保存对比图
            save_path = os.path.join(save_dir, f'independent_agent_bev_batch_{batch_idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[Multi-Agent BEV Independent] Saved: {save_path}")
            
            # 打印统计信息
            print(f"[Multi-Agent BEV Independent] Batch {batch_idx} Statistics:")
            for agent_name in agent_names:
                occupancy_map = all_occupancy_maps[agent_names.index(agent_name)]
                occupied_cells = np.count_nonzero(occupancy_map)
                total_cells = occupancy_map.size
                occupancy_ratio = occupied_cells / total_cells
                print(f"  {agent_name}: {occupied_cells:,}/{total_cells:,} cells ({occupancy_ratio:.2%})")
    
    def visualize_multi_agent_bev(self, batch_dict, save_dir="./multi_agent_bev", max_features=4):
        """
        可视化多Agent的BEV特征图对比
        
        Args:
            batch_dict: 包含多个agent数据的字典
            save_dir: 保存图片的目录
            max_features: 最多可视化的特征通道数
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 收集所有agent的BEV特征
        agent_bev_features = {}
        agent_names = []
        
        for k, v in batch_dict.items():
            if isinstance(v, dict) and 'spatial_features' in v:
                agent_bev_features[k] = v['spatial_features']
                agent_names.append(k)
        
        if len(agent_bev_features) == 0:
            print("Warning: No agent spatial_features found in batch_dict")
            return
            
        print(f"[Multi-Agent BEV] Found {len(agent_names)} agents: {agent_names}")
        
        # 获取特征维度
        first_agent = list(agent_bev_features.values())[0]
        batch_size, num_channels, height, width = first_agent.shape
        num_vis_channels = min(max_features, num_channels)
        
        # 为每个batch和每个特征通道创建对比图
        for batch_idx in range(batch_size):
            for channel_idx in range(num_vis_channels):
                fig, axes = plt.subplots(1, len(agent_names), figsize=(4*len(agent_names), 4))
                if len(agent_names) == 1:
                    axes = [axes]
                
                for i, agent_name in enumerate(agent_names):
                    features = agent_bev_features[agent_name][batch_idx, channel_idx].detach().cpu().numpy()
                    
                    im = axes[i].imshow(features, cmap='viridis', aspect='equal')
                    axes[i].set_title(f'{agent_name}\nChannel {channel_idx}')
                    axes[i].set_xlabel('X')
                    axes[i].set_ylabel('Y')
                    
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                
                fig.suptitle(f'Multi-Agent BEV Comparison - Batch {batch_idx}, Channel {channel_idx}', fontsize=14)
                
                # 调整布局以避免重叠
                plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.3)
                
                # 保存对比图
                save_path = os.path.join(save_dir, f'multi_agent_bev_batch_{batch_idx}_channel_{channel_idx}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"[Multi-Agent BEV] Saved: {save_path}")
    
    def visualize_bev_statistics(self, batch_dict, save_dir="./bev_statistics"):
        """
        可视化BEV特征的统计信息
        
        Args:
            batch_dict: 包含spatial_features的字典
            save_dir: 保存图片的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if 'spatial_features' not in batch_dict:
            print("Warning: No spatial_features found in batch_dict")
            return
            
        spatial_features = batch_dict['spatial_features']  # [B, C, H, W]
        batch_size, num_channels, height, width = spatial_features.shape
        
        # 计算统计信息
        features_np = spatial_features.detach().cpu().numpy()
        
        # 创建统计图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 特征值分布直方图
        all_values = features_np.flatten()
        axes[0, 0].hist(all_values, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Feature Value Distribution')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 每个通道的平均激活强度
        channel_means = np.mean(features_np, axis=(0, 2, 3))
        axes[0, 1].plot(channel_means)
        axes[0, 1].set_title('Mean Activation per Channel')
        axes[0, 1].set_xlabel('Channel Index')
        axes[0, 1].set_ylabel('Mean Activation')
        
        # 3. 非零像素比例
        non_zero_ratios = []
        for batch_idx in range(batch_size):
            for channel_idx in range(num_channels):
                channel_data = features_np[batch_idx, channel_idx]
                non_zero_ratio = np.count_nonzero(channel_data) / channel_data.size
                non_zero_ratios.append(non_zero_ratio)
        
        axes[1, 0].hist(non_zero_ratios, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('Non-zero Pixel Ratio Distribution')
        axes[1, 0].set_xlabel('Non-zero Ratio')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. 特征图尺寸信息
        axes[1, 1].text(0.1, 0.8, f'Batch Size: {batch_size}', fontsize=12)
        axes[1, 1].text(0.1, 0.7, f'Channels: {num_channels}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Height: {height}', fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'Width: {width}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Total Elements: {features_np.size:,}', fontsize=12)
        axes[1, 1].text(0.1, 0.3, f'Memory Size: {features_np.nbytes / 1024**2:.2f} MB', fontsize=12)
        axes[1, 1].set_title('BEV Feature Statistics')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        fig.suptitle('BEV Features Statistical Analysis', fontsize=16)
        
        # 调整布局以避免重叠
        plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.3)
        
        # 保存统计图
        save_path = os.path.join(save_dir, 'bev_statistics.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[BEV Statistics] Saved: {save_path}")
        print(f"[BEV Statistics] Feature shape: {spatial_features.shape}")
        print(f"[BEV Statistics] Mean value: {np.mean(features_np):.4f}")
        print(f"[BEV Statistics] Std value: {np.std(features_np):.4f}")
        print(f"[BEV Statistics] Min value: {np.min(features_np):.4f}")
        print(f"[BEV Statistics] Max value: {np.max(features_np):.4f}")
