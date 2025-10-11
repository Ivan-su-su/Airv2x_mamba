import torch
from torch import nn

from ...vmamba.vmamba import SS2D, VSSBlock, Linear2d, LayerNorm2d
from collections import OrderedDict
from ..base_bev_backbone import BasicBlock
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import os
import numpy as np
class ConvFuser(nn.Module):
    """
    【AirV2X多agent ConvFuser模块】
    
    功能：融合多agent的图像BEV特征和激光雷达BEV特征
    
    架构对比：
    - MambaFusion: 单agent多视角 -> 统一BEV -> 双模态融合
    - AirV2X: 多agent独立 -> 多BEV -> 多agent融合 -> 双模态融合
    
    输入：
    - 多agent情况: batch_dict[agent]['spatial_features_img'] 每个 [B_i, 80, H, W]
    - 单agent情况: batch_dict['spatial_features_img'] [B, 80, H, W]
    - 激光雷达: batch_dict['spatial_features'] [B, 128, H, W]
    
    输出：
    - batch_dict['spatial_features'] [B, 128, H, W] 与MambaFusion对齐
    
    融合策略：
    - mean: 平均融合，保留所有agent信息
    - max: 最大融合，突出最强特征
    - concat: 通道拼接，保留所有原始信息
    """
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.merge_type = self.model_cfg.get('MERGE_TYPE', 'default')
        
        # 支持只用雷达数据的情况
        self.lidar_only = self.model_cfg.get('LIDAR_ONLY', False)

        # 根据是否只用雷达数据调整输入通道数
        if self.lidar_only:
            # 只用雷达数据时，输入通道数只有雷达特征维度
            lidar_channel = 64  # 雷达特征维度
            actual_in_channel = lidar_channel
        else:
            # 使用图像+雷达数据时，输入通道数是两者之和
            actual_in_channel = in_channel
            
        if self.merge_type == 'default':
            self.conv = nn.Sequential(
                nn.Conv2d(actual_in_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
                )
        else:
            self.conv = nn.Sequential(
                # DepthwiseSeparableConv(actual_in_channel, actual_in_channel, 3, 1, 1),
                nn.Conv2d(actual_in_channel, out_channel * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel * 2),
                nn.ReLU(),
                nn.Conv2d(out_channel * 2, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                )
        self.use_vmamba = model_cfg.get('USE_VMAMBA', False)
        self.use_checkpoint = model_cfg.get('USE_CHECKPOINT', True)
        self.use_merge_after = model_cfg.get('USE_MERGE_AFTER', False)
        self.agent_fusion_strategy = model_cfg.get('AGENT_FUSION_STRATEGY', 'mean')  # 'mean', 'concat', 'max'
        if self.use_merge_after:
            depths = [1]
            num_block = len(depths)
            merge_dim = 208
            self.merge_blocks = nn.ModuleList()
            # self.merge_norm = nn.ModuleList()
            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
            for i_layer in range(num_block):
                self.merge_blocks.append(self._make_vmamba_layer(
                    dim=merge_dim,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=1,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))
        if self.use_vmamba:
            # self.img_pos_embed_layer = PositionEmbeddingLearned(20, 128)
            # self.lidar_pos_embed_layer = PositionEmbeddingLearned(3, 128)
            self.use_dw_conv = True
            depths = [1, 1, 1] # [1, 2, 2]
            num_block = len(depths)
            image_dim = 80
            point_dim = 128
            cross_dim = 128
            ssm_conv = 3
            max_channel = 1
            use_4x = False
            self.use_cross = False
            self.use_res_merge = False
            d_state = 1
            self.image_down_blocks = nn.ModuleList()
            self.image_de_blocks = nn.ModuleList()
            self.lidar_de_blocks = nn.ModuleList()
            self.lidar_down_blocks = nn.ModuleList()

            
            if self.use_res_merge:
                self.image_norm = nn.ModuleList()
                self.point_norm = nn.ModuleList()


            self.image_vmamba_blocks = nn.ModuleList()
            self.point_vmamba_blocks = nn.ModuleList()
            num_block_cross = 0

            if self.use_cross:
                depths_cross = [1, 1, 1]
                self.use_res_merge = False
                if not self.use_res_merge:
                    self.image_cross_blocks = nn.ModuleList()
                    self.point_cross_blocks = nn.ModuleList()
                
                self.image_up_blocks = nn.ModuleList()
                
                num_block_cross = len(depths_cross)
                dpr_cross = []
                for x in torch.linspace(0, 0.1, sum(depths_cross)):
                    dpr_cross.extend([x.item(), x.item()])
                self.cross_vmamba_blocks = nn.ModuleList()
                for i_layer in range(num_block_cross):
                    self.image_up_blocks.append(
                        nn.Sequential(
                            nn.Conv2d(image_dim, cross_dim, kernel_size=1),
                            nn.BatchNorm2d(cross_dim),
                            nn.ReLU(),
                            DepthwiseSeparableConv(cross_dim, cross_dim, 3, 1, 1),
                        )
                    )
                    if not self.use_res_merge:
                        self.image_cross_blocks.append(
                            nn.Sequential(
                                nn.Conv2d(cross_dim * 2, image_dim,  3, padding=1, bias=False),
                                nn.BatchNorm2d(image_dim),
                                nn.ReLU(),
                                # DepthwiseSeparableConv(cross_dim * 2, cross_dim * 2, 3, 1, 1),
                                DepthwiseSeparableConv(image_dim, image_dim, 3, 1, 1),
                            )
                        )
                        self.point_cross_blocks.append(
                            nn.Sequential(
                                nn.Conv2d(cross_dim * 2, cross_dim, 3, padding=1, bias=False),
                                nn.BatchNorm2d(cross_dim),
                                nn.ReLU(),
                                # DepthwiseSeparableConv(cross_dim * 2, cross_dim * 2, 3, 1, 1),
                                DepthwiseSeparableConv(cross_dim, cross_dim, 3, 1, 1),
                            )
                        )
                    self.cross_vmamba_blocks.append(self._make_vmamba_layer(
                        dim=cross_dim,
                        cross_dim=cross_dim,
                        drop_path = dpr_cross[sum(depths_cross[:i_layer]):sum(depths_cross[:i_layer + 1])],
                        use_checkpoint=False,
                        norm_layer=LayerNorm2d,
                        downsample=nn.Identity(),
                        channel_first=True,
                        # =================
                        ssm_d_state=d_state,
                        ssm_ratio=1.0,
                        ssm_dt_rank='auto',
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type='cross_noz',
                        # =================
                        mlp_ratio=4.0,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=0.0,
                        gmlp=False,
                        cross=True,
                    ))
            if not self.use_res_merge:
                self.image_conv = nn.Sequential(
                        nn.Conv2d(image_dim * (num_block + 1), image_dim * 2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(image_dim * 2),
                        nn.ReLU(),
                        DepthwiseSeparableConv(image_dim * 2, image_dim, 3, 1, 1),
                    )
                self.lidar_conv = nn.Sequential(
                        nn.Conv2d(point_dim * (num_block + 1), point_dim *2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(point_dim * 2),
                        nn.ReLU(),
                        DepthwiseSeparableConv(point_dim * 2, point_dim, 3, 1, 1),
                    )

            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

            for i_layer in range(num_block):
                if self.use_res_merge:
                    self.image_norm.append(nn.BatchNorm2d(image_dim))
                    self.point_norm.append(nn.BatchNorm2d(point_dim))

                # if i_layer == 0 and use_4x:
                #     point_cur_layers.append(BasicBlock(point_dim, point_dim, 2, 1, True))
                if self.use_dw_conv:
                    image_cur_layers = [
                        BasicBlock(image_dim*min(i_layer + 1, max_channel), image_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        DepthwiseSeparableConv(image_dim*min(i_layer + 2, max_channel), image_dim*min(i_layer + 2, max_channel), 3, 1, 1),
                    ]
                    point_cur_layers = [
                        BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        DepthwiseSeparableConv(point_dim*min(i_layer + 2, max_channel), point_dim*min(i_layer + 2, max_channel), 3, 1, 1),
                    ]
                else:
                    image_cur_layers = [
                        BasicBlock(image_dim*min(i_layer + 1, max_channel), image_dim*min(i_layer + 2, max_channel), 2, 1, True),
                    ]
                    # if i_layer == 0 and use_4x:
                    #     image_cur_layers.append(BasicBlock(image_dim, image_dim, 2, 1, True))
                    
                    point_cur_layers = [
                        BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                    ]
                self.image_down_blocks.append(nn.Sequential(*image_cur_layers))
                self.lidar_down_blocks.append(nn.Sequential(*point_cur_layers))
                
                    
                image_cur_de_layers = []
                point_cur_de_layers = []

                for j in range(i_layer + 1):
                    # if self.use_cross:
                    #     image_cur_de_layers.append(nn.ConvTranspose2d(point_dim, image_dim, kernel_size=2, stride=2, bias=False))
                    # else:
                    image_cur_de_layers.append(nn.ConvTranspose2d(image_dim*min(i_layer + 2 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                    image_cur_de_layers.append(nn.BatchNorm2d(image_dim*min(i_layer + 1 - j, max_channel)))
                    image_cur_de_layers.append(nn.ReLU())
                    point_cur_de_layers.append(nn.ConvTranspose2d(point_dim*min(i_layer + 2 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                    point_cur_de_layers.append(nn.BatchNorm2d(point_dim*min(i_layer + 1 - j, max_channel)))
                    point_cur_de_layers.append(nn.ReLU())
                    if self.use_dw_conv:
                        image_cur_de_layers.append(DepthwiseSeparableConv(image_dim*min(i_layer + 1 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                        point_cur_de_layers.append(DepthwiseSeparableConv(point_dim*min(i_layer + 1 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                self.image_de_blocks.append(nn.Sequential(*image_cur_de_layers))
                self.lidar_de_blocks.append(nn.Sequential(*point_cur_de_layers))
                self.image_vmamba_blocks.append(self._make_vmamba_layer(
                    dim = image_dim*min(i_layer + 2, max_channel),
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=d_state,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))

                self.point_vmamba_blocks.append(self._make_vmamba_layer(
                    dim = point_dim*min(i_layer + 2, max_channel),
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=d_state,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))
    @staticmethod
    def _make_vmamba_layer(
        dim=96,
        cross_dim=0,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        cross=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        
        if cross_dim != 0:
            blocks1 = []
            blocks2 = []
        else:
            blocks = []
        for d in range(depth):
            if cross_dim != 0:
                blocks1.append(VSSBlock(
                    hidden_dim=dim, 
                    cross_dim=cross_dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
                blocks2.append(VSSBlock(
                    hidden_dim=cross_dim, 
                    cross_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
            else:
                blocks.append(VSSBlock(
                    hidden_dim=dim, 
                    cross_dim=0,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
        if not cross:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
                downsample=downsample,
            ))
        else:
            return nn.Sequential(OrderedDict(
                blocks1=nn.Sequential(*blocks1),
                blocks2=nn.Sequential(*blocks2),
            ))
    def forward(self,batch_dict):

        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality (单agent情况)
                或 batch_dict[agent]['spatial_features_img']: 各agent的BEV特征 (多agent情况)
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        # 如果只用雷达数据，直接处理雷达特征
        if self.lidar_only:
            lidar_bev = batch_dict['spatial_features']
            
            # 直接对雷达特征进行卷积处理
            mm_bev = self.conv(lidar_bev)
            
            # 下采样到目标尺寸 (180, 180) 减少内存占用，使用adaptive_avg_pool确保尺寸一致
            print("mm_bev.shape",mm_bev.shape)
            batch_dict['spatial_features'] = mm_bev
            return batch_dict
        
        # 检查是否是多agent情况
        agent_keys = [k for k, v in batch_dict.items() if isinstance(v, dict) and ('spatial_features_img' in v)]
        
        if len(agent_keys) > 0:
            # 多agent情况：整合所有agent的spatial_features_img
            img_bev_list = []
            valid_agent_keys = []
            for agent_name in agent_keys:
                # 检查agent数据是否有效
                agent_dict = batch_dict[agent_name]
                if 'spatial_features_img' in agent_dict:
                    spatial_features_img = agent_dict['spatial_features_img']
                    # 检查特征是否有效（非空且非零）
                    if spatial_features_img is not None and spatial_features_img.numel() > 0 and torch.count_nonzero(spatial_features_img).item() > 0:
                        img_bev_list.append(spatial_features_img)
                        valid_agent_keys.append(agent_name)
            
            # 如果没有有效的agent，使用默认处理
            if len(valid_agent_keys) == 0:
                agent_keys = valid_agent_keys
            else:
                agent_keys = valid_agent_keys
                
                # 调试信息：检查所有agent的特征形状
                # # print(f"[ConvFuser] Multi-agent mode: {len(agent_keys)} agents")
        
            
            # 检查所有张量形状是否一致
            shapes = [img_bev.shape for img_bev in img_bev_list]
            if len(set(shapes)) > 1:
                # # print(f"[ConvFuser] Warning: Inconsistent shapes detected: {shapes}")
                # 如果形状不一致，尝试统一batch_size
                batch_sizes = [img_bev.shape[0] for img_bev in img_bev_list]
                max_batch_size = max(batch_sizes)
                # # print(f"[ConvFuser] Batch sizes: {batch_sizes}, using max: {max_batch_size}")
                
                # 统一所有agent的batch_size
                unified_img_bev_list = []
                for i, img_bev in enumerate(img_bev_list):
                    if img_bev.shape[0] < max_batch_size:
                        # 重复最后一个样本到目标batch_size
                        repeat_times = max_batch_size - img_bev.shape[0]
                        last_sample = img_bev[-1:].repeat(repeat_times, 1, 1, 1)
                        unified_img_bev = torch.cat([img_bev, last_sample], dim=0)
                        # # print(f"[ConvFuser] Agent {agent_keys[i]}: {img_bev.shape} -> {unified_img_bev.shape}")
                    else:
                        unified_img_bev = img_bev
                    unified_img_bev_list.append(unified_img_bev)
                
                img_bev_list = unified_img_bev_list
            
            # 【多agent特征融合策略】
            # 目标：将多个agent的spatial_features_img融合成单一特征
            # 输入：img_bev_list = [agent1_feat, agent2_feat, ...] 每个 [B_i, 80, H, W]
            # 输出：img_bev [B_max, 80, H, W] 与MambaFusion格式对齐
            
            if self.agent_fusion_strategy == 'mean':
                # 【平均融合策略】
                # 1. 堆叠：将所有agent特征堆叠到新维度 [B_max, N_agents, 80, H, W]
                # 2. 平均：在agent维度上取平均值 [B_max, 80, H, W]
                # 优势：保留所有agent信息，减少噪声
                img_bev = torch.stack(img_bev_list, dim=1)  # [B_max, N_agents, 80, H, W]
                img_bev = img_bev.mean(dim=1)  # [B_max, 80, H, W]
                # # print(f"[ConvFuser] Multi-agent mode (mean): {len(agent_keys)} agents, img_bev shape: {img_bev.shape}")
                
            elif self.agent_fusion_strategy == 'max':
                # 【最大融合策略】
                # 1. 堆叠：将所有agent特征堆叠到新维度 [B_max, N_agents, 80, H, W]
                # 2. 最大值：在agent维度上取最大值 [B_max, 80, H, W]
                # 优势：突出最强特征，适合互补场景
                img_bev = torch.stack(img_bev_list, dim=1)  # [B_max, N_agents, 80, H, W]
                img_bev = img_bev.max(dim=1)[0]  # [B_max, 80, H, W]
                # # print(f"[ConvFuser] Multi-agent mode (max): {len(agent_keys)} agents, img_bev shape: {img_bev.shape}")
                
            elif self.agent_fusion_strategy == 'concat':
                # 【通道拼接策略】
                # 1. 拼接：在通道维度上拼接所有agent特征 [B_max, N_agents*80, H, W]
                # 2. 注意：需要调整后续网络结构以适应增加的通道数
                # 优势：保留所有原始信息，但增加计算复杂度
                img_bev = torch.cat(img_bev_list, dim=1)  # [B_max, N_agents*80, H, W]
                # # print(f"[ConvFuser] Multi-agent mode (concat): {len(agent_keys)} agents, img_bev shape: {img_bev.shape}")
                
            else:
                raise ValueError(f"Unknown agent fusion strategy: {self.agent_fusion_strategy}")
        else:
            # 单agent情况：直接从batch_dict获取
            img_bev = batch_dict['spatial_features_img']
            # # print(f"[ConvFuser] Single-agent mode: img_bev shape: {img_bev.shape}")
        
        lidar_bev = batch_dict['spatial_features']
        # # print(f"[ConvFuser] lidar_bev shape: {lidar_bev.shape}")
        
        # 【Batch Size对齐】确保img_bev和lidar_bev的batch_size一致
        if img_bev.shape[0] != lidar_bev.shape[0]:
            # # print(f"[ConvFuser] Batch size mismatch: img_bev {img_bev.shape[0]} vs lidar_bev {lidar_bev.shape[0]}")
            # 使用较大的batch_size，将较小的张量扩展到相同大小
            max_batch_size = max(img_bev.shape[0], lidar_bev.shape[0])
            
            if img_bev.shape[0] < max_batch_size:
                # 扩展img_bev
                repeat_times = max_batch_size // img_bev.shape[0]
                img_bev = img_bev.repeat(repeat_times, 1, 1, 1)
                # # print(f"[ConvFuser] Extended img_bev to shape: {img_bev.shape}")
            
            if lidar_bev.shape[0] < max_batch_size:
                # 扩展lidar_bev
                repeat_times = max_batch_size // lidar_bev.shape[0]
                lidar_bev = lidar_bev.repeat(repeat_times, 1, 1, 1)
                # # print(f"[ConvFuser] Extended lidar_bev to shape: {lidar_bev.shape}")
        
        # 【空间尺寸对齐】确保img_bev和lidar_bev的空间尺寸一致
        if img_bev.shape[2:] != lidar_bev.shape[2:]:
            # # print(f"[ConvFuser] Spatial size mismatch: img_bev {img_bev.shape[2:]} vs lidar_bev {lidar_bev.shape[2:]}")
            # 使用lidar_bev的空间尺寸作为目标尺寸
            target_h, target_w = lidar_bev.shape[2], lidar_bev.shape[3]
            img_bev = F.interpolate(img_bev, size=(target_h, target_w), mode='bilinear', align_corners=False)
            # # print(f"[ConvFuser] Resized img_bev to shape: {img_bev.shape}")
        
        # 【AirV2X多agent融合策略】
        # 1. 输入：img_bev [B, 80, H, W] (多agent融合后的图像BEV特征)
        # 2. 输入：lidar_bev [B, 128, H, W] (激光雷达BEV特征)
        # 3. 目标：输出 [B, 128, H, W] 与MambaFusion对齐
        
        if self.use_vmamba:
            # 【VMamba融合】使用复杂的多尺度VMamba块进行融合
            if self.use_checkpoint:
                cat_bev = checkpoint.checkpoint(self.mamba_forward, img_bev, lidar_bev)
            else:
                cat_bev = self.mamba_forward(img_bev, lidar_bev)
        else:
            # 【简单拼接融合】直接在通道维度拼接两种模态
            # img_bev: [B, 80, H, W] + lidar_bev: [B, 128, H, W] = [B, 208, H, W]
            cat_bev = torch.cat([img_bev, lidar_bev], dim=1)
        
        # 【后处理融合】可选的额外融合层
        if self.use_merge_after:
            for block in self.merge_blocks:
                cat_bev = block(cat_bev)
        
        # 【最终卷积】将融合后的特征映射到目标通道数
        # 输入：cat_bev [B, 208, H, W] -> 输出：mm_bev [B, 128, H, W]
        # 与MambaFusion保持一致：spatial_features shape = [B, 128, H, W]
        mm_bev = self.conv(cat_bev)
        
        # 现在MAP_TO_BEV输出已经是(180, 180)，不需要额外下采样
        print(f"[ConvFuser] mm_bev shape: {mm_bev.shape}")
        # 调试信息：打印最终输出shape，确保与MambaFusion对齐
        # # print(f"[ConvFuser] Final spatial_features shape: {mm_bev.shape}")

        batch_dict['spatial_features'] = mm_bev
        return batch_dict

    def mamba_forward(self, img_bev, lidar_bev):
        ups_img = []
        ups_img.append(img_bev)
        ups_lidar = []
        ups_lidar.append(lidar_bev)
        for i, (block_img, block_lidar) in enumerate(zip(self.image_vmamba_blocks, self.point_vmamba_blocks)):
            img_bev = self.image_down_blocks[i](img_bev) # [2, 80, 90, 90]
            img_bev = block_img(img_bev)
            lidar_bev = self.lidar_down_blocks[i](lidar_bev)
            lidar_bev = block_lidar(lidar_bev)
            if self.use_cross:
                img_bev = self.image_up_blocks[i](img_bev) # [batch_size, 128, 180, 180]
                img_bev_cross = self.cross_vmamba_blocks[i].blocks1((img_bev, lidar_bev)) # [batch_size, 128, 180, 180]
                lidar_bev_cross = self.cross_vmamba_blocks[i].blocks2((lidar_bev, img_bev))
                if not self.use_res_merge:
                    img_bev = self.image_cross_blocks[i](torch.cat([img_bev, lidar_bev_cross], dim=1)) # [batch_size, 128, 180, 180]
                    lidar_bev = self.point_cross_blocks[i](torch.cat([lidar_bev, img_bev_cross], dim=1)) # [batch_size, 128, 180, 180]
                else:
                    img_bev = img_bev_cross
                    lidar_bev = lidar_bev_cross
            if self.use_res_merge:
                img_bev = self.image_norm[i](img_bev + self.image_de_blocks[i](img_bev))
                lidar_bev = self.point_norm[i](lidar_bev + self.lidar_de_blocks[i](lidar_bev))
            else:
                ups_img.append(self.image_de_blocks[i](img_bev))
                ups_lidar.append(self.lidar_de_blocks[i](lidar_bev))
        if self.use_res_merge:
            merge_img = img_bev
            merge_lidar = lidar_bev
        else:
            merge_img = self.image_conv(torch.cat(ups_img, dim=1)) # [2, 80, 360, 360]
            merge_lidar = self.lidar_conv(torch.cat(ups_lidar, dim=1)) # [2, 128, 360, 360]
        cat_bev = torch.cat([merge_img,merge_lidar],dim=1)

        return cat_bev

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x