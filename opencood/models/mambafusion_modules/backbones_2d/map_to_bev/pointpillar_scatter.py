import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

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

    def forward(self, batch_dict, **kwargs):
        # Merge pillar features from multiple agents if present
        if not ('pillar_features' in batch_dict and 'voxel_coords' in batch_dict):
            fused_pillars = []
            fused_coords = []
            # collect agent sub-dicts that carry pillars/coords
            for k, v in batch_dict.items():
                if isinstance(v, dict) and ('pillar_features' in v) and ('voxel_coords' in v):
                    fused_pillars.append(v['pillar_features'])
                    fused_coords.append(v['voxel_coords'])
            if len(fused_pillars) == 0:
                raise KeyError('No pillar_features/voxel_coords found in batch_dict or agent sub-dicts')
            pillar_features = torch.cat(fused_pillars, dim=0)
            coords = torch.cat(fused_coords, dim=0)
            batch_dict['pillar_features'] = pillar_features
            batch_dict['voxel_coords'] = coords
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] # [num_of_voxel, 128] [num_of_voxel, 4] batch_idx, z, y, x
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
        
        # 将360x360池化到180x180以减少内存占用
        if self.ny == 360 and self.nx == 360:
            batch_spatial_features = F.adaptive_avg_pool2d(batch_spatial_features, (180, 180))
            print(f"[PointPillarScatter] 池化后尺寸: {batch_spatial_features.shape}")
        
        batch_dict['spatial_features'] = batch_spatial_features # torch.Size([2, 128, 180, 180])
        return batch_dict
