import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ..utils import loss_utils
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils


class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict



class TransFusionHead(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(TransFusionHead, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class

        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')
        # self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'Custom')

        hidden_channel=self.model_cfg.HIDDEN_CHANNEL
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.bn_momentum = self.model_cfg.BN_MOMENTUM
        self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
        self.post_center_range_cuda = None

        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']
        if self.model_cfg.LOSS_CONFIG.get('LOSS_IOU_REG', False):
            self.loss_iou_reg = True
            self.loss_iou_reg_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_reg_weight']
        if self.model_cfg.LOSS_CONFIG.get('LOSS_IOU', False):
            self.loss_iou = True
            self.loss_iou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']

        self.code_size = 10 #TODO

        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=num_class,kernel_size=3,padding=1))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.forward_ret_dict = {}

        # NMS - 临时禁用NMS确保流程跑通
        self.nms_cfg = None  # self.model_cfg.get("NMS_CONFIG",None)
        # query local feature around proposals in Cross-Attention
        self.query_local = self.model_cfg.get("QUERY_LOCAL",False)
        if self.query_local:
            self.query_radius = self.model_cfg.QUERY_RADIUS
            self.query_range = torch.arange(-self.query_radius, self.query_radius+1)
            self.query_r_coor_x, self.query_r_coor_y = torch.meshgrid(self.query_range, self.query_range) 

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def predict(self, inputs):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs) # torch.Size([batch_size, 512, 180, 180]) -> torch.Size([batch_size, 128, 180, 180])

        lidar_feat_flatten = lidar_feat.view( # torch.Size([batch_size, 128, 180, 180]) -> torch.Size([batch_size, 128, 32400])
            batch_size, lidar_feat.shape[1], -1
        )
        if str(self.bev_pos.device) == 'cpu':
            self.bev_pos = self.bev_pos.to(lidar_feat.device) # torch.Size([1, 32400, 2])
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1) # torch.Size([batch_size, 32400, 2])

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat) # torch.Size([2, 10, 180, 180])
        heatmap = dense_heatmap.detach().sigmoid() # torch.Size([2, 10, 180, 180])
        padding = self.nms_kernel_size // 2 # 1
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0 # torch.Size([2, 10, 180, 180]) kernel_size=3, stride=1, padding=1 -> torch.Size([2, 10, 178, 178])
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner # 相当于把local_max_inner的结果放到local_max的中间
        # for Pedestrian & Traffic_cone in nuScenes
        if self.dataset_name == "nuScenes":
            local_max[ :, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0) # 最大池化操作实际上只是从每个通道中选择最大的像素值，并将其存储
            local_max[ :, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        # for Pedestrian & Cyclist in Waymo
        elif self.dataset_name == "Waymo":
            local_max[ :, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[ :, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max) # torch.Size([2, 10, 180, 180]) 对于每个通道，只有最大值的位置才会保留，其他位置都会置为0
        x_grid, y_grid = heatmap.shape[-2:] # torch.Size([10, 32400])
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1) # torch.Size([2, 10, 32400])
 
        # top num_proposals among all classes 
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[ # 先reshape得到torch.Size([2, 324000])  torch.Size([2, 200])
            ..., : self.num_proposals
        ]
        top_proposals_class = top_proposals // heatmap.shape[-1] # torch.Size([2, 200]) 得到的是top_proposals的类别
        top_proposals_index = top_proposals % heatmap.shape[-1] # torch.Size([2, 200]) 得到的是top_proposals的索引
        query_feat = lidar_feat_flatten.gather( # 从lidar_feat_flatten中取出top_proposals_index对应的特征
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        ) # torch.Size([2, 128, 200])
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1) # torch.Size([2, 10, 200])
        
        query_cat_encoding = self.class_encoding(one_hot.float()) # torch.Size([2, 128, 200])
        query_feat += query_cat_encoding # torch.Size([2, 128, 200])

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        ) # torch.Size([2, 200, 2])
        # convert to xy
        query_pos = query_pos.flip(dims=[-1]) # torch.Size([2, 200, 2])
        bev_pos = bev_pos.flip(dims=[-1]) # torch.Size([2, 32400, 2])
        
        if self.query_local: # True 目的是为了提高效率，只在top_proposals周围的区域进行查询
            # compute local key 
            top_proposals_x = top_proposals_index // x_grid # bs, num_proposals 
            top_proposals_y = top_proposals_index % y_grid # bs, num_proposals
            
            # bs, num_proposal, radius * 2 + 1, radius * 2 + 1
            if str(self.query_r_coor_x.device) == 'cpu':
                self.query_r_coor_x = self.query_r_coor_x.to(top_proposals.device)
                self.query_r_coor_y = self.query_r_coor_y.to(top_proposals.device)
            top_proposals_key_x = top_proposals_x[:, :, None, None] + self.query_r_coor_x[None, None, :, :] # bs, num_proposals, radius * 2 + 1, radius * 2 + 1
            top_proposals_key_y = top_proposals_y[:, :, None, None] + self.query_r_coor_y[None, None, :, :] # bs, num_proposals, radius * 2 + 1, radius * 2 + 1
            top_proposals_key_x = torch.clamp(top_proposals_key_x, min=0, max=x_grid-1) # 限制在0到x_grid-1之间 0 -> 179
            top_proposals_key_y = torch.clamp(top_proposals_key_y, min=0, max=y_grid-1)
            # bs, num_proposals, key_num
            top_proposals_key_index = top_proposals_key_x.view(batch_size, top_proposals_key_x.shape[1], -1) * x_grid + \
                                                    top_proposals_key_y.view(batch_size, top_proposals_key_y.shape[1], -1) # torch.Size([2, 200, 41*41])
            key_mask = (top_proposals_key_index < 0) + (top_proposals_key_index >= (x_grid * y_grid)) # mask掉超出范围的索引
            top_proposals_key_index = torch.clamp(top_proposals_key_index, min=0, max=x_grid * y_grid-1) # bs, num_proposals, key_num 
            num_proposals = top_proposals_key_index.shape[1] # 200
            # bs, feat_dim, num_proposals * key_num
            key_feat = lidar_feat_flatten.gather( # 从lidar_feat_flatten中取出top_proposals_key_index对应的特征
                index=top_proposals_key_index.view(batch_size, 1, -1).expand(
                    -1, lidar_feat_flatten.shape[1], -1),
                dim=-1,)  # torch.Size([2, 128, 200*41*41]) 
            # bs, feat_dim, num_proposals, key_num
            key_feat = key_feat.view(batch_size, lidar_feat_flatten.shape[1], num_proposals, -1) # torch.Size([2, 128, 200, 41*41])
            key_pos = bev_pos.gather(
                index=top_proposals_key_index.view(batch_size, 1, -1)
                .permute(0, 2, 1)
                .expand(-1, -1, bev_pos.shape[-1]),
                dim=1,) # torch.Size([2, 200*41*41, 2])
            # bs, num_proposals, key_num, 2
            key_pos = key_pos.view(batch_size, num_proposals, -1, bev_pos.shape[-1]) # torch.Size([2, 200, 1681, 2])
            # bs, num_proposals, feat_dim, key_num
            key_feat = key_feat.permute(0, 2, 1, 3).reshape(batch_size*num_proposals, lidar_feat_flatten.shape[1], -1) # torch.Size([400, 128, 1681])
            # bs, num_proposals, key_num, 2
            key_pos = key_pos.view(-1, key_pos.shape[2], key_pos.shape[-1])  # torch.Size([400, 1681, 2])
            key_padding_mask = key_mask.view(-1, key_mask.shape[-1]) # torch.Size([400, 1681])
            # bs, num_proposals, feat_dim, 1
            query_feat_T = query_feat.permute(0, 2, 1).reshape(batch_size*num_proposals, -1, 1) # torch.Size([400, 128, 1])
            # bs, num_proposals, 1, 2
            query_pos_T = query_pos.view(-1, 1, query_pos.shape[-1]) # torch.Size([400, 1, 2])
            query_feat_T = self.decoder( # torch.Size([400, 128, 1])
                query_feat_T, key_feat, query_pos_T, key_pos, key_padding_mask=key_padding_mask
            ) #            query,                  key,                  query_pos,               key_pos,              key_padding_mask=None, attn_mask=None
            # torch.Size([400, 128, 1]) torch.Size([400, 128, 1681]) torch.Size([400, 1, 2]) torch.Size([400, 1681, 2]) torch.Size([400, 1681])
            query_feat = query_feat_T.reshape(batch_size, num_proposals, -1).permute(0, 2, 1) # torch.Size([2, 128, 200])
        else:
            query_feat = self.decoder(
                query_feat, lidar_feat_flatten, query_pos, bev_pos
            )
        res_layer = self.prediction_head(query_feat) # dict_keys(['center', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap'])
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1) # torch.Size([2, 2, 200])

        res_layer["query_heatmap_score"] = heatmap.gather( # 按照top_proposals_index的索引取出heatmap的值
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        ) # torch.Size([2, 10, 200])
        res_layer["dense_heatmap"] = dense_heatmap # torch.Size([2, 10, 180, 180])

        return res_layer

    def forward(self, batch_dict):
        feats = batch_dict['spatial_features_2d'] # torch.Size([batch_size, 512, 180, 180]) 这个是lidar融合了image的特征

        res = self.predict(feats)
        if not self.training:
            bboxes = self.get_bboxes(res)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes'] # torch.Size([2, 71, 10])
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d =  gt_boxes[...,-1].long() - 1
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res)
            if 'loss_global_align' in batch_dict.keys():
                loss += batch_dict['loss_global_align']
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap
        

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        
        num_proposals = preds_dict["center"].shape[-1]
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]["pred_boxes"]
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)

        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
            score, self.point_cloud_range,
        )
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.code_size], device=center.device)
        bbox_weights = torch.zeros([num_proposals, self.code_size], device=center.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # compute dense heatmap targets
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = (self.grid_size[:2] // self.feature_map_stride) 
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)


        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None])

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER if head_name != 'iou'], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        if "iou" in pred_dicts.keys() or self.model_cfg.LOSS_CONFIG.get('LOSS_IOU_REG', False):
            bbox_targets_for_iou = bbox_targets.permute(0, 2, 1)
            batch_box_targets_for_iou = self.decode_bbox_from_pred(bbox_targets_for_iou[:, 6:8, :].clone(),
                                        bbox_targets_for_iou[:, 3:6, :].clone(), bbox_targets_for_iou[:, 0:2, :].clone(), 
                                        bbox_targets_for_iou[:, 2:3, :].clone()) # (B, Proposal, 7)
            center = pred_dicts["center"][...,: self.num_proposals]
            height = pred_dicts["height"][..., : self.num_proposals]
            rot = pred_dicts["rot"][..., : self.num_proposals]
            dim = pred_dicts["dim"][..., : self.num_proposals]
            
            batch_box_preds = self.decode_bbox_from_pred(rot.clone(), dim.clone(),
                                    center.clone(), height.clone()) # (B, Proposal, 7)

            if "iou" in pred_dicts.keys():
                batch_box_preds_for_iou = batch_box_preds.clone().detach() # (B, Proposal, 7)
                batch_box_targets_for_iou = batch_box_targets_for_iou.detach() # (B, Proposal, 7)
                iou_loss = loss_utils.calculate_iou_loss_transfusionhead(
                    iou_preds=pred_dicts['iou'],  # (B, 1, Proposal)
                    batch_box_preds=batch_box_preds_for_iou,
                    gt_boxes=batch_box_targets_for_iou,
                    weights=bbox_weights,
                    num_pos=num_pos
                )
                loss_all += (iou_loss * self.loss_iou_weight)
                loss_dict["loss_iou"] = iou_loss.item() * self.loss_iou_weight

            if self.model_cfg.LOSS_CONFIG.get('LOSS_IOU_REG', False):
                iou_reg_loss = loss_utils.calculate_iou_reg_loss_transfusionhead(
                    batch_box_preds=batch_box_preds, gt_boxes=batch_box_targets_for_iou,
                    weights=bbox_weights[:, :, :7], num_pos=num_pos)
                loss_all += (iou_reg_loss * self.loss_iou_reg_weight)
                loss_dict["loss_iou_reg"] = iou_reg_loss.item() * self.loss_iou_reg_weight

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all

        return loss_all,loss_dict

    def encode_bbox(self, bboxes):
        code_size = 10 #TODO
        targets = torch.zeros([bboxes.shape[0], code_size], device=bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])#TODO
        if code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:]
        return targets
    
    def decode_bbox_acc(self, heatmap, rot, dim, center, height, vel, filter=False):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        # Assume self.post_center_range is already moved to GPU in class initialization
        if self.post_center_range_cuda is None:
            # 获取当前张量设备，如果 heatmap 在 GPU 上，则将 post_center_range 移动到同一设备
            post_center_range = post_process_cfg.POST_CENTER_RANGE
            device = heatmap.device
            self.post_center_range_cuda = torch.tensor(post_center_range, device=device).float()
        post_center_range = self.post_center_range_cuda

        # Class label
        final_scores, final_preds = heatmap.max(1)
        
        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        if not filter:
            predictions_dicts = [{
                'pred_boxes': final_box_preds[i],
                'pred_scores': final_scores[i],
                'pred_labels': final_preds[i]
            } for i in range(heatmap.shape[0])]
            return predictions_dicts

        # Vectorized operations without using torch.all()
        mask = (
            (final_box_preds[..., 0] >= post_center_range[0]) & (final_box_preds[..., 0] <= post_center_range[3]) &
            (final_box_preds[..., 1] >= post_center_range[1]) & (final_box_preds[..., 1] <= post_center_range[4]) &
            (final_box_preds[..., 2] >= post_center_range[2]) & (final_box_preds[..., 2] <= post_center_range[5])
        )

        thresh_mask = final_scores > score_thresh
        mask &= thresh_mask

        predictions_dicts = [{
            'pred_boxes': final_box_preds[i][mask[i]],
            'pred_scores': final_scores[i][mask[i]],
            'pred_labels': final_preds[i][mask[i]],
            'cmask': mask[i],
        } for i in range(heatmap.shape[0])]

        return predictions_dicts

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        # 确保post_center_range与heatmap在同一个设备上
        post_center_range = torch.tensor(post_center_range).float().to(heatmap.device)
        # post_center_range = torch.tensor(post_center_range).cuda().float()
        # post_center_range = torch.tensor(post_center_range).cuda().float()
        # 在第一次调用时，将 post_center_range 移动到当前的 GPU 上
        # if self.post_center_range_cuda is None:
        #     # 获取当前张量设备，如果 heatmap 在 GPU 上，则将 post_center_range 移动到同一设备
        #     post_center_range = post_process_cfg.POST_CENTER_RANGE
        #     device = heatmap.device
        #     self.post_center_range_cuda = torch.tensor(post_center_range, device=device).float()
        # post_center_range = self.post_center_range_cuda
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices # [1, 200]
        final_scores = heatmap.max(1, keepdims=False).values # [1, 200]

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh        
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
        
        # 【调试信息】打印检测统计
        print(f"[TransFusionHead] 总proposals: {final_scores.shape[1]}")
        print(f"[TransFusionHead] 分数范围: [{final_scores.min():.4f}, {final_scores.max():.4f}]")
        print(f"[TransFusionHead] 分数阈值: {score_thresh}")
        print(f"[TransFusionHead] 超过分数阈值的数量: {thresh_mask.sum()}")
        print(f"[TransFusionHead] 位置范围: {post_center_range}")
        print(f"[TransFusionHead] 位置在范围内的数量: {mask.sum()}")
        print(f"[TransFusionHead] 最终保留的数量: {(mask & thresh_mask).sum()}")

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
                'cmask':cmask,
            }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def decode_bbox_from_pred(self, rot, dim, center, height, head_index=None):
        # change size to real world metric
        if head_index is not None:
            center[:, 0, :] = center[:, 0, :] * self.feature_map_stride[head_index] * self.voxel_size[0] + self.point_cloud_range[0]
            center[:, 1, :] = center[:, 1, :] * self.feature_map_stride[head_index] * self.voxel_size[1] + self.point_cloud_range[1]
        else:
            center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)
        final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        return final_box_preds

    def get_bboxes(self, preds_dicts):

        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid()
        one_hot = F.one_hot(
            self.query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
        batch_center = preds_dicts["center"]
        batch_height = preds_dicts["height"]
        batch_dim = preds_dicts["dim"]
        batch_rot = preds_dicts["rot"]
        batch_vel = None
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]
        batch_iou = (preds_dicts['iou'] + 1) * 0.5 if 'iou' in preds_dicts else None
        # ret_dict_new = self.decode_bbox_acc(
        #     batch_score.clone(), batch_rot.clone(), batch_dim.clone(),
        #     batch_center.clone(), batch_height.clone(), batch_vel.clone(),
        #     filter=True,
        # )
        # ret_dict_old = self.decode_bbox(
        #     batch_score.clone(), batch_rot.clone(), batch_dim.clone(),
        #     batch_center.clone(), batch_height.clone(), batch_vel.clone(),
        #     filter=True,
        # )
        # for key in ret_dict_new[0].keys():
        #     for i in range(batch_size):
        #         assert torch.allclose(ret_dict_new[i][key], ret_dict_old[i][key], atol=1e-3)
        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim,
            batch_center, batch_height, batch_vel,
            filter=True,
        )
        if self.dataset_name == "nuScenes":
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=["pedestrian"],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=["traffic_cone"],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
        elif self.dataset_name == "Waymo":
            self.tasks = [
                dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                dict(
                    num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7
                ),
                dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
            ]
        if self.dataset_name not in ("nuScenes", "Waymo"):
            self.tasks = [dict(num_class=1, class_names=["object"], indices=[0], radius=-1)]

        for i in range(batch_size):
            boxes3d = ret_dict[i]["pred_boxes"]
            scores = ret_dict[i]["pred_scores"]
            labels = ret_dict[i]["pred_labels"]
            cmask = ret_dict[i]['cmask']

            # IOU refine 
            if self.model_cfg.POST_PROCESSING.get('USE_IOU_TO_RECTIFY_SCORE', False) and batch_iou is not None:
                pred_iou = torch.clamp(batch_iou[i][0][cmask], min=0, max=1.0)
                IOU_RECTIFIER = scores.new_tensor(self.model_cfg.POST_PROCESSING.IOU_RECTIFIER)
                if len(IOU_RECTIFIER) == 1:
                    IOU_RECTIFIER = IOU_RECTIFIER.repeat(self.num_classes)
                scores = torch.pow(scores, 1 - IOU_RECTIFIER[labels]) * torch.pow(pred_iou, IOU_RECTIFIER[labels])
            

            if self.nms_cfg != None:
                print(f"[TransFusionHead NMS] 输入检测框数量: {len(boxes3d)}")
                print(f"[TransFusionHead NMS] 输入分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
                print(f"[TransFusionHead NMS] 输入标签: {torch.unique(labels)}")
                print(f"[TransFusionHead NMS] NMS配置: {self.nms_cfg}")
                print(f"[TransFusionHead NMS] Tasks: {self.tasks}")
                
                keep_mask = torch.zeros_like(scores)
                for task in self.tasks:
                    task_mask = torch.zeros_like(scores)
                    for cls_idx in task["indices"]:
                        task_mask += labels == cls_idx
                    task_mask = task_mask.bool()
                    print(f"[TransFusionHead NMS] Task {task['class_names']}: 匹配数量 {task_mask.sum()}")
                    
                    if task["radius"] > 0:
                        top_scores = scores[task_mask]
                        boxes_for_nms = boxes3d[task_mask][:, :7].clone().detach()
                        task_nms_config = copy.deepcopy(self.nms_cfg)
                        task_nms_config.NMS_THRESH = task["radius"]
                        print(f"[TransFusionHead NMS] 执行NMS: 输入{len(top_scores)}个, 阈值{task['radius']}")
                        task_keep_indices, _ = model_nms_utils.class_agnostic_nms(
                                box_scores=top_scores, box_preds=boxes_for_nms,
                                nms_config=task_nms_config, score_thresh=task_nms_config.SCORE_THRES)
                        print(f"[TransFusionHead NMS] NMS后保留: {len(task_keep_indices)}个")
                    else:
                        task_keep_indices = torch.arange(task_mask.sum())
                        print(f"[TransFusionHead NMS] 跳过NMS: 保留{len(task_keep_indices)}个")
                    if task_keep_indices.shape[0] != 0:
                        keep_indices = torch.where(task_mask != 0)[0][
                            task_keep_indices
                        ]
                        keep_mask[keep_indices] = 1
                keep_mask = keep_mask.bool()
                print(f"[TransFusionHead NMS] 最终保留: {keep_mask.sum()}个")
                ret_dict[i]['pred_boxes'] = boxes3d[keep_mask]
                ret_dict[i]['pred_scores'] = scores[keep_mask]
                ret_dict[i]['pred_labels'] = labels[keep_mask].int() + 1
            else:  
                # no nms
                ret_dict[i]['pred_labels'] = ret_dict[i]['pred_labels'].int() + 1


        return ret_dict 
    