from opencood.models.mambafusion_modules.detector3d_template import Detector3DTemplate
from opencood.models.mambafusion_modules import backbones_image, view_transforms, mm_backbone #get
from opencood.models.mambafusion_modules.backbones_image import img_neck #get 
from opencood.models.mambafusion_modules.backbones_2d import fuser #get
from opencood.models.mambafusion_modules.spconv_utils import find_all_spconv_keys #get
from opencood.models.mambafusion_modules.vmamba import build_vssm_model #ge
import torch.profiler
import torch.nn.functional as F
from easydict import EasyDict
class Airv2xMambafusion(Detector3DTemplate):
    def __init__(self, model_cfg, dataset, num_class = 7):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.model_cfg = EasyDict(self.model_cfg)
        self.use_voxel_mamba = self.model_cfg.get('USE_VOXEL_MAMBA', False)
        self.new_order = self.model_cfg.VTRANSFORM.get('USE_MAMBA', False)
        self.agent = ['vehicle','drone','rsu']
        if self.use_voxel_mamba:
            self.module_topology = [
                'vfe', 'backbone_3d', 'mm_backbone', 'map_to_bev_module', 
                'neck','vtransform', 'fuser',
                'backbone_2d','dense_head',
            ]
            if self.new_order:
                self.module_topology = [
                    'vfe', 'backbone_3d', 'mm_backbone', 
                    'neck','vtransform', 'map_to_bev_module', 'fuser',
                    'backbone_2d'
                ]
        else:
            self.module_topology = [
                'vfe','mm_backbone', 'map_to_bev_module', 
                'neck','vtransform', 'fuser',
                'backbone_2d','dense_head',
            ]
            if self.new_order:
                self.module_topology = [
                    'vfe','mm_backbone', 
                    'neck','vtransform', 'map_to_bev_module', 'fuser',
                    'backbone_2d','dense_head',
                ]
        num_anchors = 2  # 从配置文件 anchor_args.num = 2
        num_classes = 7  # 从配置文件 num_class = 7（与Where2Comm一致）
        C = 512
        self.cls_head = torch.nn.Conv2d(C, num_anchors * num_classes, kernel_size=1)
        self.reg_head = torch.nn.Conv2d(C, 7 * num_anchors, kernel_size=1)
        self.obj_head = torch.nn.Conv2d(C, num_anchors, kernel_size=1)
        self.module_list = self.build_networks()
        self.time_list = []
        
        # 打印各模块参数量
        self.print_module_params()
    
    def print_module_params(self):
        """打印各模块的参数量"""
        print("=" * 80)
        print("AirV2X MambaFusion 模型参数量分析")
        print("=" * 80)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print()
        
        # 按模块统计参数量
        module_params = {}
        
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    # 提取模块名称（去掉具体层名）
                    module_name = name.split('.')[0] if '.' in name else name
                    if module_name not in module_params:
                        module_params[module_name] = 0
                    module_params[module_name] += param_count
        
        # 按参数量排序
        sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
        
        print("各模块参数量 (按参数量排序):")
        print("-" * 50)
        for module_name, param_count in sorted_modules:
            percentage = (param_count / total_params) * 100
            print(f"{module_name:25s}: {param_count:10,} ({percentage:5.1f}%)")
        
        print()
        print("模块拓扑结构参数量:")
        print("-" * 50)
        for i, module_name in enumerate(self.module_topology):
            param_count = module_params.get(module_name, 0)
            percentage = (param_count / total_params) * 100
            print(f"{i+1:2d}. {module_name:20s}: {param_count:10,} ({percentage:5.1f}%)")
        
        print()
        print("检测头参数量:")
        print("-" * 50)
        cls_params = sum(p.numel() for p in self.cls_head.parameters())
        reg_params = sum(p.numel() for p in self.reg_head.parameters())
        obj_params = sum(p.numel() for p in self.obj_head.parameters())
        head_total = cls_params + reg_params + obj_params
        
        print(f"cls_head: {cls_params:10,}")
        print(f"reg_head: {reg_params:10,}")
        print(f"obj_head: {obj_params:10,}")
        print(f"检测头总计: {head_total:10,} ({(head_total/total_params)*100:5.1f}%)")
        
        print("=" * 80)
       
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict
    
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict

    def build_mm_backbone(self, model_info_dict):
        if self.model_cfg.get('MM_BACKBONE', None) is None:
            return None, model_info_dict
        mm_backbone_name = self.model_cfg.MM_BACKBONE.NAME
        del self.model_cfg.MM_BACKBONE['NAME']
        mm_backbone_module = mm_backbone.__all__[mm_backbone_name](
            model_cfg=self.model_cfg.MM_BACKBONE
            )
        model_info_dict['module_list'].append(mm_backbone_module)

        return mm_backbone_module, model_info_dict
    
    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()
            # adapt pretrain image backbone to mm backbone
            if 'image_backbone' in key:
                key = key.replace("image","mm")
                if 'input_layer' in key:
                    key = key.replace("input_layer","image_input_layer")


            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print("not exist",key)
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def forward(self, batch_dict): 
        available_agents = []
        for agent in self.agent:
            if agent == 'vehicle' and 'origin_lidar' in batch_dict:
                # 检查vehicle的origin_lidar是否有效
                origin_lidar = batch_dict['origin_lidar']
                if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                    available_agents.append(agent)
            elif agent != 'vehicle' and f'origin_lidar_{agent}' in batch_dict:
                # 检查RSU/Drone的origin_lidar是否有效
                origin_lidar = batch_dict[f'origin_lidar_{agent}']
                if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                    available_agents.append(agent)
        
        # AirV2X需要agent循环处理，但要保持数据流一致
        print("available_agents:",available_agents)
        for cur_module, model_name in zip(self.module_list, self.module_topology):
            if model_name in ['vfe', 'backbone_3d', 'mm_backbone', 'map_to_bev_module','vtransform']:
                # 这些模块需要agent参数，但只处理有效的agent
                for agent in available_agents:
                    batch_dict = cur_module(batch_dict, agent)
            elif model_name in ['fuser']:
                batch_dict = cur_module(batch_dict, available_agents)
            else:
                # 其他模块不需要agent参数
                batch_dict = cur_module(batch_dict)

        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

        #     ret_dict = {
        #         'loss': loss
        #     }
        #     return ret_dict, tb_dict, disp_dict
        # else:
            # 输出BEV特征格式，让AirV2X的post_process处理
            # 获取fuser输出的BEV特征
        spatial_features = batch_dict.get('spatial_features_2d', None)  # [B, C, H, W]
        
        if spatial_features is not None:
            B, C, H, W = spatial_features.shape
            
            # 通过头部网络得到最终-output
            psm = self.cls_head(spatial_features)  # [B, A*C, H, W] = [B, 2*7, 180, 180]
            rm = self.reg_head(spatial_features)   # [B, A*7, H, W] = [B, 2*7, 180, 180]
            obj = self.obj_head(spatial_features)  # [B, A, H, W] = [B, 2, 180, 180]
            
            # 输出尺寸已经匹配，不需要调整
            print(f"[Airv2xMambafusion] 输出尺寸: psm={psm.shape}, rm={rm.shape}, obj={obj.shape}")
            
        else:
            # 如果没有特征，创建空的特征
            # 使用默认尺寸 H=100, W=352 (考虑feature_stride=2)
            default_H, default_W = 100, 352
            psm = torch.zeros(1, 14, default_H, default_W, device=next(self.parameters()).device)  # 2*7=14
            rm = torch.zeros(1, 14, default_H, default_W, device=next(self.parameters()).device)   # 2*7=14
            obj = torch.zeros(1, 2, default_H, default_W, device=next(self.parameters()).device)   # 2
        
        # 创建AirV2X兼容的输出格式
        output_dict = {
            'psm': psm,                  # [1, A*C, H, W] - 分类特征
            'rm': rm,                    # [1, A*7, H, W] - 回归特征  
            'obj': obj,                  # [1, A, H, W] - 目标特征
            'mask': 0,                   # 占位符
            'com': None,                 # 通信相关
            'comm_rate': None            # 通信率
        }
            
        return output_dict

    def get_training_loss(self, batch_dict):
        """
        使用MambaFusion标准的loss计算方法
        """
        disp_dict = {}
        
        # 使用dense_head的loss计算方法
        if hasattr(self, 'dense_head') and self.dense_head is not None:
            loss, tb_dict = self.dense_head.get_loss()
        else:
            # 如果没有dense_head，使用默认的loss计算
            loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
            tb_dict = {
                'loss_trans': loss_trans.item(),
                **tb_dict
            }
            loss = loss_trans
            
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
    
