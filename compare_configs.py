#!/usr/bin/env python3
import yaml
import sys
sys.path.append('.')

def compare_configs():
    # 加载两个配置文件
    with open('opencood/hypes_yaml/airv2x/camera_lidar/mambafusion.yaml', 'r') as f:
        original_config = yaml.safe_load(f)
    
    with open('AirV2X-Perception-Checkpoints/airv2x_intermediate_mambafusion/config.yaml', 'r') as f:
        airv2x_config = yaml.safe_load(f)
    
    print("=" * 80)
    print("MambaFusion 配置对比分析")
    print("=" * 80)
    
    # 对比关键配置
    def compare_section(original, airv2x, section_name, key_path=""):
        print(f"\n{section_name}:")
        print("-" * 50)
        
        if key_path:
            orig_section = original
            airv2x_section = airv2x
            for key in key_path.split('.'):
                orig_section = orig_section.get(key, {})
                airv2x_section = airv2x_section.get(key, {})
        else:
            orig_section = original
            airv2x_section = airv2x
        
        # 对比关键参数
        key_params = [
            'LAYER_DIM', 'NUM_LAYERS', 'DEPTHS', 'GROUP_SIZE', 'WINDOW_SHAPE',
            'NUM_PROPOSALS', 'HIDDEN_CHANNEL', 'NUM_CLASSES', 'NUM_HEADS', 'FFN_CHANNEL',
            'LAYER_NUMS', 'LAYER_STRIDES', 'NUM_FILTERS', 'NUM_UPSAMPLE_FILTERS',
            'IN_CHANNELS', 'OUT_CHANNELS', 'NUM_OUTS'
        ]
        
        for param in key_params:
            orig_val = orig_section.get(param, "N/A")
            airv2x_val = airv2x_section.get(param, "N/A")
            
            if orig_val != airv2x_val:
                print(f"  {param:20s}: 原始={orig_val} | AirV2X={airv2x_val}")
    
    # 对比各个模块
    compare_section(original_config, airv2x_config, "BACKBONE_3D", "model.args.BACKBONE_3D")
    compare_section(original_config, airv2x_config, "MM_BACKBONE", "model.args.MM_BACKBONE")
    compare_section(original_config, airv2x_config, "NECK", "model.args.NECK")
    compare_section(original_config, airv2x_config, "BACKBONE_2D", "model.args.BACKBONE_2D")
    compare_section(original_config, airv2x_config, "DENSE_HEAD", "model.args.DENSE_HEAD")
    
    print("\n" + "=" * 80)
    print("关键差异总结:")
    print("=" * 80)
    
    # 分析关键差异
    orig_backbone3d = original_config['model']['args']['BACKBONE_3D']
    airv2x_backbone3d = airv2x_config['model']['args']['BACKBONE_3D']
    
    orig_mm_backbone = original_config['model']['args']['MM_BACKBONE']
    airv2x_mm_backbone = airv2x_config['model']['args']['MM_BACKBONE']
    
    orig_backbone2d = original_config['model']['args']['BACKBONE_2D']
    airv2x_backbone2d = airv2x_config['model']['args']['BACKBONE_2D']
    
    orig_dense_head = original_config['model']['args']['DENSE_HEAD']
    airv2x_dense_head = airv2x_config['model']['args']['DENSE_HEAD']
    
    print("1. BACKBONE_3D 差异:")
    print(f"   WINDOW_SHAPE: 原始={orig_backbone3d['WINDOW_SHAPE']} | AirV2X={airv2x_backbone3d['WINDOW_SHAPE']}")
    
    print("\n2. MM_BACKBONE 差异:")
    print(f"   AGENT_INTERACT: 原始=无 | AirV2X={airv2x_mm_backbone.get('AGENT_INTERACT', 'N/A')}")
    print(f"   AGENTS_AS_VIEWS: 原始=无 | AirV2X={airv2x_mm_backbone.get('AGENTS_AS_VIEWS', 'N/A')}")
    print(f"   IMAGE_INPUT_LAYER.sparse_shape: 原始={orig_mm_backbone['IMAGE_INPUT_LAYER']['sparse_shape']} | AirV2X={airv2x_mm_backbone['IMAGE_INPUT_LAYER']['sparse_shape']}")
    
    print("\n3. BACKBONE_2D 差异:")
    print(f"   LAYER_NUMS: 原始={orig_backbone2d['LAYER_NUMS']} | AirV2X={airv2x_backbone2d['LAYER_NUMS']}")
    print(f"   NUM_FILTERS: 原始={orig_backbone2d['NUM_FILTERS']} | AirV2X={airv2x_backbone2d['NUM_FILTERS']}")
    
    print("\n4. DENSE_HEAD 差异:")
    print(f"   NUM_CLASSES: 原始={orig_dense_head['NUM_CLASSES']} | AirV2X={airv2x_dense_head['NUM_CLASSES']}")
    
    print("\n5. 参数量影响分析:")
    print("   - WINDOW_SHAPE 从 [13,13,32] 变为 [54,15,1]: 大幅增加参数量")
    print("   - AGENT_INTERACT=True: 增加多agent交互参数")
    print("   - IMAGE_INPUT_LAYER.sparse_shape 从 [32,88,1] 变为 [32,88,32]: 增加32倍参数量")
    print("   - BACKBONE_2D 层数和通道数增加")
    print("   - DENSE_HEAD 类别数从10减少到7，但AirV2X没有使用DENSE_HEAD")

if __name__ == "__main__":
    compare_configs()












