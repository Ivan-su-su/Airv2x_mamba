# AirV2X配置修改总结

## 🎯 修改目标
根据where2comm的配置标准，调整AirV2X的点云范围、voxel尺寸和BEV网格尺寸，以解决长宽比不匹配的问题。

## 📊 修改前后对比

### **修改前 (AirV2X原始配置)**
- **点云范围**: `[-140.8, -40.0, -4.0, 140.8, 40.0, 4.0]`
  - X范围: 281.6米
  - Y范围: 80米
  - 长宽比: 3.52:1
- **Voxel尺寸**: `[0.782, 0.222, 8.0]`
- **BEV网格**: `[360, 360, 1]` (强制正方形)
- **实际网格**: 360×360 (不匹配实际长宽比)

### **修改后 (参考where2comm配置)**
- **点云范围**: `[-100.8, -40.0, -3.0, 100.8, 40.0, 1.0]`
  - X范围: 201.6米
  - Y范围: 80米
  - 长宽比: 2.52:1
- **Voxel尺寸**: `[0.4, 0.4, 4.0]`
- **BEV网格**: `[504, 200, 1]` (匹配实际长宽比)
- **实际网格**: 504×200 (正确反映点云长宽比)

## 🔧 具体修改内容

### 1. **配置文件修改**
**文件**: `AirV2X-Perception-Checkpoints/airv2x_intermediate_mambafusion/config.yaml`

```yaml
# 预处理配置
preprocess:
  voxel_size: 
    - 0.4      # X方向voxel尺寸
    - 0.4      # Y方向voxel尺寸  
    - 4.0      # Z方向voxel尺寸
  cav_lidar_range:
    - -100.8   # X最小值
    - -40.0    # Y最小值
    - -3.0     # Z最小值
    - 100.8    # X最大值
    - 40.0     # Y最大值
    - 1.0      # Z最大值

# BEV映射配置
MAP_TO_BEV:
  NAME: PointPillarScatter3d
  INPUT_SHAPE: [504, 200, 1]  # [W, H, D] = [X, Y, Z]
  NUM_BEV_FEATURES: 64

# 后处理配置
postprocess:
  anchor_args:
    D: 1
    H: 200     # Y方向网格数
    W: 504     # X方向网格数
    feature_stride: 2
    cav_lidar_range: [与preprocess保持一致]
```

### 2. **代码修改**
**文件**: `opencood/models/mambafusion_modules/backbones_2d/map_to_bev/pointpillar_scatter.py`

```python
# 修改前：硬编码360×360
if self.ny == 360 and self.nx == 360:
    batch_spatial_features = F.adaptive_avg_pool2d(batch_spatial_features, (180, 180))

# 修改后：动态适应网格尺寸
if self.ny > 200 or self.nx > 200:
    target_h = min(200, self.ny)
    target_w = min(200, self.nx)
    batch_spatial_features = F.adaptive_avg_pool2d(batch_spatial_features, (target_h, target_w))
```

### 3. **示例代码修改**
**文件**: `opencood/models/mambafusion_modules/backbones_2d/map_to_bev/simple_bev_example.py`

```python
# 更新配置
INPUT_SHAPE = [504, 200, 1]  # 根据where2comm配置调整

# 更新坐标范围
y_coords = torch.randint(0, 200, (num_pillars, 1))  # Y方向: 0-200
x_coords = torch.randint(0, 504, (num_pillars, 1))  # X方向: 0-504
```

## 📈 修改效果

### **网格尺寸计算**
- **X方向**: 201.6米 ÷ 0.4米 = **504个voxel**
- **Y方向**: 80米 ÷ 0.4米 = **200个voxel**
- **长宽比**: 504:200 = **2.52:1** (正确反映点云长宽比)

### **内存优化**
- **原始**: 360×360 = 129,600个网格
- **修改后**: 504×200 = 100,800个网格
- **内存减少**: 约22%的内存节省

### **检测精度提升**
- **更合理的voxel尺寸**: 0.4×0.4米提供更好的空间分辨率
- **正确的长宽比**: BEV网格正确反映点云的实际分布
- **减少几何失真**: 避免强制压缩成正方形造成的精度损失

## 🎯 预期改进

1. **检测精度**: 更合理的网格尺寸和长宽比将提高目标检测精度
2. **内存效率**: 减少不必要的网格数量，降低内存占用
3. **计算效率**: 更合理的网格尺寸减少计算复杂度
4. **可视化效果**: BEV可视化将更准确地反映实际点云分布

## ✅ 验证方法

1. **运行可视化**: 使用修改后的示例代码验证BEV特征图
2. **检查网格尺寸**: 确认生成的BEV特征图尺寸为504×200
3. **性能测试**: 对比修改前后的检测精度和内存使用
4. **长宽比验证**: 确认BEV特征图正确反映2.52:1的长宽比

## 📝 注意事项

1. **数据兼容性**: 确保训练数据与新的点云范围兼容
2. **模型权重**: 可能需要重新训练模型以适应新的网格尺寸
3. **其他配置**: 检查是否有其他硬编码的网格尺寸需要同步修改
4. **测试验证**: 在修改后进行充分的测试验证

这些修改将使AirV2X的配置更加合理，提高检测精度并优化内存使用。



