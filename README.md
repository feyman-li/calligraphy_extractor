# 工业级书法单字提取管线

基于 OpenCV 的传统 CV 形态学管线，适用于楷书、隶书、行书等字字独立的书法提取。

## 目录结构

```
calligraphy_extractor/
├── calligraphy_extractor.py   # Python 实现
├── calligraphy_extractor.hpp  # C++ 头文件 (Qt 集成)
├── calligraphy_extractor.cpp  # C++ 实现
├── default_config.json        # 默认配置
└── README.md
```

## Python 使用

### 快速开始

```bash
# 安装依赖
pip install opencv-python numpy

# 处理单张图像
python calligraphy_extractor.py input.jpg -o output_chars

# 处理整个目录
python calligraphy_extractor.py ./calligraphy_images/ -o extracted

# 启用详细输出
python calligraphy_extractor.py input.jpg -v

# 使用配置文件
python calligraphy_extractor.py input.jpg -c custom_config.json
```

### Python API

```python
from calligraphy_extractor import CalligraphyExtractor, PipelineConfig

# 使用默认配置
extractor = CalligraphyExtractor(output_dir="chars")

# 自定义配置
config = PipelineConfig(
    min_area=300,
    max_area=300000,
    morph_operation="erode",  # 断开粘连
    morph_iterations=2
)
extractor = CalligraphyExtractor(output_dir="chars", config=config)

# 处理图像
count = extractor.process_image("calligraphy.jpg")
print(f"提取了 {count} 个单字")

# 批量处理
results = extractor.process_batch(["img1.jpg", "img2.jpg"], parallel=True)
```

## C++ Qt 集成

```cpp
#include "calligraphy_extractor.hpp"

using namespace CalligraphyExtraction;

// 创建提取器
CalligraphyExtractor extractor("output_chars");

// 处理图像
int count = extractor.processImage(":/resources/calligraphy.jpg");

// 获取统计
auto stats = extractor.getStatistics();
qDebug() << "总计:" << stats.total << "个字";
```

### Qt 并行处理

```cpp
QStringList images = utils::getImageFiles("./calligraphy/");
auto results = extractor.processBatch(images, true, 4);  // 4 线程并行
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_area` | 500 | 最小轮廓面积，过滤噪点 |
| `max_area` | 500000 | 最大轮廓面积，过滤边框 |
| `morph_operation` | dilate | 形态学操作：dilate/erode/none |
| `morph_iterations` | 1 | 形态学迭代次数 |
| `binarization_method` | otsu | 二值化方法 |
| `pad_pixels` | 10 | 输出图像边框填充 |

## 核心算法流程

```
原图 → 灰度化 → 高斯模糊 → 二值化(Otsu/自适应)
       ↓
    形态学处理(膨胀/腐蚀)
       ↓
    轮廓检测(RETR_EXTERNAL)
       ↓
    面积/长宽比过滤
       ↓
    ROI裁切 → Alpha通道生成 → PNG输出
```

## Alpha 通道处理

传统方法直接用二值图做透明度会有锯齿。本管线使用**灰度反相作为 Alpha 通道**，完美保留宣纸洇墨的半透明过渡边缘。

## 狂草/连绵书体处理策略

对于王羲之《十七帖》那种**极度连绵的狂草**（笔画物理上完全粘连），传统 CV 方法会失效。

### 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| Watershed | 计算快，无需训练 | 依赖梯度信息，连绵笔画无"谷"可分 |
| U-Net | 端到端，学到语义 | 需要大量标注数据 |

### 推荐方案

对于狂草，推荐 **U-Net 语义分割**：

1. 收集/标注 1000+ 书法图像
2. 使用预训练 backbone (ResNet50/EfficientNet)
3. 训练字符分割模型
4. 集成到管线中作为 Fallback

传统 CV → 轮廓数异常 → 触发 U-Net 推理

## 性能优化 (C++)

1. **内存池**: 预分配 `cv::Mat` 减少 `malloc` 调用
2. **零拷贝**: 使用 `cv::merge` 而非手动拷贝
3. **并行**: `QThreadPool` + `QRunnable` 多核并行
4. **SIMD**: OpenCV 默认启用 SSE/NEON 加速

## License

MIT