/**
 * @file calligraphy_extractor.hpp
 * @brief 工业级书法单字提取管线 - C++ Qt 集成版
 *
 * 将 Python 原型翻译为 C++ 类，可无缝接入 Qt 架构
 *
 * Author: Industrial CV Pipeline
 * Date: 2026-04-10
 */

#ifndef CALLIGRAPHY_EXTRACTOR_HPP
#define CALLIGRAPHY_EXTRACTOR_HPP

#pragma once

#include <opencv2/opencv.hpp>
#include <QString>
#include <QDir>
#include <QDebug>
#include <QThreadPool>
#include <QRunnable>
#include <atomic>
#include <memory>
#include <vector>
#include <string>
#include <array>

namespace CalligraphyExtraction {

// ============================================================================
// 枚举定义
// ============================================================================

enum class BinarizationMethod {
    Otsu,
    AdaptiveMean,
    AdaptiveGaussian,
    Fixed
};

enum class MorphologyOperation {
    Dilate,
    Erode,
    None
};

// ============================================================================
// 配置结构体
// ============================================================================

struct PipelineConfig {
    // 轮廓过滤
    int minArea = 500;
    int maxArea = 500000;
    float minAspectRatio = 0.2f;
    float maxAspectRatio = 5.0f;

    // 形态学处理
    cv::Size morphKernelSize{3, 3};
    MorphologyOperation morphOperation = MorphologyOperation::Dilate;
    int morphIterations = 1;

    // 二值化
    BinarizationMethod binarizationMethod = BinarizationMethod::Otsu;
    int fixedThreshold = 127;
    int adaptiveBlockSize = 11;
    int adaptiveC = 2;

    // 高斯模糊
    cv::Size gaussianBlurSize{5, 5};
    double gaussianSigma = 0.0;

    // 输出
    int padPixels = 10;
    std::string outputFormat = "png";

    // 静态工厂方法
    static PipelineConfig defaultConfig() { return PipelineConfig{}; }

    static PipelineConfig fallbackConfig() {
        PipelineConfig cfg;
        cfg.binarizationMethod = BinarizationMethod::AdaptiveGaussian;
        cfg.morphOperation = MorphologyOperation::Erode;
        cfg.morphIterations = 2;
        return cfg;
    }
};

// ============================================================================
// 提取结果结构体
// ============================================================================

struct ExtractionResult {
    std::string charId;
    cv::Rect bbox;           // (x, y, w, h)
    cv::Mat bgraImage;       // 4通道 BGRA 图像
    float confidence = 1.0f;

    ExtractionResult() = default;
    ExtractionResult(const std::string& id, const cv::Rect& rect, const cv::Mat& img, float conf = 1.0f)
        : charId(id), bbox(rect), bgraImage(img), confidence(conf) {}
};

// ============================================================================
// 核心提取器类
// ============================================================================

class CalligraphyExtractor {
public:
    explicit CalligraphyExtractor(const QString& outputDir = QStringLiteral("extracted_chars"),
                                   const PipelineConfig& config = PipelineConfig{},
                                   bool verbose = true);

    // 禁用拷贝，允许移动
    CalligraphyExtractor(const CalligraphyExtractor&) = delete;
    CalligraphyExtractor& operator=(const CalligraphyExtractor&) = delete;

    // =========================================================================
    // 核心 API
    // =========================================================================

    /**
     * @brief 处理单张图像
     * @param imagePath 图像路径 (支持中文)
     * @return 成功提取的单字数量，-1 表示失败
     */
    int processImage(const QString& imagePath);

    /**
     * @brief 批量处理
     * @param imagePaths 图像路径列表
     * @param parallel 是否并行处理
     * @param workerCount 并行 worker 数量
     * @return 结果映射
     */
    std::map<QString, int> processBatch(const QStringList& imagePaths,
                                         bool parallel = false,
                                         int workerCount = 4);

    // =========================================================================
    // 配置管理
    // =========================================================================

    void setConfig(const PipelineConfig& config);
    PipelineConfig config() const { return m_config; }

    void setVerbose(bool verbose) { m_verbose = verbose; }

    // =========================================================================
    // 结果访问
    // =========================================================================

    const std::vector<ExtractionResult>& results() const { return m_results; }
    void clearResults() { m_results.clear(); }

    // =========================================================================
    // 统计信息
    // =========================================================================

    struct Statistics {
        int total = 0;
        double avgArea = 0.0;
        double minArea = 0.0;
        double maxArea = 0.0;
        double stdArea = 0.0;
    };

    Statistics getStatistics() const;

private:
    // =======================================================================
    // 私有成员
    // =======================================================================

    QString m_outputDir;
    PipelineConfig m_config;
    bool m_verbose;
    std::vector<ExtractionResult> m_results;

    // =======================================================================
    // 私有方法
    // =======================================================================

    // 1. 图像加载与校验
    cv::Mat loadImage(const QString& path) const;

    // 2. 二值化
    cv::Mat binarize(const cv::Mat& gray) const;

    // 3. 形态学处理
    cv::Mat morphology(const cv::Mat& binary) const;

    // 4. 轮廓过滤
    bool filterContour(const std::vector<cv::Point>& contour, double area) const;

    // 5. 创建 Alpha 通道 BGRA 图像
    cv::Mat createAlphaChannel(const cv::Mat& roiGray) const;

    // 6. 添加透明边框
    cv::Mat padBgra(const cv::Mat& bgra) const;

    // 7. 生成唯一 ID
    std::string generateId() const;

    // 内部日志
    void log(const QString& msg) const;

    // 零拷贝优化: 预分配内存池
    cv::Mat m_memoryPool;  // 用于避免频繁 alloc
};

// ============================================================================
// 自适应提取器 (带算法降级)
// ============================================================================

class AdaptiveCalligraphyExtractor : public CalligraphyExtractor {
public:
    explicit AdaptiveCalligraphyExtractor(const QString& outputDir = QStringLiteral("extracted_chars"),
                                           const PipelineConfig& config = PipelineConfig{},
                                           bool verbose = true,
                                           int maxRetries = 2);

    int processImage(const QString& imagePath) override;

private:
    int m_maxRetries;
    PipelineConfig m_fallbackConfig;

    void tryFallback(const QString& imagePath);
};

// ============================================================================
// 并行处理任务 (供 QThreadPool 使用)
// ============================================================================

class ExtractionTask : public QRunnable {
public:
    ExtractionTask(const QString& imagePath,
                   CalligraphyExtractor* extractor,
                   std::atomic<int>* counter);

    void run() override;

private:
    QString m_imagePath;
    CalligraphyExtractor* m_extractor;
    std::atomic<int>* m_counter;
};

// ============================================================================
// 实现细节 (inline 以减少编译单元)
// ============================================================================

inline cv::Mat CalligraphyExtractor::loadImage(const QString& path) const {
    // OpenCV imread 对中文路径支持不好，使用 FileStorage 或 imdecode
    QByteArray ba = path.toLocal8Bit();
    cv::Mat img = cv::imread(ba.constData());

    if (img.empty()) {
        // 尝试用 imdecode 读取
        QFile file(path);
        if (file.open(QIODevice::ReadOnly)) {
            QByteArray data = file.readAll();
            file.close();
            std::vector<uchar> buf(data.begin(), data.end());
            img = cv::imdecode(buf, cv::IMREAD_COLOR);
        }
    }

    return img;
}

inline cv::Mat CalligraphyExtractor::binarize(const cv::Mat& gray) const {
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, m_config.gaussianBlurSize, m_config.gaussianSigma);

    cv::Mat binary;

    switch (m_config.binarizationMethod) {
        case BinarizationMethod::Otsu: {
            cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
            break;
        }
        case BinarizationMethod::AdaptiveMean: {
            cv::adaptiveThreshold(blurred, binary, 255,
                                  cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY_INV,
                                  m_config.adaptiveBlockSize,
                                  m_config.adaptiveC);
            break;
        }
        case BinarizationMethod::AdaptiveGaussian: {
            cv::adaptiveThreshold(blurred, binary, 255,
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY_INV,
                                  m_config.adaptiveBlockSize,
                                  m_config.adaptiveC);
            break;
        }
        case BinarizationMethod::Fixed:
        default: {
            cv::threshold(blurred, binary, m_config.fixedThreshold, 255, cv::THRESH_BINARY_INV);
            break;
        }
    }

    return binary;
}

inline cv::Mat CalligraphyExtractor::morphology(const cv::Mat& binary) const {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, m_config.morphKernelSize);
    cv::Mat result;

    switch (m_config.morphOperation) {
        case MorphologyOperation::Dilate: {
            cv::dilate(binary, result, kernel, cv::Point(-1, -1), m_config.morphIterations);
            break;
        }
        case MorphologyOperation::Erode: {
            cv::erode(binary, result, kernel, cv::Point(-1, -1), m_config.morphIterations);
            break;
        }
        case MorphologyOperation::None:
        default: {
            result = binary;
            break;
        }
    }

    return result;
}

inline bool CalligraphyExtractor::filterContour(const std::vector<cv::Point>& contour, double area) const {
    if (area <= m_config.minArea || area >= m_config.maxArea) {
        return false;
    }

    cv::Rect bbox = cv::boundingRect(contour);
    float aspectRatio = static_cast<float>(bbox.width) / static_cast<float>(bbox.height);

    if (aspectRatio > m_config.maxAspectRatio || aspectRatio < m_config.minAspectRatio) {
        return false;
    }

    return true;
}

inline cv::Mat CalligraphyExtractor::createAlphaChannel(const cv::Mat& roiGray) const {
    // 灰度反相 -> Alpha
    cv::Mat alpha;
    cv::bitwise_not(roiGray, alpha);

    // 创建黑色 BGR
    std::vector<cv::Mat> channels = {
        cv::Mat::zeros(roiGray.size(), CV_8UC1),  // B
        cv::Mat::zeros(roiGray.size(), CV_8UC1),  // G
        cv::Mat::zeros(roiGray.size(), CV_8UC1),  // R
        alpha                                     // A
    };

    cv::Mat bgra;
    cv::merge(channels, bgra);

    return bgra;
}

inline cv::Mat CalligraphyExtractor::padBgra(const cv::Mat& bgra) const {
    cv::Mat padded;
    cv::copyMakeBorder(bgra, padded,
                       m_config.padPixels, m_config.padPixels,
                       m_config.padPixels, m_config.padPixels,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
    return padded;
}

inline std::string CalligraphyExtractor::generateId() const {
    // 简单的 UUID v4 简化版
    static const char hexChars[] = "0123456789abcdef";
    std::string id = "char_";
    for (int i = 0; i < 8; ++i) {
        int idx = rand() % 16;
        id += hexChars[idx];
    }
    return id;
}

inline void CalligraphyExtractor::log(const QString& msg) const {
    if (m_verbose) {
        qDebug() << msg;
    }
}

// ============================================================================
// 工具函数命名空间
// ============================================================================

namespace utils {

/**
 * @brief 检查文件是否为图像
 */
inline bool isImageFile(const QString& path) {
    static const QStringList imageExts = {
        QStringLiteral(".jpg"), QStringLiteral(".jpeg"),
        QStringLiteral(".png"), QStringLiteral(".bmp"),
        QStringLiteral(".tiff"), QStringLiteral(".tif")
    };

    QString ext = QFileInfo(path).suffix().toLower();
    return imageExts.contains(ext);
}

/**
 * @brief 获取目录下所有图像文件
 */
inline QStringList getImageFiles(const QString& dirPath) {
    QStringList result;
    QDir dir(dirPath);

    if (!dir.exists()) {
        return result;
    }

    for (const QFileInfo& info : dir.entryInfoList(QDir::Files)) {
        if (isImageFile(info.absoluteFilePath())) {
            result.append(info.absoluteFilePath());
        }
    }

    return result;
}

/**
 * @brief 创建带 Alpha 通道的图像文件
 */
inline bool saveBgraImage(const cv::Mat& bgra, const QString& path) {
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 9};
    QByteArray ba = path.toLocal8Bit();
    return cv::imwrite(ba.constData(), bgra, params);
}

} // namespace utils

} // namespace CalligraphyExtraction

#endif // CALLIGRAPHY_EXTRACTOR_HPP