/**
 * @file calligraphy_extractor.cpp
 * @brief 工业级书法单字提取管线 - C++ 实现
 *
 * Author: Industrial CV Pipeline
 * Date: 2026-04-10
 */

#include "calligraphy_extractor.hpp"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QDateTime>
#include <random>

namespace CalligraphyExtraction {

// ============================================================================
// CalligraphyExtractor 实现
// ============================================================================

CalligraphyExtractor::CalligraphyExtractor(const QString& outputDir,
                                           const PipelineConfig& config,
                                           bool verbose)
    : m_outputDir(outputDir),
      m_config(config),
      m_verbose(verbose)
{
    QDir().mkpath(m_outputDir);

    // 预分配内存池 (用于减少大图像处理的内存分配开销)
    m_memoryPool = cv::Mat(2048, 2048, CV_8UC4);
}

int CalligraphyExtractor::processImage(const QString& imagePath) {
    log(QStringLiteral("[*] 开始处理: %1").arg(imagePath));

    // 1. 加载图像
    cv::Mat img = loadImage(imagePath);
    if (img.empty()) {
        log(QStringLiteral("[!] 无法读取图像: %1").arg(imagePath));
        return -1;
    }

    int height = img.rows;
    int width = img.cols;
    log(QStringLiteral("    图像尺寸: %1x%2").arg(width).arg(height));

    // 预存灰度原图
    cv::Mat grayOriginal;
    cv::cvtColor(img, grayOriginal, cv::COLOR_BGR2GRAY);

    // 2. 二值化
    cv::Mat binary = binarize(grayOriginal);
    log(QStringLiteral("    二值化方法: %1").arg(
        static_cast<int>(m_config.binarizationMethod)));

    // 3. 形态学处理
    cv::Mat morphed = morphology(binary);
    log(QStringLiteral("    形态学操作: %1").arg(
        static_cast<int>(m_config.morphOperation)));

    // 4. 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morphed, contours, cv::noArray(),
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    log(QStringLiteral("    检测到轮廓数: %1").arg(contours.size()));

    // 5. 过滤与分割
    m_results.clear();
    int extractedCount = 0;

    for (size_t i = 0; i < contours.size(); ++i) {
        const auto& contour = contours[i];
        double area = cv::contourArea(contour);

        if (!filterContour(contour, area)) {
            continue;
        }

        cv::Rect bbox = cv::boundingRect(contour);
        int x = bbox.x;
        int y = bbox.y;
        int w = bbox.width;
        int h = bbox.height;

        // 提取 ROI
        cv::Mat roiGray = grayOriginal(cv::Rect(x, y, w, h));

        // 生成 Alpha 通道 BGRA
        cv::Mat bgraChar = createAlphaChannel(roiGray);
        cv::Mat bgraPadded = padBgra(bgraChar);

        // 保存文件
        std::string charId = generateId();
        QString filename = QStringLiteral("char_%1.%2")
                             .arg(QString::fromStdString(charId))
                             .arg(QString::fromStdString(m_config.outputFormat));
        QString filepath = QDir(m_outputDir).filePath(filename);

        if (utils::saveBgraImage(bgraPadded, filepath)) {
            m_results.emplace_back(charId, bbox, bgraPadded, 1.0f);
            ++extractedCount;
        }
    }

    log(QStringLiteral("[+] 提取完成！共成功分割 %1 个单字。").arg(extractedCount));

    return extractedCount;
}

std::map<QString, int> CalligraphyExtractor::processBatch(const QStringList& imagePaths,
                                                           bool parallel,
                                                           int workerCount) {
    std::map<QString, int> results;

    if (parallel) {
        std::atomic<int> counter(0);
        QThreadPool pool;
        pool.setMaxThreadCount(workerCount);

        for (const QString& path : imagePaths) {
            ExtractionTask* task = new ExtractionTask(path, this, &counter);
            pool.start(task);
        }

        pool.waitForFinish();

        // 汇总结果
        for (const QString& path : imagePaths) {
            results[path] = 0;  // 并行模式下不追踪单个结果
        }
    } else {
        for (const QString& path : imagePaths) {
            int count = processImage(path);
            results[path] = count >= 0 ? count : 0;
        }
    }

    return results;
}

void CalligraphyExtractor::setConfig(const PipelineConfig& config) {
    m_config = config;
}

CalligraphyExtractor::Statistics CalligraphyExtractor::getStatistics() const {
    Statistics stats;
    stats.total = static_cast<int>(m_results.size());

    if (m_results.empty()) {
        return stats;
    }

    std::vector<double> areas;
    areas.reserve(m_results.size());

    for (const auto& result : m_results) {
        const cv::Rect& bbox = result.bbox;
        areas.push_back(static_cast<double>(bbox.width * bbox.height));
    }

    double sum = 0.0;
    double minArea = areas[0];
    double maxArea = areas[0];

    for (double area : areas) {
        sum += area;
        minArea = std::min(minArea, area);
        maxArea = std::max(maxArea, area);
    }

    stats.avgArea = sum / areas.size();
    stats.minArea = minArea;
    stats.maxArea = maxArea;

    // 标准差
    double sqSum = 0.0;
    for (double area : areas) {
        sqSum += (area - stats.avgArea) * (area - stats.avgArea);
    }
    stats.stdArea = std::sqrt(sqSum / areas.size());

    return stats;
}

// ============================================================================
// AdaptiveCalligraphyExtractor 实现
// ============================================================================

AdaptiveCalligraphyExtractor::AdaptiveCalligraphyExtractor(
        const QString& outputDir,
        const PipelineConfig& config,
        bool verbose,
        int maxRetries)
    : CalligraphyExtractor(outputDir, config, verbose),
      m_maxRetries(maxRetries),
      m_fallbackConfig(PipelineConfig::fallbackConfig())
{}

int AdaptiveCalligraphyExtractor::processImage(const QString& imagePath) {
    // 先尝试主配置
    int count = CalligraphyExtractor::processImage(imagePath);

    // 异常检测
    if (count == 0 || count > 1000) {
        log(QStringLiteral("[!] Otsu 结果异常 (count=%1)，尝试自适应阈值...").arg(count));
        tryFallback(imagePath);
    }

    return static_cast<int>(results().size());
}

void AdaptiveCalligraphyExtractor::tryFallback(const QString& imagePath) {
    PipelineConfig originalConfig = config();

    for (int retry = 0; retry < m_maxRetries; ++retry) {
        log(QStringLiteral("    重试 %1/%2...").arg(retry + 1).arg(m_maxRetries));

        setConfig(m_fallbackConfig);

        try {
            int count = CalligraphyExtractor::processImage(imagePath);
            if (count > 0) {
                log(QStringLiteral("[+] Fallback 成功，提取 %1 个单字").arg(count));
                break;
            }
        } catch (const std::exception& e) {
            log(QStringLiteral("    Fallback 重试失败: %1").arg(e.what()));
        }
    }

    setConfig(originalConfig);
}

// ============================================================================
// ExtractionTask 实现
// ============================================================================

ExtractionTask::ExtractionTask(const QString& imagePath,
                                CalligraphyExtractor* extractor,
                                std::atomic<int>* counter)
    : m_imagePath(imagePath),
      m_extractor(extractor),
      m_counter(counter)
{}

void ExtractionTask::run() {
    int result = m_extractor->processImage(m_imagePath);
    if (result > 0) {
        (*m_counter) += result;
    }
}

} // namespace CalligraphyExtraction