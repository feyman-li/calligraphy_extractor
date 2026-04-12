"""
工业级书法单字提取管线 (Industrial-Grade Calligraphy Character Extraction Pipeline)
===============================================================================

基于 OpenCV 的传统 CV 形态学管线，适用于楷书、隶书、行书等字字独立的书法。
对于极度连绵的狂草，建议切换至 U-Net 语义分割管线。

Author: Industrial CV Pipeline
Date: 2026-04-10
"""

import cv2
import numpy as np
import os
import uuid
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


def imread_chinese(path: str) -> Optional[np.ndarray]:
    """解决 OpenCV 不支持中文路径的问题"""
    try:
        # 方法1: 使用 np.fromfile + cv2.imdecode
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        # 方法2: 尝试用 PIL 作为 fallback
        from PIL import Image
        pil_img = Image.open(path)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


class BinarizationMethod(Enum):
    """二值化方法枚举"""
    OTSU = "otsu"
    ADAPTIVE_MEAN = "adaptive_mean"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    FIXED = "fixed"


class MorphologyOperation(Enum):
    """形态学操作枚举"""
    DILATE = "dilate"
    ERODE = "erode"
    CLOSE = "close"  # 闭运算：先膨胀后腐蚀，连接靠近的区域
    NONE = "none"


@dataclass
class PipelineConfig:
    """管线配置参数"""
    # 轮廓过滤
    min_area: int = 500
    max_area: int = 500000
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0

    # 形态学处理
    morph_kernel_size: Tuple[int, int] = (5, 5)  # 增大核尺寸，连接更远的偏旁
    morph_operation: MorphologyOperation = MorphologyOperation.CLOSE
    morph_iterations: int = 2

    # 裁剪扩展
    crop_margin: int = 3  # 裁剪时扩展像素数，防止边缘被截断

    # 二值化
    binarization_method: BinarizationMethod = BinarizationMethod.OTSU
    fixed_threshold: int = 127
    adaptive_block_size: int = 11
    adaptive_c: int = 2

    # 高斯模糊
    gaussian_blur_size: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 0.0

    # 输出
    pad_pixels: int = 10
    output_format: str = "png"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.value if isinstance(v, Enum) else k for k, v in self.__dict__.items()}

    @classmethod
    def from_json(cls, json_path: str) -> 'PipelineConfig':
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if key == 'morph_operation':
                    setattr(config, key, MorphologyOperation(value))
                elif key == 'binarization_method':
                    setattr(config, key, BinarizationMethod(value))
                else:
                    setattr(config, key, value)
        return config


class ExtractionResult:
    """单字提取结果"""

    def __init__(self, char_id: str, bbox: Tuple[int, int, int, int],
                 image: np.ndarray, confidence: float = 1.0, image_path: str = ""):
        self.char_id = char_id
        self.bbox = bbox  # (x, y, w, h)
        self.image = image
        self.confidence = confidence
        self.image_path = image_path

    def save(self, output_path: str) -> bool:
        try:
            return cv2.imwrite(output_path, self.image)
        except Exception:
            return False


class CalligraphyExtractor:
    """
    工业级书法单字提取器

    管线流程:
    1. 图像加载与校验
    2. 灰度化 + 高斯去噪
    3. 二值化 (Otsu / 自适应阈值)
    4. 形态学处理 (膨胀/腐蚀)
    5. 轮廓检测与过滤
    6. Alpha 通道生成与裁切
    7. PNG 输出
    """

    def __init__(self, output_dir: str = "extracted_chars",
                 config: Optional[PipelineConfig] = None,
                 verbose: bool = True):
        """
        初始化提取器

        Args:
            output_dir: 输出目录
            config: 管线配置，如果为 None 则使用默认配置
            verbose: 是否打印详细信息
        """
        self.output_dir = output_dir
        self.config = config or PipelineConfig()
        self.verbose = verbose
        self.results: List[ExtractionResult] = []

        os.makedirs(self.output_dir, exist_ok=True)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _create_morph_kernel(self) -> np.ndarray:
        return np.ones(self.config.morph_kernel_size, np.uint8)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        执行二值化

        Args:
            gray: 灰度图像

        Returns:
            二值化后的图像 (黑底白字)
        """
        blurred = cv2.GaussianBlur(
            gray,
            self.config.gaussian_blur_size,
            self.config.gaussian_sigma
        )

        method = self.config.binarization_method

        if method == BinarizationMethod.OTSU:
            _, binary = cv2.threshold(
                blurred, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        elif method == BinarizationMethod.ADAPTIVE_MEAN:
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                self.config.adaptive_block_size,
                self.config.adaptive_c
            )

        elif method == BinarizationMethod.ADAPTIVE_GAUSSIAN:
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.config.adaptive_block_size,
                self.config.adaptive_c
            )

        else:  # FIXED
            _, binary = cv2.threshold(
                blurred,
                self.config.fixed_threshold,
                255,
                cv2.THRESH_BINARY_INV
            )

        return binary

    def _morphology(self, binary: np.ndarray) -> np.ndarray:
        """
        执行形态学处理

        Args:
            binary: 二值化图像

        Returns:
            形态学处理后的图像
        """
        kernel = self._create_morph_kernel()
        operation = self.config.morph_operation

        if operation == MorphologyOperation.DILATE:
            return cv2.dilate(binary, kernel, iterations=self.config.morph_iterations)
        elif operation == MorphologyOperation.ERODE:
            return cv2.erode(binary, kernel, iterations=self.config.morph_iterations)
        elif operation == MorphologyOperation.CLOSE:
            # 闭运算：先膨胀后腐蚀，填充内部空洞，连接靠近的区域
            dilated = cv2.dilate(binary, kernel, iterations=self.config.morph_iterations)
            return cv2.erode(dilated, kernel, iterations=self.config.morph_iterations)
        else:
            return binary

    def _filter_contour(self, cnt: np.ndarray, area: float) -> bool:
        """
        根据面积和长宽比过滤轮廓

        Args:
            cnt: 轮廓点阵
            area: 轮廓面积

        Returns:
            True 表示保留，False 表示过滤掉
        """
        if not (self.config.min_area < area < self.config.max_area):
            return False

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0

        if aspect_ratio > self.config.max_aspect_ratio or aspect_ratio < self.config.min_aspect_ratio:
            return False

        return True

    def _create_alpha_channel(self, roi_gray: np.ndarray) -> np.ndarray:
        """
        创建带 Alpha 通道的图像

        核心 Trick: 使用灰度反相作为 Alpha 通道，完美保留宣纸洇墨的半透明过渡边缘！
        避免直接用二值图造成的边缘锯齿问题。

        Args:
            roi_gray: 裁切区域的灰度图

        Returns:
            BGRA 格式的四通道图像
        """
        # 灰度反相 -> Alpha 通道 (墨迹深的地方 Alpha 越不透明)
        alpha_channel = 255 - roi_gray

        # 创建纯黑 BGR 通道
        h, w = roi_gray.shape
        bgr_black = np.zeros((h, w, 3), dtype=np.uint8)

        # 合并为 BGRA
        bgra = cv2.merge((bgr_black[:, :, 0], bgr_black[:, :, 1], bgr_black[:, :, 2], alpha_channel))

        return bgra

    def _pad_bgra(self, bgra: np.ndarray) -> np.ndarray:
        """为 BGRA 图像添加透明边框"""
        pad = self.config.pad_pixels
        return cv2.copyMakeBorder(
            bgra, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0, 0)
        )

    def process_image(self, image_path: str,
                      save_original_color: bool = False,
                      color_suffix: str = "_color") -> int:
        """
        处理单张书法图像

        Args:
            image_path: 图像路径 (支持中文路径)
            save_original_color: 是否保存原图彩色版本
            color_suffix: 彩色版本文件名的后缀

        Returns:
            成功提取的单字数量

        Raises:
            ValueError: 图像无法读取
        """
        self._log(f"[*] 开始处理: {image_path}")

        # 1. 图像加载与校验 (使用 imread_chinese 处理中文路径)
        img = imread_chinese(image_path)
        if img is None:
            raise ValueError(f"无法读取图像，请检查路径或包含中文字符: {image_path}")

        h, w = img.shape[:2]
        self._log(f"    图像尺寸: {w}x{h}")

        # 预存灰度原图，用于制作平滑 Alpha 通道
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 二值化
        binary = self._binarize(gray_original)
        self._log(f"    二值化方法: {self.config.binarization_method.value}")

        # 3. 形态学处理
        morphed = self._morphology(binary)
        self._log(f"    形态学操作: {self.config.morph_operation.value}")

        # 4. 轮廓检测
        # RETR_EXTERNAL: 只提取最外层轮廓
        contours, _ = cv2.findContours(
            morphed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        self._log(f"    检测到轮廓数: {len(contours)}")

        # 5. 过滤与分割
        extracted_count = 0
        self.results.clear()

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            if not self._filter_contour(cnt, area):
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            # 扩展裁剪区域，防止边缘被截断
            margin = self.config.crop_margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + cw + margin)
            y2 = min(img.shape[0], y + ch + margin)

            # 提取 ROI
            roi_gray = gray_original[y1:y2, x1:x2]

            # 生成 Alpha 通道 BGRA 图像 (灰度版)
            bgra_char = self._create_alpha_channel(roi_gray)
            bgra_padded = self._pad_bgra(bgra_char)

            # 保存灰度版本
            char_id = uuid.uuid4().hex[:8]
            filename = f"char_{char_id}.{self.config.output_format}"
            filepath = os.path.join(self.output_dir, filename)

            if cv2.imwrite(filepath, bgra_padded):
                self.results.append(ExtractionResult(
                    char_id=char_id,
                    bbox=(x, y, cw, ch),
                    image=bgra_padded,
                    image_path=filepath
                ))
                extracted_count += 1

            # 保存原图彩色版本
            if save_original_color:
                roi_color = img[y1:y2, x1:x2]
                color_filename = f"char_{char_id}{color_suffix}.{self.config.output_format}"
                color_filepath = os.path.join(self.output_dir, color_filename)
                cv2.imwrite(color_filepath, roi_color)

        self._log(f"[+] 提取完成！共成功分割 {extracted_count} 个单字。")

        return extracted_count

    def process_batch(self, image_paths: List[str],
                      parallel: bool = False,
                      workers: int = 4,
                      save_original_color: bool = False) -> Dict[str, int]:
        """
        批量处理多张图像

        Args:
            image_paths: 图像路径列表
            parallel: 是否并行处理
            workers: 并行 worker 数量
            save_original_color: 是否保存原图彩色版本

        Returns:
            字典，key 为图像路径，value 为提取数量
        """
        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_path = {
                    executor.submit(self.process_image, p, save_original_color): p
                    for p in image_paths
                }
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[path] = future.result()
                    except Exception as e:
                        self._log(f"[!] 处理失败 {path}: {e}")
                        results[path] = 0
            return results
        else:
            return {p: self.process_image(p, save_original_color) for p in image_paths}

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.results:
            return {"total": 0}

        areas = [bbox[2] * bbox[3] for bbox in [r.bbox for r in self.results]]
        return {
            "total": len(self.results),
            "avg_area": np.mean(areas),
            "min_area": np.min(areas),
            "max_area": np.max(areas),
            "std_area": np.std(areas)
        }


class AdaptiveCalligraphyExtractor(CalligraphyExtractor):
    """
    自适应书法单字提取器

    具备算法降级与容错机制:
    - 当 Otsu 提取结果异常时，自动切换至局部自适应阈值
    - 监控轮廓数量，异常时触发重试
    """

    def __init__(self, output_dir: str = "extracted_chars",
                 config: Optional[PipelineConfig] = None,
                 verbose: bool = True,
                 max_retries: int = 2):
        super().__init__(output_dir, config, verbose)
        self.max_retries = max_retries
        self.fallback_config = PipelineConfig(
            binarization_method=BinarizationMethod.ADAPTIVE_GAUSSIAN,
            morph_operation=MorphologyOperation.ERODE,  # 粘连时用腐蚀断开
            morph_iterations=2
        )

    def process_image(self, image_path: str) -> int:
        """
        处理图像，带自动降级机制

        Args:
            image_path: 图像路径

        Returns:
            成功提取的单字数量
        """
        try:
            count = super().process_image(image_path)
        except ValueError:
            raise

        # 异常检测: 轮廓数量为 0 或异常多（可能是满屏噪点）
        expected_range = (1, 1000)
        if count == 0 or count > expected_range[1]:
            self._log(f"[!] Otsu 结果异常 (count={count})，尝试自适应阈值...")
            self._try_fallback(image_path)

        return len(self.results)

    def _try_fallback(self, image_path: str) -> None:
        """尝试备用配置处理"""
        original_config = self.config
        for retry in range(self.max_retries):
            self.config = self.fallback_config
            self._log(f"    重试 {retry + 1}/{self.max_retries}...")

            try:
                super().process_image(image_path)
                if len(self.results) > 0:
                    self._log(f"[+] Fallback 成功，提取 {len(self.results)} 个单字")
                    break
            except Exception as e:
                self._log(f"    Fallback 重试失败: {e}")
            finally:
                self.config = original_config


# --- 运行管线 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="工业级书法单字提取管线")
    parser.add_argument("input", help="输入图像路径或包含图像的目录")
    parser.add_argument("-o", "--output", default="extracted_chars", help="输出目录")
    parser.add_argument("-c", "--config", help="JSON 配置文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--parallel", action="store_true", help="并行处理")
    parser.add_argument("--workers", type=int, default=4, help="并行 worker 数")
    parser.add_argument("--color", action="store_true", help="同时保存原图彩色版本")

    # OCR 自动打标参数
    parser.add_argument("--ocr", action="store_true", help="启用 OCR 自动打标")
    parser.add_argument("--use-paddle", action="store_true", help="使用 PaddleOCR 原生模式 (自动下载模型)")
    parser.add_argument("--easyocr", action="store_true", help="使用 EasyOCR 模式")
    parser.add_argument("--ocr-model", help="OCR ONNX 模型路径")
    parser.add_argument("--ocr-dict", help="OCR 字典文件路径")
    parser.add_argument("--author", default="未知", help="作者姓名")
    parser.add_argument("--work", default="未知", help="作品名称")

    args = parser.parse_args()

    # 加载配置
    if args.config and os.path.exists(args.config):
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig()

    extractor = CalligraphyExtractor(
        output_dir=args.output,
        config=config,
        verbose=args.verbose
    )

    # 初始化 OCR (如果启用)
    ocr_tagger = None
    metadata_gen = None
    if args.ocr:
        try:
            from ocr_auto_tagger import OcrAutoTagger, MetadataGenerator
            if args.use_paddle:
                # PaddleOCR 原生模式 (自动下载模型)
                ocr_tagger = OcrAutoTagger(use_onnx=False)
                print("[*] OCR 已启用: PaddleOCR 原生模式")
            elif args.easyocr:
                # EasyOCR 模式
                ocr_tagger = OcrAutoTagger(use_easyocr=True)
                print("[*] OCR 已启用: EasyOCR 模式")
            elif args.ocr_model and os.path.exists(args.ocr_model):
                # ONNX 模式
                ocr_tagger = OcrAutoTagger(
                    model_path=args.ocr_model,
                    dict_path=args.ocr_dict,
                    use_onnx=True,
                    use_opencv_dnn=True
                )
                print(f"[*] OCR 已启用: {args.ocr_model}")
            else:
                print("[!] OCR 模式需要指定 --use-paddle, --easyocr 或 --ocr-model <模型路径>")
                print("[*] 推荐使用 --easyocr (需安装: pip install easyocr)")
                print("[*] 或使用 --use-paddle 自动下载 PaddleOCR 模型")
                exit(1)
            metadata_gen = MetadataGenerator(
                author=args.author,
                work_name=args.work,
                output_dir=args.output
            )
        except ImportError as e:
            print(f"[!] OCR 模块导入失败: {e}")
            exit(1)

    # 判断输入是文件还是目录
    input_path = Path(args.input)

    def process_with_ocr(extractor, image_path, ocr_tagger, metadata_gen, save_color):
        """处理单张图像并生成 OCR 元数据"""
        count = extractor.process_image(image_path, save_original_color=save_color)

        if ocr_tagger and metadata_gen:
            # 对整张图像运行 OCR（仅用于元数据记录）
            print("[*] 正在识别整幅图像文字...")
            ocr_results = ocr_tagger.recognize_full_image(image_path)
            print(f"[*] 检测到 {len(ocr_results)} 个文本区域")

            # 匹配 OCR 结果（用于元数据）
            char_counter = 1
            for result in extractor.results:
                # 查找最近的 OCR 结果
                ocr_result = ocr_tagger.match_char_to_ocr_result(
                    result.bbox, ocr_results
                )

                # 文件名使用序号，不使用 OCR 文本
                new_basename = f"{args.author}_{args.work}_{char_counter:03d}"
                char_counter += 1

                # 更新结果中的文件名
                result.char_id = new_basename

                # 生成元数据（OCR 结果记录在 JSON 中）
                metadata = metadata_gen.generate(
                    char_img=result.image,
                    ocr_result=ocr_result,
                    bbox=result.bbox,
                    image_path=result.image_path
                )
                metadata["char_id"] = new_basename

                # 保存 JSON
                json_path = os.path.join(args.output, new_basename + ".json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

                # 重命名图像文件
                old_path = result.image_path
                new_path = os.path.join(args.output, new_basename + ".png")
                if os.path.exists(old_path) and old_path != new_path:
                    os.rename(old_path, new_path)
                    result.image_path = new_path

                # 重命名彩色版本
                if save_color:
                    old_color_path = old_path.replace(".png", "_color.png")
                    new_color_path = new_path.replace(".png", "_color.png")
                    if os.path.exists(old_color_path):
                        os.rename(old_color_path, new_color_path)

        return count

    if input_path.is_file():
        try:
            count = process_with_ocr(extractor, str(input_path), ocr_tagger, metadata_gen, args.color)
            print(f"[*] 处理完成，提取 {count} 个单字")
        except Exception as e:
            print(f"[!] Error: {e}")

    elif input_path.is_dir():
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            str(f) for f in input_path.glob("*")
            if f.suffix.lower() in image_exts
        ]

        if not image_files:
            print(f"[!] 在 {input_path} 中未找到图像文件")
        else:
            print(f"[*] 找到 {len(image_files)} 张图像")
            results = extractor.process_batch(image_files, parallel=args.parallel, workers=args.workers, save_original_color=args.color)
            for path, count in results.items():
                print(f"    {path}: {count} 个单字")
            for path, count in results.items():
                print(f"    {path}: {count} 个单字")
    else:
        print(f"[!] 输入路径无效: {args.input}")