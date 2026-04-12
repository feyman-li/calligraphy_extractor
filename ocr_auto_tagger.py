"""
OcrAutoTagger: 书法单字 OCR 自动打标模块
==========================================

支持两种推理后端:
1. ONNX Runtime (推荐): 使用 OpenCV DNN 或 onnxruntime 加载 ONNX 模型
2. PaddleOCR Native: 直接使用 PaddleOCR Python API

依赖 (ONNX模式):
    pip install opencv-python numpy onnxruntime

依赖 (PaddleOCR模式):
    pip install paddleocr paddlepaddle

Author: Industrial CV Pipeline
Date: 2026-04-10
"""

import cv2
import numpy as np
import os
import json
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OcrResult:
    """OCR 识别结果"""
    text: str           # 识别出的汉字
    confidence: float   # 置信度 (0.0 ~ 1.0)


class OcrAutoTagger:
    """
    轻量级书法单字识别器

    支持两种模式:
    1. ONNX模式: 使用 OpenCV DNN 或 ONNXRuntime 加载导出的 ONNX 模型
    2. PaddleOCR模式: 直接使用 PaddleOCR 原生 API (自动下载模型)
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 dict_path: Optional[str] = None,
                 use_onnx: bool = True,
                 use_opencv_dnn: bool = False,
                 confidence_threshold: float = 0.5,
                 lang: str = 'ch',
                 use_easyocr: bool = False):
        """
        初始化 OCR 识别器

        Args:
            model_path: ONNX 模型文件路径 (ONNX模式)
            dict_path: 字典文件路径 (ONNX模式, PaddleOCR ppocr_keys_v1.txt)
            use_onnx: True=使用ONNX模式, False=使用PaddleOCR原生API
            use_opencv_dnn: True=使用OpenCV DNN (ONNX模式), False=使用ONNXRuntime
            confidence_threshold: 置信度阈值
            lang: 语言选项 ('ch' for Chinese)
            use_easyocr: True=使用EasyOCR模式 (备用方案)
        """
        self.confidence_threshold = confidence_threshold
        self.use_onnx = use_onnx
        self.use_easyocr = use_easyocr
        self.vocabulary: List[str] = []

        if use_easyocr:
            # EasyOCR 模式
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            except ImportError:
                raise ImportError("EasyOCR 未安装。请运行: pip install easyocr")
            return

        if use_onnx:
            # ONNX 模式
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX模型文件不存在: {model_path}")

            # 加载字典
            if dict_path and os.path.exists(dict_path):
                self._load_dict(dict_path)
            else:
                self._init_default_dict()

            # 加载模型
            if use_opencv_dnn:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.use_opencv = True
                self.session = None
            else:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 4
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options,
                    providers=['CPUExecutionProvider']
                )
                self.input_name = self.session.get_inputs()[0].name
                self.use_opencv = False
        else:
            # PaddleOCR 原生模式 - 优先使用 venv_paddle 中的版本
            import sys
            # 获取脚本所在目录的父目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            venv_python = os.path.join(script_dir, 'venv_paddle', 'Scripts', 'python.exe')

            if os.path.exists(venv_python):
                # 使用 venv_paddle 中的 Python
                self.paddle_ocr = None
                self.venv_python = venv_python
                self.use_opencv = False
                self.session = None
            else:
                # 回退到系统 Python
                try:
                    from paddleocr import PaddleOCR
                    self.paddle_ocr = PaddleOCR(
                        lang=lang,
                        use_angle_cls=False
                    )
                    self.venv_python = None
                except ImportError:
                    raise ImportError(
                        "PaddleOCR 未安装。请运行: pip install paddleocr paddlepaddle"
                    )

    def _load_dict(self, dict_path: str):
        """加载字典文件"""
        self.vocabulary = ['blank']  # CTC blank token
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip()
                if char:
                    self.vocabulary.append(char)
        self.vocabulary.append(' ')  # space

    def _init_default_dict(self):
        """初始化默认汉字字典 (常用字)"""
        common_chars = (
            "的一是不了在人我有他这中大来上个国们为上 "
            "地到方以就出要年时得才可下过量起政好小部其此"
            "心前于学还天分能对程军民言果党十地城石里运五"
            "喜彩晨壶碎得益绿萍角鼓编胡飘叶风春异截横飘"
        )
        self.vocabulary = ['blank'] + list(common_chars) + [' ']

    def _preprocess(self, char_img: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """预处理单字图像"""
        h = 48
        w = max(48, int(char_img.shape[1] * (48.0 / char_img.shape[0])))

        resized = cv2.resize(char_img, (w, h))

        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1/127.5,
            size=(w, h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )

        return blob, 48.0 / h, w

    def _ctc_decode_greedy(self, net_output: np.ndarray) -> OcrResult:
        """CTC 贪心解码"""
        if len(net_output.shape) == 4:
            net_output = net_output.squeeze(0)

        time_steps, vocab_size = net_output.shape
        data = net_output

        result_text = ""
        total_confidence = 0.0
        valid_chars = 0
        last_index = -1

        for t in range(time_steps):
            max_idx = np.argmax(data[t])
            max_prob = float(data[t, max_idx])

            if max_idx > 0 and max_idx != last_index:
                if max_idx < len(self.vocabulary):
                    result_text += self.vocabulary[max_idx]
                    total_confidence += max_prob
                    valid_chars += 1

            last_index = max_idx

        avg_confidence = total_confidence / valid_chars if valid_chars > 0 else 0.0

        if not result_text:
            result_text = "未知"

        return OcrResult(text=result_text, confidence=avg_confidence)

    def recognize(self, char_img: np.ndarray) -> OcrResult:
        """
        识别单字

        Args:
            char_img: BGR 或灰度图像 (单字裁切图)

        Returns:
            OcrResult: 识别结果
        """
        if self.use_easyocr:
            return self._recognize_easyocr(char_img)
        elif self.use_onnx:
            return self._recognize_onnx(char_img)
        else:
            return self._recognize_paddle(char_img)

    def _recognize_onnx(self, char_img: np.ndarray) -> OcrResult:
        """ONNX 模式识别"""
        blob, scale, target_w = self._preprocess(char_img)

        if self.use_opencv:
            self.net.setInput(blob)
            output = self.net.forward()
        else:
            output = self.session.run(None, {self.input_name: blob})[0]

        return self._ctc_decode_greedy(output)

    def _recognize_paddle(self, char_img: np.ndarray) -> OcrResult:
        """PaddleOCR 原生模式识别"""
        # 转换为 BGR 如果是灰度图
        if len(char_img.shape) == 2:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)

        try:
            # 放大图像 (PaddleOCR 对小图像支持不好，统一放大3倍)
            h, w = char_img.shape[:2]
            scale = 3
            char_img = cv2.resize(char_img, (w * scale, h * scale))

            # 保存临时文件 (使用唯一名称避免冲突)
            import uuid
            temp_path = f'_temp_paddle_{uuid.uuid4().hex[:8]}.png'
            cv2.imwrite(temp_path, char_img)

            result = self.paddle_ocr.ocr(temp_path)

            os.remove(temp_path)

            if result and len(result) > 0 and result[0] and len(result[0]) > 0:
                # PaddleOCR 返回格式: [[bbox, (text, confidence)]]
                text = result[0][0][1][0] if result[0][0] else "未知"
                confidence = result[0][0][1][1] if result[0][0] else 0.0

                # 清理文本
                text = text.strip() if text else "未知"

                return OcrResult(text=text, confidence=float(confidence))
            else:
                return OcrResult(text="未知", confidence=0.0)

        except Exception as e:
            return OcrResult(text="待确认", confidence=0.0)

    def _recognize_easyocr(self, char_img: np.ndarray) -> OcrResult:
        """EasyOCR 模式识别"""
        try:
            # 转换为 BGR 如果是灰度图
            if len(char_img.shape) == 2:
                char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)

            # 保存临时文件
            temp_path = '_temp_char_img.png'
            cv2.imwrite(temp_path, char_img)

            result = self.easyocr_reader.readtext(temp_path)

            if result and len(result) > 0:
                # EasyOCR 返回: [(bbox, text, confidence), ...]
                # 取置信度最高的结果
                best_result = max(result, key=lambda x: x[2])
                text = best_result[1]
                confidence = best_result[2]

                # 清理文本，只保留汉字
                import re
                chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
                text = ''.join(chinese_chars) if chinese_chars else "待确认"

                os.remove(temp_path)
                return OcrResult(text=text, confidence=float(confidence))
            else:
                os.remove(temp_path)
                return OcrResult(text="未知", confidence=0.0)

        except Exception as e:
            return OcrResult(text="待确认", confidence=0.0)

    def recognize_batch(self, char_imgs: List[np.ndarray]) -> List[OcrResult]:
        """批量识别"""
        return [self.recognize(img) for img in char_imgs]

    def recognize_full_image(self, image_path: str) -> List[Tuple[List[float], OcrResult]]:
        """
        对整张图像进行 OCR 识别，返回所有检测到的文本区域

        Args:
            image_path: 图像文件路径

        Returns:
            List of ((bbox_points), OcrResult) tuples for each detected text region
        """
        if self.use_easyocr:
            return self._recognize_full_easyocr(image_path)
        elif self.use_onnx:
            # ONNX 模式需要外部提供全图 OCR，这里返回空
            return []
        else:
            return self._recognize_full_paddle(image_path)

    def _recognize_full_paddle(self, image_path: str) -> List[Tuple[List[float], OcrResult]]:
        """PaddleOCR 全图识别"""
        try:
            # 如果有 venv_python，使用 subprocess 调用
            if self.venv_python:
                return self._recognize_full_paddle_subprocess(image_path)

            # 否则直接使用导入的 paddle_ocr
            result = self.paddle_ocr.ocr(image_path)
            if not result or not result[0]:
                return []

            ocr_results = []
            for line in result[0]:
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = line[1][0] if line[1][0] else "未知"
                confidence = float(line[1][1]) if line[1][1] else 0.0
                ocr_results.append((bbox, OcrResult(text=text, confidence=confidence)))
            return ocr_results
        except Exception as e:
            return []

    def _recognize_full_paddle_subprocess(self, image_path: str) -> List[Tuple[List[float], OcrResult]]:
        """使用 venv_paddle 中的 PaddleOCR 进行全图识别"""
        import subprocess
        import json as json_module

        script = '''
import sys
import os
# 抑制 PaddlePaddle 的 warning 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from paddleocr import PaddleOCR
import json

# 完全禁用日志
import logging
logging.disable(logging.CRITICAL)

ocr = PaddleOCR(lang='ch', show_log=False)
result = ocr.ocr(sys.argv[1])

output = []
if result and result[0]:
    for line in result[0]:
        bbox = [[float(p[0]), float(p[1])] for p in line[0]]
        text = line[1][0] if line[1][0] else "未知"
        conf = float(line[1][1]) if line[1][1] else 0.0
        output.append({"bbox": bbox, "text": text, "confidence": conf})

print(json.dumps(output, ensure_ascii=False))
'''

        try:
            result = subprocess.run(
                [self.venv_python, '-c', script, image_path],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0 or not result.stdout.strip():
                return []

            # 解析 JSON（忽略 stderr 中的警告）
            json_str = result.stdout.strip()
            # 如果 stdout 开头有日志行，尝试找到 JSON 部分
            if not json_str.startswith('['):
                # 找到第一个 '[' 的位置
                idx = json_str.find('[')
                if idx >= 0:
                    json_str = json_str[idx:]
                else:
                    return []

            data = json_module.loads(json_str)
            return [(item["bbox"], OcrResult(text=item["text"], confidence=item["confidence"])) for item in data]
        except Exception as e:
            return []

    def _recognize_full_easyocr(self, image_path: str) -> List[Tuple[List[float], OcrResult]]:
        """EasyOCR 全图识别"""
        try:
            result = self.easyocr_reader.readtext(image_path)
            if not result:
                return []

            import re
            ocr_results = []
            for line in result:
                bbox = line[0]
                text = line[1]
                confidence = float(line[2])
                # 只保留汉字
                chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
                clean_text = ''.join(chinese_chars) if chinese_chars else "待确认"
                ocr_results.append((bbox, OcrResult(text=clean_text, confidence=confidence)))
            return ocr_results
        except Exception as e:
            return []

    @staticmethod
    def match_char_to_ocr_result(char_bbox: Tuple[int, int, int, int],
                                 ocr_results: List[Tuple[List[float], OcrResult]],
                                 iou_threshold: float = 0.3) -> OcrResult:
        """
        将裁切字符匹配到 OCR 结果

        Args:
            char_bbox: 字符边界框 (x, y, w, h)
            ocr_results: OCR 结果列表
            iou_threshold: IoU 阈值

        Returns:
            匹配的 OCR 结果，如果无匹配则返回默认结果
        """
        cx = char_bbox[0] + char_bbox[2] / 2
        cy = char_bbox[1] + char_bbox[3] / 2

        best_iou = iou_threshold
        best_result = OcrResult(text="待确认", confidence=0.0)

        for bbox, ocr_result in ocr_results:
            # bbox 是四个角点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # 计算 OCR 区域的中心点
            ocr_cx = (bbox[0][0] + bbox[2][0]) / 2
            ocr_cy = (bbox[0][1] + bbox[2][1]) / 2

            # 计算距离
            dist = ((cx - ocr_cx) ** 2 + (cy - ocr_cy) ** 2) ** 0.5

            # 如果距离在字符宽度的一半内，认为匹配
            char_diag = (char_bbox[2] ** 2 + char_bbox[3] ** 2) ** 0.5
            if dist < char_diag * 1.5:  # 允许一些容差
                if ocr_result.confidence > best_result.confidence:
                    best_result = ocr_result

        return best_result

    @staticmethod
    def match_chars_to_ocr_results(
        char_bboxes: List[Tuple[int, int, int, int]],
        ocr_results: List[Tuple[List[float], OcrResult]]
    ) -> List[OcrResult]:
        """
        将多个裁切字符匹配到 OCR 结果，并拆分多字符文本

        Args:
            char_bboxes: 字符边界框列表 [(x, y, w, h), ...]
            ocr_results: OCR 结果列表

        Returns:
            每个字符对应的 OCR 结果列表
        """
        if not char_bboxes:
            return []

        # 按 x 坐标排序字符
        sorted_indices = sorted(range(len(char_bboxes)),
                               key=lambda i: char_bboxes[i][0])

        # 按 OCR 区域的 x 坐标排序
        sorted_ocr = sorted(ocr_results,
                           key=lambda x: (x[0][0][0] + x[0][2][0]) / 2)

        # 为每个 OCR 区域，找到所有落在该区域内的字符
        ocr_char_assignments = []  # [(ocr_result, [char_indices])]

        for bbox, ocr_result in sorted_ocr:
            ocr_x1 = min(p[0] for p in bbox)
            ocr_x2 = max(p[0] for p in bbox)
            ocr_y1 = min(p[1] for p in bbox)
            ocr_y2 = max(p[1] for p in bbox)

            matching_chars = []
            for i, char_bbox in enumerate(char_bboxes):
                cx = char_bbox[0] + char_bbox[2] / 2
                cy = char_bbox[1] + char_bbox[3] / 2

                # 检查字符中心是否在 OCR 区域内
                if ocr_x1 <= cx <= ocr_x2 and ocr_y1 <= cy <= ocr_y2:
                    matching_chars.append(i)

            if matching_chars:
                ocr_char_assignments.append((ocr_result, matching_chars))

        # 构建结果列表
        results = [OcrResult(text="待确认", confidence=0.0) for _ in char_bboxes]

        for ocr_result, char_indices in ocr_char_assignments:
            text = ocr_result.text
            text_len = len(text) if text else 0
            num_chars = len(char_indices)

            if text_len == 0:
                continue

            # 计算每个字符应分配的字符数
            if num_chars == 0:
                continue

            # 拆分文本
            chars_per_slot = max(1, text_len // num_chars)
            remainder = text_len % num_chars

            char_idx = 0
            for i, char_idx_in_bbox in enumerate(char_indices):
                if char_idx >= text_len:
                    break

                # 计算这个字符应该分到的字符数
                count = chars_per_slot + (1 if i < remainder else 0)
                if count == 0:
                    count = 1

                # 取出一个字符（或多个如果 count > 1）
                assigned_text = text[char_idx:char_idx + min(count, len(text) - char_idx)]
                char_idx += len(assigned_text)

                # 分配结果
                result = OcrResult(
                    text=assigned_text,
                    confidence=ocr_result.confidence
                )
                results[char_idx_in_bbox] = result

        return results


class MetadataGenerator:
    """
    元数据生成器

    为每个裁切单字生成标准化的 JSON 描述文件，
    形成完整的"数字资产档案"。
    """

    def __init__(self,
                 author: str = "未知",
                 work_name: str = "未知",
                 output_dir: str = "."):
        self.author = author
        self.work_name = work_name
        self.output_dir = output_dir
        self.counter = 1
        os.makedirs(output_dir, exist_ok=True)

    def generate(self,
                 char_img: np.ndarray,
                 ocr_result: OcrResult,
                 bbox: Tuple[int, int, int, int],
                 image_path: str) -> dict:
        """
        生成元数据

        Args:
            char_img: 裁切图像
            ocr_result: OCR 识别结果
            bbox: (x, y, w, h)
            image_path: 图像文件路径

        Returns:
            dict: 元数据字典
        """
        x, y, w, h = bbox
        char_label = ocr_result.text if ocr_result.confidence > self.confidence_threshold else "待确认"

        # 生成 ID
        char_id = f"{self.author}_{self.work_name}_{char_label}_{self.counter:03d}"
        self.counter += 1

        # 构建元数据
        metadata = {
            "id": char_id,
            "author": self.author,
            "source_work": self.work_name,
            "character": char_label,
            "ai_confidence": round(ocr_result.confidence, 4),
            "confidence_flag": "verified" if ocr_result.confidence > self.confidence_threshold else "pending_review",
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "image_format": "BGRA_Alpha_Premultiplied",
            "image_filename": os.path.basename(image_path),
            "timestamp": self._get_timestamp()
        }

        return metadata

    def save_metadata(self, metadata: dict, json_path: str):
        """保存元数据到 JSON 文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def _get_timestamp(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def confidence_threshold(self) -> float:
        """获取置信度阈值（供外部访问）"""
        return 0.5


def integrate_ocr_to_extractor():
    """
    演示：如何将 OCR 打标集成到 CalligraphyExtractor

    模式1: ONNX 模式 (需要先转换模型)
    ```python
    from calligraphy_extractor import CalligraphyExtractor
    from ocr_auto_tagger import OcrAutoTagger, MetadataGenerator

    extractor = CalligraphyExtractor(output_dir="output_chars")

    # ONNX 模式
    tagger = OcrAutoTagger(
        model_path="ch_PP-OCRv4_rec.onnx",
        dict_path="ppocr_keys_v1.txt",
        use_onnx=True
    )

    metadata_gen = MetadataGenerator(author="苏轼", work_name="祭黄几道文")
    ```

    模式2: PaddleOCR 原生模式 (自动下载模型)
    ```python
    from calligraphy_extractor import CalligraphyExtractor
    from ocr_auto_tagger import OcrAutoTagger, MetadataGenerator

    extractor = CalligraphyExtractor(output_dir="output_chars")

    # PaddleOCR 原生模式 (推荐，无需手动转换模型)
    tagger = OcrAutoTagger(use_onnx=False)

    metadata_gen = MetadataGenerator(author="苏轼", work_name="祭黄几道文")
    ```
    """
    pass


# --- 运行测试 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="书法单字 OCR 识别测试")
    parser.add_argument("image", help="单字图像路径")
    parser.add_argument("-m", "--model", help="ONNX 模型路径 (可选)")
    parser.add_argument("-d", "--dict", help="字典文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--onnx", action="store_true", help="强制使用ONNX模式")
    parser.add_argument("--easyocr", action="store_true", help="使用EasyOCR模式 (备用方案)")

    args = parser.parse_args()

    # 加载图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"[!] 无法读取图像: {args.image}")
        exit(1)

    try:
        if args.easyocr:
            # EasyOCR 模式
            tagger = OcrAutoTagger(use_easyocr=True)
            print("[*] 使用 EasyOCR 模式")
        elif args.onnx or (args.model and os.path.exists(args.model)):
            # ONNX 模式
            tagger = OcrAutoTagger(
                model_path=args.model,
                dict_path=args.dict,
                use_onnx=True,
                use_opencv_dnn=True
            )
            print("[*] 使用 ONNX 模式")
        else:
            # PaddleOCR 原生模式
            tagger = OcrAutoTagger(use_onnx=False)
            print("[*] 使用 PaddleOCR 原生模式 (会自动下载模型)")

        # 识别
        result = tagger.recognize(img)

        print(f"[*] 识别结果: {result.text}")
        print(f"[*] 置信度: {result.confidence:.4f}")

    except Exception as e:
        print(f"[!] 错误: {e}")
        exit(1)
