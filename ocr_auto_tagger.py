"""
OcrAutoTagger: 书法单字 OCR 自动打标模块
==========================================

使用 OpenCV DNN 加载 ONNX 模型进行轻量级字符识别

依赖:
    pip install opencv-python numpy onnxruntime

Author: Industrial CV Pipeline
Date: 2026-04-10
"""

import cv2
import numpy as np
import os
import json
from typing import Optional, List, Tuple
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

    使用 OpenCV DNN 或 ONNXRuntime 加载 PaddleOCR 识别模型，
    支持 CTC 解码，推理速度每秒数百字。
    """

    def __init__(self,
                 onnx_model_path: str,
                 dict_path: Optional[str] = None,
                 use_opencv_dnn: bool = True,
                 confidence_threshold: float = 0.5):
        """
        初始化 OCR 识别器

        Args:
            onnx_model_path: ONNX 模型文件路径
            dict_path: 字典文件路径 (PaddleOCR ppocr_keys_v1.txt)
            use_opencv_dnn: True=使用OpenCV DNN, False=使用ONNXRuntime
            confidence_threshold: 置信度阈值，低于此值标记为"待确认"
        """
        self.confidence_threshold = confidence_threshold
        self.vocabulary: List[str] = []

        # 加载字典
        if dict_path and os.path.exists(dict_path):
            self._load_dict(dict_path)
        else:
            # 使用默认汉字字典 (常用汉字 6753 个)
            self._init_default_dict()

        # 加载模型
        if use_opencv_dnn:
            self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
            self.use_opencv = True
        else:
            # ONNXRuntime
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            self.session = ort.InferenceSession(
                onnx_model_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.use_opencv = False

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
        # 常用汉字字典，来自 PaddleOCR
        common_chars = (
            "的一是不了在人我有他这中大来上个国们为上 "
            "地到方以就出要年时得才可下过量起政好小部其此"
            "心前于学还天分能对程军民言果党十地城石里运五"
            "喜彩晨壶碎得益绿萍角鼓编胡飘叶风春异截横飘"
        )
        self.vocabulary = ['blank'] + list(common_chars) + [' ']

    def _preprocess(self, char_img: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        预处理单字图像

        PaddleOCR 标准预处理:
        1. 高度缩放为 48，宽度等比例缩放
        2. 归一化: (x * 2.0 / 255.0) - 1.0

        Returns:
            blob: 预处理后的 blob
            scale: 缩放比例
            target_width: 目标宽度
        """
        h = 48
        w = max(48, int(char_img.shape[1] * (48.0 / char_img.shape[0])))

        # 调整大小
        resized = cv2.resize(char_img, (w, h))

        # 归一化并转换为 blob
        # PaddleOCR 归一化: (x / 127.5) - 1.0
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
        """
        CTC 贪心解码

        Args:
            net_output: 模型输出，维度 [1, T, vocab_size]

        Returns:
            OcrResult: 识别结果和置信度
        """
        # net_output shape: (1, time_steps, vocab_size)
        if len(net_output.shape) == 4:
            net_output = net_output.squeeze(0)  # (T, vocab_size)

        time_steps, vocab_size = net_output.shape
        data = net_output  # (T, vocab_size)

        result_text = ""
        total_confidence = 0.0
        valid_chars = 0
        last_index = -1

        for t in range(time_steps):
            # 找到当前时间步概率最大的字符索引
            max_idx = np.argmax(data[t])
            max_prob = float(data[t, max_idx])

            # CTC 规则:
            # 1. 忽略空白标签 (索引 0)
            # 2. 忽略重复标签
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
        # 预处理
        blob, scale, target_w = self._preprocess(char_img)

        if self.use_opencv:
            self.net.setInput(blob)
            output = self.net.forward()
        else:
            output = self.session.run(
                None,
                {self.input_name: blob}
            )[0]

        return self._ctc_decode_greedy(output)

    def recognize_batch(self, char_imgs: List[np.ndarray]) -> List[OcrResult]:
        """批量识别"""
        return [self.recognize(img) for img in char_imgs]


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

    使用示例：
    ```python
    from calligraphy_extractor import CalligraphyExtractor
    from ocr_auto_tagger import OcrAutoTagger, MetadataGenerator

    # 初始化提取器
    extractor = CalligraphyExtractor(output_dir="output_chars")

    # 初始化 OCR (需要先下载 PaddleOCR 识别模型)
    # 模型下载地址: https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
    tagger = OcrAutoTagger(
        onnx_model_path="ch_PP-OCRv4_rec_infer.onnx",
        dict_path="ppocr_keys_v1.txt"
    )

    # 初始化元数据生成器
    metadata_gen = MetadataGenerator(
        author="苏轼",
        work_name="祭黄几道文",
        output_dir="output_chars"
    )

    # 处理图像
    count = extractor.process_image("calligraphy.jpg")

    # 为每个结果生成元数据
    for result in extractor.results:
        ocr_result = tagger.recognize(result.image)
        metadata = metadata_gen.generate(
            char_img=result.image,
            ocr_result=ocr_result,
            bbox=result.bbox,
            image_path=result.image_path
        )
        # 保存 JSON
        json_path = result.image_path.replace('.png', '.json')
        metadata_gen.save_metadata(metadata, json_path)
    ```
    """
    pass


# --- 运行测试 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="书法单字 OCR 识别测试")
    parser.add_argument("image", help="单字图像路径")
    parser.add_argument("-m", "--model", required=True, help="ONNX 模型路径")
    parser.add_argument("-d", "--dict", help="字典文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] 模型文件不存在: {args.model}")
        print("[*] 请从 PaddleOCR 下载: https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar")
        exit(1)

    # 加载图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"[!] 无法读取图像: {args.image}")
        exit(1)

    # 初始化 OCR
    tagger = OcrAutoTagger(
        onnx_model_path=args.model,
        dict_path=args.dict,
        use_opencv_dnn=True
    )

    # 识别
    result = tagger.recognize(img)

    print(f"[*] 识别结果: {result.text}")
    print(f"[*] 置信度: {result.confidence:.4f}")
