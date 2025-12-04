import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import re
from typing import Optional

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from torchvision import transforms as T
import numpy as np
import cv2
try:
    from ultralytics.models.sam import Predictor as SAMPredictor
except ImportError:
    SAMPredictor = None
from ultralytics import SAM
from PIL import Image


def load_grounding_dino(config_path, checkpoint_path, device="cpu"):
    cfg = SLConfig.fromfile(config_path)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/erlin/work/dobot_python_api/models/bert-base-uncased",
        local_files_only=True  
    )
    print("Building model...")
    model = build_model(cfg)
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Loading state dict...")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    print(f"Moving model to {device}...")
    model.to(device)
    print("Model loaded.")
    return model


# ========= 4. 工具函数 =========
def detect_mask(model, image_bgr, prompt, box_thresh=0.35, text_thresh=0.25,show=False):
    '''
    args:
        model: the model to detect
        image_bgr: the image to detect
        prompt: the prompt to detect
        box_thresh: the threshold to filter the box
        text_thresh: the threshold to filter the text
        show: whether to show the mask
    return:
        mask: the mask of the detected object

    '''
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image_bgr).unsqueeze(0).to("cuda")
    caption = prompt.lower().strip()
    if not caption.endswith("."):
        caption += "."

    with torch.no_grad():
        outputs = model(image, captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  
    boxes = outputs["pred_boxes"].cpu()[0]              

    # 过滤低分数目标
    valid_detections = logits.max(dim=1)[0] > box_thresh
    logits, boxes = logits[valid_detections], boxes[valid_detections]           
    
    if len(boxes) == 0:
        return None
    
    # 取分数最高的目标
    best_idx = logits.max(dim=1)[0].argmax()
    box = boxes[best_idx].tolist()
    
    if len(box) == 4:
        cx, cy, w, h = box
    else:
        cx, cy, w, h = box[:4] 
    
    print(cx, cy, w, h)



    h_img, w_img = image_bgr.shape[:2]
    
    # 将归一化坐标转换为像素坐标
    x1 = int((cx - w/2) * w_img)-20
    y1 = int((cy - h/2) * h_img)-20
    x2 = int((cx + w/2) * w_img)+20
    y2 = int((cy + h/2) * h_img)+20
    



    x1 = max(0, min(x1, w_img))
    y1 = max(0, min(y1, h_img))
    x2 = max(0, min(x2, w_img))
    y2 = max(0, min(y2, h_img))


    bbox = [x1, y1, x2, y2]

    if show:
        # 创建图像副本避免修改原图
        debug_image = image_bgr.copy()
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("detection_result", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    return bbox

    # mask = np.zeros((h_img, w_img), dtype=np.uint8)
    # mask[y1:y2, x1:x2] = 255

    # # 可选：绘制边界框（仅在show=True时）
    # if show:
    #     # 创建图像副本避免修改原图
    #     debug_image = image_bgr.copy()
    #     cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     cv2.imshow("detection_result", debug_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    # if show:
    #     cv2.imshow('mask', mask)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # return mask

def segment_image(image_rgb, bbox):
    """
    使用 SAM 进行分割
    image_rgb: RGB 图像数组
    bbox: 检测框 [x1, y1, x2, y2]
    """
    if SAMPredictor is None:
        raise ImportError("未找到SAMPredictor，请安装ultralytics>=8.1并确保可用。")
    # predictor = SAMPredictor(overrides=dict(model='/home/erlin/TCP-IP-Python-V4/sam_b.pt'))
    predictor = SAMPredictor(overrides=dict(model='/home/erlin/TCP-IP-Python-V4/sam2.1_b.pt', save=False, project=None, name=None, exist_ok=True, verbose=False))
    predictor.set_image(image_rgb)
    predictor.set_image(image_rgb)

    # 使用检测框进行分割
    results = predictor(bboxes=[bbox])
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # 保存分割结果
    # cv2.imwrite("segmentation_mask.png", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mask



# 全局模型变量，避免重复加载
_global_dino_model = None
_global_sam_predictor = None
_global_qwen_model = None
_global_qwen_processor = None
_global_qwen_device = None
_global_qwen_model_id = None

def get_mask_from_GD(rgb_image, prompt):
    global _global_dino_model, _global_sam_predictor
    
    # 只在第一次调用时加载模型
    if _global_dino_model is None:
        CONFIG_PATH = "/home/erlin/work/labgrasp/GroundingDino/GroundingDINO_SwinT_OGC.py"
        CHECKPOINT_PATH = "/home/erlin/work/labgrasp/GroundingDino/groundingdino_swint_ogc.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"   
        _global_dino_model = load_grounding_dino(CONFIG_PATH, CHECKPOINT_PATH, device)
        print("GroundingDINO模型已加载")
    
    # 只在第一次调用时初始化SAM
    if _global_sam_predictor is None:
        print("SAM模型正在加载...")
        # _global_sam_predictor = SAMPredictor(overrides=dict(model='/home/erlin/TCP-IP-Python-V4/sam2.1_b.pt', save=False, project=None, name=None, exist_ok=True, verbose=False))
        _global_sam_predictor = SAM(model="/home/erlin/TCP-IP-Python-V4/sam2.1_b.pt")
        print("SAM模型已加载")
    
    bbox = detect_mask(_global_dino_model, rgb_image, prompt=prompt, show=False)  # 关闭显示以提高性能
    if bbox is None:
        print("GroundingDINO未能检测到有效的目标边界框。")
        return None
    
    output_mask = segment_image_fast(rgb_image, bbox, _global_sam_predictor)
    return output_mask


def segment_image_fast(image_rgb, bbox, predictor):
    """快速分割版本，复用已加载的predictor"""
    # predictor.set_image(image_rgb)
    results = predictor(image_rgb, bboxes=[bbox])
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


