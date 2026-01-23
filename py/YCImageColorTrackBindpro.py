import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
import math
from typing import Tuple, Optional, List, Dict, Any

# --- 基础工具函数 ---

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def tensor2cv2(image_tensor):
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    img_np = image_tensor.detach().cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    # 转换 HWC/CHW
    if img_np.ndim == 3 and img_np.shape[-1] in (3, 4):
        img_hwc = img_np
    elif img_np.ndim == 3 and img_np.shape[0] in (3, 4):
        img_hwc = np.transpose(img_np, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image layout")
    
    img_u8 = (img_hwc * 255.0).astype(np.uint8)
    # 处理RGBA转BGR (OpenCV默认处理)
    if img_u8.shape[2] == 4:
        # 如果是RGBA，我们这里只取RGB做追踪，但后续处理要保留Alpha
        pass 
    
    # 返回RGB转BGR的结果用于cv2处理
    if img_u8.shape[2] == 4:
        return cv2.cvtColor(img_u8, cv2.COLOR_RGBA2BGR)
    else:
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

def parse_color(color_str: str) -> Tuple[int, int, int]:
    color_str = color_str.strip().lstrip('#')
    if ',' in color_str:
        parts = color_str.split(',')
    elif ' ' in color_str:
        parts = color_str.split()
    elif len(color_str) == 6:
        return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
    else:
        parts = [color_str]
    
    if len(parts) >= 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    return (255, 0, 0) # Default Red

# --- 核心算法 ---

def apply_motion_blur(image_pil: Image.Image, velocity_x: float, velocity_y: float, strength: float) -> Image.Image:
    """应用基于速度的方向性运动模糊"""
    if strength <= 0 or (abs(velocity_x) < 1 and abs(velocity_y) < 1):
        return image_pil

    # 计算模糊核大小和角度
    magnitude = math.sqrt(velocity_x**2 + velocity_y**2)
    kernel_size = int(magnitude * strength)
    
    # 限制核大小，避免过大导致崩溃或极度卡顿
    if kernel_size < 2: return image_pil
    if kernel_size > 50: kernel_size = 50
    
    # 确保核大小为奇数
    if kernel_size % 2 == 0: kernel_size += 1

    # 转换PIL到CV2 (保留Alpha通道)
    img_np = np.array(image_pil)
    
    # 检查图像尺寸，避免对过小图像应用模糊
    if img_np.shape[0] < kernel_size or img_np.shape[1] < kernel_size:
        return image_pil
    
    # 创建方向性模糊核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    # 归一化向量
    vec_x = velocity_x / (magnitude + 1e-5)
    vec_y = velocity_y / (magnitude + 1e-5)
    
    # 画线生成核
    # 线的起点和终点
    p1 = (int(center - vec_x * (kernel_size/2)), int(center - vec_y * (kernel_size/2)))
    p2 = (int(center + vec_x * (kernel_size/2)), int(center + vec_y * (kernel_size/2)))
    
    # 确保坐标在有效范围内
    p1 = (max(0, min(kernel_size-1, p1[0])), max(0, min(kernel_size-1, p1[1])))
    p2 = (max(0, min(kernel_size-1, p2[0])), max(0, min(kernel_size-1, p2[1])))
    
    cv2.line(kernel, p1, p2, 1.0, 1)
    
    # 归一化核
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        # 如果核为空，创建一个简单的中心点核
        kernel[center, center] = 1.0
    
    # 应用滤波
    try:
        blurred = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(blurred)
    except Exception as e:
        # 如果滤波失败，返回原图像
        print(f"Motion blur failed: {e}")
        return image_pil

def find_contour_metrics(frame_cv2: np.ndarray, target_color_rgb: Tuple[int, int, int], 
                        tolerance: int, processing_width: int,
                        prev_center: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    """
    在降采样后的图像上查找：中心点、旋转角度、面积
    """
    # 1. 降采样 (Downsampling)
    h, w = frame_cv2.shape[:2]
    scale_ratio = 1.0
    work_frame = frame_cv2
    
    if processing_width > 0 and w > processing_width:
        scale_ratio = processing_width / w
        new_h = int(h * scale_ratio)
        work_frame = cv2.resize(frame_cv2, (processing_width, new_h), interpolation=cv2.INTER_AREA)

    # 2. HSV Masking
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    target_bgr = np.uint8([[target_color_rgb[::-1]]])
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    
    th, ts, tv = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
    
    # 动态容差策略
    h_tol = max(5, int(tolerance * 0.5))
    sv_tol = max(40, int(tolerance * 1.5))
    
    lower = np.array([max(0, th - h_tol), max(20, ts - sv_tol), max(20, tv - sv_tol)], dtype=np.uint8)
    upper = np.array([min(180, th + h_tol), min(255, ts + sv_tol), min(255, tv + sv_tol)], dtype=np.uint8)
    
    # 环绕色相处理
    if lower[0] > upper[0]: 
        # 这种情况通常不需要特殊处理，除非跨越0/180，这里简化逻辑，通常直接clip就好
        pass 
        
    mask = cv2.inRange(hsv, lower, upper)
    
    # 形态学去噪
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel_small, iterations=1)
    mask = cv2.dilate(mask, kernel_large, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    valid_candidates = []
    
    # 将上一帧的中心点映射到缩小后的坐标系
    prev_center_scaled = None
    if prev_center:
        prev_center_scaled = (prev_center[0] * scale_ratio, prev_center[1] * scale_ratio)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10: continue
        
        # 最小外接矩形 (Center(x,y), (w,h), angle)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        
        # 简单的评分机制
        score = area
        if prev_center_scaled:
            dist = math.sqrt((cx - prev_center_scaled[0])**2 + (cy - prev_center_scaled[1])**2)
            # 距离惩罚：距离越远，分数扣得越多
            score = area - (dist * dist * 0.5) 
            
        valid_candidates.append({
            'center': (cx / scale_ratio, cy / scale_ratio), # 还原回原分辨率
            'angle': angle,
            'area': area / (scale_ratio * scale_ratio),     # 还原回原分辨率
            'score': score,
            'rect_size': (rw, rh)
        })
    
    if not valid_candidates:
        return None
        
    # 选分最高的
    best = max(valid_candidates, key=lambda x: x['score'])
    return best

def smooth_value(current, previous, factor):
    """通用的一阶平滑"""
    if previous is None: return current
    return previous * (1 - factor) + current * factor

def interpolate_missing(data_list):
    """线性插值填补None"""
    n = len(data_list)
    for i in range(n):
        if data_list[i] is None:
            # Find prev
            prev_idx = i - 1
            while prev_idx >= 0 and data_list[prev_idx] is None: prev_idx -= 1
            # Find next
            next_idx = i + 1
            while next_idx < n and data_list[next_idx] is None: next_idx += 1
            
            if prev_idx >= 0 and next_idx < n:
                start = data_list[prev_idx]
                end = data_list[next_idx]
                ratio = (i - prev_idx) / (next_idx - prev_idx)
                data_list[i] = start + (end - start) * ratio
            elif prev_idx >= 0:
                data_list[i] = data_list[prev_idx]
            elif next_idx < n:
                data_list[i] = data_list[next_idx]
            else:
                data_list[i] = 0.0 # Default
    return data_list

# --- 节点定义 ---

class YCImageColorTrackBindPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "video_frames": ("IMAGE",),
                "target_color": ("STRING", {"default": "#FF0000", "multiline": False}),
                "color_tolerance": ("INT", {"default": 30, "min": 0, "max": 255}),
                "smoothing_factor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01}),
                "processing_width": ("INT", {"default": 640, "min": 100, "max": 4096, "step": 10, "label": "Tracking Resolution (px)"}),
                
                # 位置微调
                "anchor_offset_x": ("INT", {"default": 0}),
                "anchor_offset_y": ("INT", {"default": 0}),
                "base_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                
                # 高级特性开关
                "enable_rotation": ("BOOLEAN", {"default": False}),
                "rotation_offset": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0}),
                
                "enable_auto_scale": ("BOOLEAN", {"default": False, "label": "Auto Z-Depth Scale"}),
                
                "motion_blur_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_frames",)
    FUNCTION = "process"
    CATEGORY = "YC_VideoCutHelper/Image"
    
    def process(self, image, video_frames, target_color, color_tolerance, smoothing_factor, processing_width,
                anchor_offset_x, anchor_offset_y, base_scale, 
                enable_rotation, rotation_offset, enable_auto_scale, motion_blur_strength, mask=None):
        
        # 1. 准备输入
        target_color_rgb = parse_color(target_color)
        if image.shape[0] > 1: image = image[0:1]
        input_pil_base = tensor2pil(image)
        
        if mask is not None:
            if mask.shape[0] > 1: mask = mask[0:1]
            mask_np = mask.cpu().numpy()
            if mask_np.ndim == 3: mask_np = mask_np[0]
            # Resize mask to image size if needed
            if mask_np.shape != (input_pil_base.height, input_pil_base.width):
                m_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(input_pil_base.size, Image.Resampling.LANCZOS)
                mask_np = np.array(m_img).astype(np.float32) / 255.0
            
            # Apply Mask to Alpha
            if input_pil_base.mode != 'RGBA': input_pil_base = input_pil_base.convert('RGBA')
            r, g, b, a = input_pil_base.split()
            a_np = np.array(a).astype(np.float32) / 255.0
            new_a = (a_np * mask_np * 255).astype(np.uint8)
            input_pil_base.putalpha(Image.fromarray(new_a))
        else:
             if input_pil_base.mode != 'RGBA': input_pil_base = input_pil_base.convert('RGBA')

        num_frames = video_frames.shape[0]
        
        # 2. 追踪阶段 (收集原始数据)
        raw_centers = []
        raw_angles = []
        raw_areas = []
        
        last_center = None
        
        for i in range(num_frames):
            frame = tensor2cv2(video_frames[i:i+1])
            metrics = find_contour_metrics(frame, target_color_rgb, color_tolerance, processing_width, last_center)
            
            if metrics:
                raw_centers.append(metrics['center'])
                raw_angles.append(metrics['angle'])
                raw_areas.append(metrics['area'])
                last_center = metrics['center']
            else:
                raw_centers.append(None)
                raw_angles.append(None)
                raw_areas.append(None)
        
        # 3. 数据清洗与插值
        # 分离XY以便插值
        xs = [p[0] if p else None for p in raw_centers]
        ys = [p[1] if p else None for p in raw_centers]
        
        fixed_xs = interpolate_missing(xs)
        fixed_ys = interpolate_missing(ys)
        fixed_angles = interpolate_missing(raw_angles)
        fixed_areas = interpolate_missing(raw_areas)
        
        # 确定基准面积 (取所有有效面积的中位数，避免初始噪点)
        valid_areas = [a for a in fixed_areas if a > 0]
        base_area_ref = np.median(valid_areas) if valid_areas else 100.0
        
        # 4. 平滑与生成
        output_tensors = []
        
        # 状态变量
        curr_x, curr_y = fixed_xs[0], fixed_ys[0]
        curr_angle = fixed_angles[0]
        curr_area = fixed_areas[0]
        
        # 运动模糊用的前一帧位置
        prev_x, prev_y = curr_x, curr_y
        
        # 计算图像初始中心
        img_w_orig, img_h_orig = input_pil_base.size
        
        for i in range(num_frames):
            # --- 平滑逻辑 ---
            # 动态调整平滑系数：如果有剧烈运动（距离变化大），减少平滑以防滞后
            # 这里简化处理，直接应用 EMA
            alpha = 1.0 - smoothing_factor
            
            # 保存当前位置用于运动模糊计算
            prev_x, prev_y = curr_x, curr_y
            
            curr_x = curr_x * (1-alpha) + fixed_xs[i] * alpha
            curr_y = curr_y * (1-alpha) + fixed_ys[i] * alpha
            
            # 角度处理 (注意 OpenCV minAreaRect 角度可能是 0-90 或 0-180，这里直接平滑，假设不会突变)
            curr_angle = curr_angle * (1-alpha) + fixed_angles[i] * alpha
            
            # 面积处理
            curr_area = curr_area * (1-alpha) + fixed_areas[i] * alpha
            
            # --- 变换计算 ---
            
            # 1. 缩放
            final_scale = base_scale
            if enable_auto_scale and base_area_ref > 0:
                # 面积比的平方根 = 边长比
                scale_mult = math.sqrt(curr_area / base_area_ref)
                # 限制动态缩放范围，防止无限大或消失
                scale_mult = max(0.2, min(5.0, scale_mult))
                final_scale *= scale_mult
                
            new_w = int(img_w_orig * final_scale)
            new_h = int(img_h_orig * final_scale)
            if new_w < 1 or new_h < 1: new_w, new_h = 1, 1
            
            img_transformed = input_pil_base.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 2. 旋转
            if enable_rotation:
                # 注意：OpenCV的角度通常是顺时针或逆时针，需根据实际情况调整符号
                # rotation_offset 用于修正初始方向
                total_angle = curr_angle + rotation_offset
                img_transformed = img_transformed.rotate(total_angle, expand=True, resample=Image.Resampling.BICUBIC)
            
            # 3. 运动模糊
            if motion_blur_strength > 0 and i > 0:
                # 计算速度向量（当前位置 - 前一帧位置）
                vx = curr_x - prev_x
                vy = curr_y - prev_y
                img_transformed = apply_motion_blur(img_transformed, vx, vy, motion_blur_strength)
            
            # 4. 合成
            frame_tensor = video_frames[i:i+1]
            frame_pil = tensor2pil(frame_tensor)
            bg_w, bg_h = frame_pil.size
            
            # 计算粘贴坐标 (中心对齐)
            paste_w, paste_h = img_transformed.size
            paste_x = int(curr_x - paste_w / 2 + anchor_offset_x)
            paste_y = int(curr_y - paste_h / 2 + anchor_offset_y)
            
            # PIL粘贴处理透明度
            comp = frame_pil.copy().convert('RGBA')
            comp.paste(img_transformed, (paste_x, paste_y), img_transformed)
            
            output_tensors.append(pil2tensor(comp.convert('RGB')))
            
        return (torch.cat(output_tensors, dim=0),)

NODE_CLASS_MAPPINGS = {
    "YCImageColorTrackBindPro": YCImageColorTrackBindPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCImageColorTrackBindPro": "Image Color Track Bind (Pro)",
}