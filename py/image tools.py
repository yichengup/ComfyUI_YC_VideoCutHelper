from torch import Tensor
import torch

import comfy.utils

from .utils import select_indexes_from_str, convert_str_to_indexes, select_indexes, select_description

class SelectImages:
    """图片索引选择节点 - 支持单个索引、多个索引、范围选择等多种格式"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "image": ("IMAGE",),
                    "indexes": ("STRING", {"default": "0", "multiline": True}),
                    "err_if_missing": ("BOOLEAN", {"default": True}),
                    "err_if_empty": ("BOOLEAN", {"default": True}),
                },
            }
    
    DESCRIPTION = select_description
    CATEGORY = "YC_VideoCutHelper/Image Tools"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_images",)
    FUNCTION = "select"

    def select(self, image: Tensor, indexes: str, err_if_missing: bool, err_if_empty: bool):
        """选择图片索引
        
        Args:
            image: 输入的图片tensor
            indexes: 索引字符串，支持多种格式
            err_if_missing: 索引超出范围时是否报错
            err_if_empty: 没有选中图片时是否报错
            
        Returns:
            选中的图片tensor
        """
        try:
            selected_images = select_indexes_from_str(
                input_obj=image, 
                indexes=indexes,
                err_if_missing=err_if_missing, 
                err_if_empty=err_if_empty
            )
            return (selected_images,)
        except Exception as e:
            print(f"SelectImages节点错误: {str(e)}")
            print(f"输入图片形状: {image.shape}")
            print(f"索引字符串: '{indexes}'")
            raise e


class SelectImagesAdvanced:
    """高级图片索引选择节点 - 提供更多选择选项和预设"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "image": ("IMAGE",),
                    "selection_mode": (["custom", "first_n", "last_n", "every_nth", "random"], {"default": "custom"}),
                    "indexes": ("STRING", {"default": "0", "multiline": True}),
                    "n_count": ("INT", {"default": 5, "min": 1, "max": 1000, "step": 1}),
                    "step": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "step": 1}),
                    "err_if_missing": ("BOOLEAN", {"default": True}),
                    "err_if_empty": ("BOOLEAN", {"default": True}),
                },
            }
    
    DESCRIPTION = """
Advanced Image Index Selection Node

Selection Mode:
- custom: Custom index string
- first_n: Select the first N images
- last_n: Select the last N images
- every_nth: Select every N images
- random: Select N images at random
    """
    CATEGORY = "YC_VideoCutHelper/Image Tools"

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected_images", "selected_indexes")
    FUNCTION = "select_advanced"

    def select_advanced(self, image: Tensor, selection_mode: str, indexes: str, 
                        n_count: int, step: int, seed: int,
                        err_if_missing: bool, err_if_empty: bool):
        """高级选择功能"""
        total_frames = image.shape[0]
        
        # 根据选择模式生成索引字符串
        if selection_mode == "first_n":
            actual_indexes = f"0-{min(n_count-1, total_frames-1)}"
        elif selection_mode == "last_n":
            start_idx = max(0, total_frames - n_count)
            actual_indexes = f"{start_idx}-{total_frames-1}"
        elif selection_mode == "every_nth":
            actual_indexes = f"0-{total_frames-1}:{step}"
        elif selection_mode == "random":
            import random
            random.seed(seed)
            available_indexes = list(range(total_frames))
            selected_count = min(n_count, total_frames)
            random_indexes = sorted(random.sample(available_indexes, selected_count))
            actual_indexes = ",".join(map(str, random_indexes))
        else:  # custom
            actual_indexes = indexes
        
        try:
            selected_images = select_indexes_from_str(
                input_obj=image, 
                indexes=actual_indexes,
                err_if_missing=err_if_missing, 
                err_if_empty=err_if_empty
            )
            
            return (selected_images, actual_indexes)
        except Exception as e:
            print(f"SelectImagesAdvanced节点错误: {str(e)}")
            print(f"输入图片形状: {image.shape}")
            print(f"选择模式: {selection_mode}")
            print(f"实际索引字符串: '{actual_indexes}'")
            raise e


# 节点注册
NODE_CLASS_MAPPINGS = {
    "SelectImages": SelectImages,
    "SelectImagesAdvanced": SelectImagesAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectImages": "Select Images",
    "SelectImagesAdvanced": "Select Images Advanced",
}
