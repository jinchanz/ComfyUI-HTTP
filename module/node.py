"""
节点入口文件 - 从子模块导入所有节点
"""
from .nodes import (
    COMMON_HTTP_REQUEST,
    POLLING_HTTP_REQUEST,
    LLMImageGenerate,
    LLMSmartGenerate,
    LLMResponseImageParser,
    TextConcatenate,
    ImageBatchMerge,
)

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "COMMON_HTTP_REQUEST": COMMON_HTTP_REQUEST,
    "LLMImageGenerate": LLMImageGenerate,
    "LLMSmartGenerate": LLMSmartGenerate,
    "LLMResponseImageParser": LLMResponseImageParser,
    "POLLING_HTTP_REQUEST": POLLING_HTTP_REQUEST,
    "MaletteTextConcatenate": TextConcatenate,
    "MaletteImageBatchMerge": ImageBatchMerge,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "COMMON_HTTP_REQUEST": "通用HTTP请求",
    "LLMImageGenerate": "LLM 图像生成",
    "LLMSmartGenerate": "LLM 智能生成",
    "LLMResponseImageParser": "LLM响应图片解析",
    "POLLING_HTTP_REQUEST": "轮询HTTP请求",
    "MaletteTextConcatenate": "文本拼接",
    "MaletteImageBatchMerge": "图片批次合并",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
