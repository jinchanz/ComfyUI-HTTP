"""
工具节点 - 文本拼接和图片批次合并
"""
import torch


class TextConcatenate:
    """文本拼接节点 - 支持多个文本输入和多种分隔符"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"default": "", "multiline": True, "tooltip": "第一段文本"}),
            },
            "optional": {
                "text2": ("STRING", {"default": "", "multiline": True, "tooltip": "第二段文本"}),
                "text3": ("STRING", {"default": "", "multiline": True, "tooltip": "第三段文本"}),
                "text4": ("STRING", {"default": "", "multiline": True, "tooltip": "第四段文本"}),
                "text5": ("STRING", {"default": "", "multiline": True, "tooltip": "第五段文本"}),
                "separator": (
                    ["newline", "comma", "space", "comma_space", "semicolon", "pipe", "custom"],
                    {
                        "default": "newline",
                        "tooltip": "分隔符类型"
                    }
                ),
                "custom_separator": ("STRING", {"default": "", "tooltip": "自定义分隔符（当 separator=custom 时使用）"}),
                "prefix": ("STRING", {"default": "", "tooltip": "每段文本的前缀"}),
                "suffix": ("STRING", {"default": "", "tooltip": "每段文本的后缀"}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "跳过空文本"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concatenate"
    CATEGORY = "api node/utils"

    def concatenate(
        self,
        text1,
        text2="",
        text3="",
        text4="",
        text5="",
        separator="newline",
        custom_separator="",
        prefix="",
        suffix="",
        skip_empty=True,
    ):
        """拼接多个文本"""

        # 收集所有文本
        texts = [text1, text2, text3, text4, text5]

        # 过滤空文本（如果启用）
        if skip_empty:
            texts = [t for t in texts if t and t.strip()]

        # 如果没有文本，返回空字符串
        if not texts:
            return ("",)

        # 应用前缀和后缀
        if prefix or suffix:
            texts = [f"{prefix}{t}{suffix}" for t in texts]

        # 选择分隔符
        separator_map = {
            "newline": "\n",
            "comma": ",",
            "space": " ",
            "comma_space": ", ",
            "semicolon": ";",
            "pipe": "|",
            "custom": custom_separator,
        }

        sep = separator_map.get(separator, "\n")

        # 拼接文本
        result = sep.join(texts)

        print(f"[TextConcatenate] 拼接了 {len(texts)} 段文本，总长度: {len(result)}")

        return (result,)


class ImageBatchMerge:
    """图片批次合并节点 - 将多个图片或图片批次合并成一个批次"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {"tooltip": "第一个图片或图片批次"}),
            },
            "optional": {
                "image2": ("IMAGE", {"default": None, "tooltip": "第二个图片或图片批次"}),
                "image3": ("IMAGE", {"default": None, "tooltip": "第三个图片或图片批次"}),
                "image4": ("IMAGE", {"default": None, "tooltip": "第四个图片或图片批次"}),
                "image5": ("IMAGE", {"default": None, "tooltip": "第五个图片或图片批次"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "merge"
    CATEGORY = "api node/utils"

    def merge(
        self,
        image1,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
    ):
        """合并多个图片批次"""

        # 收集所有非空图片
        images = [image1]
        if image2 is not None:
            images.append(image2)
        if image3 is not None:
            images.append(image3)
        if image4 is not None:
            images.append(image4)
        if image5 is not None:
            images.append(image5)

        # 合并所有图片批次
        merged = torch.cat(images, dim=0)
        count = merged.shape[0]

        print(f"[ImageBatchMerge] 合并了 {len(images)} 个批次，共 {count} 张图片")

        return (merged, count)
