import torch
import aiohttp
import io
import base64
from PIL import Image
import numpy as np

def tensor_to_data_uri(
    image: torch.Tensor,
    mime_type: str = "image/png",
) -> str:
    """Converts a tensor image to a Data URI string.

    Args:
        image: Input torch.Tensor image.
        mime_type: Target image MIME type (e.g., 'image/png', 'image/jpeg', 'image/webp').

    Returns:
        Data URI string (e.g., 'data:image/png;base64,...').
    """
    # 将 ComfyUI 图像转换为 PIL Image
    if isinstance(image, torch.Tensor):
        # 如果是批次图像，取第一张
        if len(image.shape) == 4:
            input_image = image[0]
        else:
            input_image = image
        # 转换为 numpy 数组
        image_array = input_image.cpu().numpy()
        # 转换为 0-255 范围
        image_array = (image_array * 255).astype(np.uint8)
        # 创建 PIL Image
        pil_image = Image.fromarray(image_array)
    else:
        pil_image = image
    
    # 确保图像是 RGB 模式
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    # 将图像转换为base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:{mime_type};base64,{image_base64}"

async def download_url_to_bytesio(url: str, timeout: int = None) -> io.BytesIO:
    """Downloads content from a URL using requests and returns it as BytesIO.

    Args:
        url: The URL to download.
        timeout: Request timeout in seconds. Defaults to None (no timeout).

    Returns:
        BytesIO object containing the downloaded content.
    """
    timeout_cfg = aiohttp.ClientTimeout(total=timeout) if timeout else None
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            return io.BytesIO(await resp.read())

def bytesio_to_image_tensor(image_bytesio: io.BytesIO, mode: str = "RGBA") -> torch.Tensor:
    """Converts image data from BytesIO to a torch.Tensor.

    Args:
        image_bytesio: BytesIO object containing the image data.
        mode: The PIL mode to convert the image to (e.g., "RGB", "RGBA").

    Returns:
        A torch.Tensor representing the image (1, H, W, C).

    Raises:
        PIL.UnidentifiedImageError: If the image data cannot be identified.
        ValueError: If the specified mode is invalid.
    """
    image = Image.open(image_bytesio)
    image = image.convert(mode)
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


def build_api_url(api_base: str, path: str) -> str:
    """智能构建 API URL，支持多种输入格式

    Args:
        api_base: API 基础地址，可以是：
            - 完整 URL: "https://api.example.com/v1/chat/completions"
            - 包含路径的 base: "https://api.example.com/v1"
            - 纯域名: "https://api.example.com"
        path: 要拼接的路径，如 "/v1/chat/completions"

    Returns:
        完整的 API URL

    Examples:
        >>> build_api_url("https://api.example.com", "/v1/chat/completions")
        "https://api.example.com/v1/chat/completions"

        >>> build_api_url("https://api.example.com/v1", "/v1/chat/completions")
        "https://api.example.com/v1/chat/completions"

        >>> build_api_url("https://api.example.com/v1/chat/completions", "/v1/chat/completions")
        "https://api.example.com/v1/chat/completions"
    """
    if not api_base:
        raise ValueError("api_base 不能为空")

    # 标准化 path（确保以 / 开头）
    if path and not path.startswith("/"):
        path = "/" + path

    # 移除 api_base 末尾的斜杠
    api_base = api_base.rstrip("/")

    # 场景 1: api_base 已经是完整的 URL（包含了完整的端点路径）
    # 检查是否已经包含了目标路径
    if path and api_base.endswith(path):
        return api_base

    # 场景 2: api_base 已经包含了 path 的部分路径
    # 例如: api_base="xxx/v1", path="/v1/chat/completions"
    # 提取 path 中 /v1/ 之后的部分
    if "/v1/" in api_base and path.startswith("/v1/"):
        # api_base 已经包含 /v1/，只拼接 /v1/ 后面的部分
        remaining_path = path.split("/v1/", 1)[-1]
        if remaining_path:
            return f"{api_base}/{remaining_path}"
        return api_base

    # 场景 3: 检查是否有其他版本路径重复（如 /v2/, /api/ 等）
    # 提取 path 的第一个有意义的路径段
    path_parts = [p for p in path.split("/") if p]
    if path_parts and api_base.endswith(f"/{path_parts[0]}"):
        # api_base 已经包含了 path 的第一个路径段
        remaining_parts = path_parts[1:]
        if remaining_parts:
            return f"{api_base}/{'/'.join(remaining_parts)}"
        return api_base

    # 场景 4: 标准拼接
    return api_base + path
