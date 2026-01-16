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
