import io
import json
import torch
import base64
import requests
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from .utils.apinode import bytesio_to_image_tensor, download_url_to_bytesio, tensor_to_data_uri

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False,
    modalities: Optional[List[str]] = None
    extendParams: Optional[Dict[str, Any]] = None


class ChatCompletionChoice(BaseModel):
    message: Dict[str, Any]


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: Optional[bool] = None
    message: Optional[str] = None
    request_id: Optional[str] = Field(default=None, alias="requestId")
    choices: Optional[List[ChatCompletionChoice]] = None

class COMMON_HTTP_REQUEST:
    """Common API"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (["POST", "GET", "PUT", "DELETE", "PATCH", "HEAD"], {}),
                "params": ("STRING", {"forceInput": True}),
                "api_key": ("STRING", {"default": ""}),
                "api_endpoint": ("STRING", {"default": "/api/v1/common"}),
            },
            "optional": {
                "headers": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 600, "min": 1, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    FUNCTION = "request"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def request(self, method, params, api_key, api_endpoint, headers="", timeout=600):
        try:
            # 验证必填参数
            if not api_key or api_key.strip() == "":
                raise ValueError("API密钥不能为空")
            
            if not api_endpoint or api_endpoint.strip() == "":
                raise ValueError("API端点不能为空")
            
            # 解析params参数
            try:
                if isinstance(params, str):
                    request_body = json.loads(params)
                else:
                    request_body = params
                
                
            except json.JSONDecodeError as e:
                raise ValueError(f"params参数JSON解析失败: {str(e)}")
            
            # 设置请求头
            real_headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": f"Bearer {api_key.strip()}"
            }

            if headers and headers.strip() != "":
                try:
                    extra_headers = json.loads(headers)
                    if isinstance(extra_headers, dict):
                        real_headers.update(extra_headers)
                except json.JSONDecodeError as e:
                    raise ValueError(f"headers参数JSON解析失败: {str(e)}")
            
            print(f"[COMMON_HTTP_REQUEST] 请求参数: {json.dumps(request_body, ensure_ascii=False)}")
            
            # 发送POST请求
            response = requests.request(
                method,
                api_endpoint.strip(),
                headers=real_headers,
                json=request_body,
                timeout=timeout,
                verify=False  # 忽略SSL证书验证
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            response_data = response.json()
            
            # 返回成功结果
            result_json = json.dumps(response_data, ensure_ascii=False, indent=2)
            print(f"[COMMON_HTTP_REQUEST] 请求结果: {result_json}")
            
            return (result_json,)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            print(f"[COMMON_HTTP_REQUEST] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            print(f"[COMMON_HTTP_REQUEST] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"[COMMON_HTTP_REQUEST] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
        
class LLMImageGenerate():
    """
    Generates images synchronously via OpenAI like(LLM) API.

    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True,}),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": "可选参考图，支持多张图片批量输入",
                    },
                ),
                "image_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "可选图片 URL，每行一个，0 个或多个",
                    },
                ),
                "image_mime_type": (
                    "STRING",
                    {
                        "default": "image/png",
                        "tooltip": "上传到接口的图片编码格式",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "gemini-3-pro-image-preview",
                        "tooltip": "可切换的模型名称",
                    },
                ),
                "api_base": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "API 基础地址，例如 https://host/api/openai",
                    },
                ),
                "auth_token": ("STRING", {"default": "", "tooltip": "Bearer Token"}),
                "headers": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 600, "min": 1, "max": 3600}),
                "extendParams": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "request", "response")
    FUNCTION = "api_call"
    CATEGORY = "api node/image/llm"
    API_NODE = True

    async def api_call(
        self,
        prompt,
        image=None,
        image_urls="",
        image_mime_type="image/png",
        model="gemini-3-pro-image-preview",
        api_base="",
        auth_token="",
        headers="",
        timeout=600,
        extendParams="",
        **kwargs,
    ):
        content_blocks: List[Dict[str, Any]] = []
        if prompt and prompt.strip():
            content_blocks.append({"type": "text", "text": prompt})

        if image is not None:
            if len(image.shape) < 4 or image.shape[0] == 0:
                raise ValueError("输入图片格式不正确")
            for idx in range(image.shape[0]):
                data_uri = tensor_to_data_uri(image[idx], mime_type=image_mime_type)
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )

        if image_urls:
            urls = [
                line.strip()
                for line in image_urls.splitlines()
                if line.strip()
            ]
            for url in urls:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )

        if not content_blocks:
            raise ValueError("请至少提供文本或图片中的一种输入")

        path = "/v1/chat/completions"
        request = ChatCompletionRequest(
            model=model or "gemini-3-pro-image-preview",
            messages=[
                {
                    "role": "user",
                    "content": content_blocks,
                }
            ],
            stream=False,
            modalities=[
                "TEXT",
                "IMAGE"
            ],
            extendParams=json.loads(extendParams) if extendParams and extendParams.strip() else None,
        )
        auth_kwargs = dict(kwargs) if kwargs else {}
        if auth_token:
            auth_kwargs["auth_token"] = auth_token

        url = api_base.rstrip("/") + path
        _headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        if (headers and headers.strip() != ""):
            try:
                extra_headers = json.loads(headers)
                if isinstance(extra_headers, dict):
                    _headers.update(extra_headers)
            except json.JSONDecodeError as e:
                raise ValueError(f"headers参数JSON解析失败: {str(e)}")
        if "auth_token" in auth_kwargs and auth_kwargs["auth_token"]:
            _headers["Authorization"] = f"Bearer {auth_kwargs['auth_token']}"
        request_json = request.model_dump_json()
        try:
            response_http = requests.post(
                url,
                headers=_headers,
                json=json.loads(request.model_dump_json()),
                timeout=timeout,
                verify=False,
            )
            response_http.raise_for_status()
            response_json = response_http.json()
            response = ChatCompletionResponse.model_validate(response_json)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"网络请求失败: {str(e)}") from e

        if response.success is False:
            raise ValueError(response.message or "接口返回失败")

        if not response.choices:
            raise ValueError("接口未返回可用的结果")

        message = response.choices[0].message or {}
        content = message.get("content")
        image_url = None

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url")
                    if image_url:
                        break
        elif isinstance(content, dict):
            if content.get("type") == "image_url":
                image_url = content.get("image_url", {}).get("url")
        elif isinstance(content, str):
            image_url = content

        if not image_url:
            raise ValueError("接口未返回图片内容")

        if image_url.startswith("http://") or image_url.startswith("https://"):
            img_bytesio = await download_url_to_bytesio(image_url)
        else:
            if image_url.startswith("data:"):
                _, _, encoded = image_url.partition(",")
            else:
                encoded = image_url
            try:
                img_bytes = base64.b64decode(encoded)
            except Exception as exc:
                raise ValueError("返回的图片内容不是有效的 Base64 数据") from exc
            img_bytesio = io.BytesIO(img_bytes)

        img_tensor = bytesio_to_image_tensor(img_bytesio)
        return (img_tensor, request_json, json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


class LLMResponseImageParser:
    """
    解析 LLM API 响应中的图片数据，支持多种图片格式。
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "response": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "LLM API 返回的 JSON 响应字符串",
                    },
                ),
            },
            "optional": {
                "image_path": (
                    "STRING",
                    {
                        "default": "choices.0.message.content",
                        "tooltip": "图片在响应中的路径，用点号分隔，例如：choices.0.message.content",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 60,
                        "min": 1,
                        "max": 600,
                        "tooltip": "下载图片的超时时间（秒）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "parse_image"
    CATEGORY = "api node/image/llm"

    async def parse_image(self, response, image_path="choices.0.message.content", timeout=60):
        """解析 LLM 响应中的图片（支持多张）"""
        try:
            # 解析 JSON 响应
            if isinstance(response, str):
                response_data = json.loads(response)
            else:
                response_data = response
        except json.JSONDecodeError as e:
            raise ValueError(f"无效的 JSON 响应: {str(e)}")

        # 根据路径提取图片内容（可能有多张）
        image_urls = self._extract_image_from_path(response_data, image_path)

        if not image_urls:
            raise ValueError(f"无法从响应中提取图片，路径：{image_path}")

        # 确保 image_urls 是列表
        if not isinstance(image_urls, list):
            image_urls = [image_urls]

        # 过滤掉空值
        image_urls = [url for url in image_urls if url]

        if not image_urls:
            raise ValueError("未找到有效的图片数据")

        # 下载或解码所有图片
        img_tensors = []
        for url in image_urls:
            img_bytesio = await self._get_image_bytesio(url, timeout)
            img_tensor = bytesio_to_image_tensor(img_bytesio)
            img_tensors.append(img_tensor)

        # 合并所有图片 tensor（沿批次维度）
        combined_tensor = torch.cat(img_tensors, dim=0)

        # 返回图片 URLs 的 JSON 字符串
        urls_json = json.dumps(image_urls, ensure_ascii=False)

        return (combined_tensor, urls_json)

    def _extract_image_from_path(self, data, path):
        """从嵌套的字典/列表中提取图片 URL"""
        parts = path.split(".")
        current = data

        for part in parts:
            if not current:
                return None

            # 处理数组索引
            if part.isdigit():
                index = int(part)
                if isinstance(current, list) and 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            # 处理字典键
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        # 从 current 中提取图片 URL
        return self._extract_image_url(current)

    def _extract_image_url(self, content):
        """从内容中提取图片 URL（支持多张）"""
        if not content:
            return []

        urls = []

        # 如果是字符串，直接返回
        if isinstance(content, str):
            return [content]

        # 如果是列表，查找所有图片类型的项
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {})
                        if isinstance(url, dict):
                            extracted = url.get("url")
                            if extracted:
                                urls.append(extracted)
                        elif isinstance(url, str):
                            urls.append(url)
                    # 尝试直接获取 url 字段
                    elif "url" in item:
                        urls.append(item["url"])

        # 如果是字典，直接查找图片
        elif isinstance(content, dict):
            if content.get("type") == "image_url":
                url = content.get("image_url", {})
                if isinstance(url, dict):
                    extracted = url.get("url")
                    if extracted:
                        urls.append(extracted)
                elif isinstance(url, str):
                    urls.append(url)
            # 尝试直接获取 url 字段
            elif "url" in content:
                urls.append(content["url"])

        return urls if urls else None

    async def _get_image_bytesio(self, image_url, timeout):
        """获取图片的 BytesIO 对象"""
        # 处理 HTTP/HTTPS URL
        if image_url.startswith("http://") or image_url.startswith("https://"):
            return await download_url_to_bytesio(image_url, timeout)

        # 处理 Base64 编码的图片
        if image_url.startswith("data:"):
            _, _, encoded = image_url.partition(",")
        else:
            encoded = image_url

        try:
            img_bytes = base64.b64decode(encoded)
        except Exception as exc:
            raise ValueError("无效的 Base64 图片数据") from exc

        return io.BytesIO(img_bytes)


NODE_CLASS_MAPPINGS = {
    "COMMON_HTTP_REQUEST": COMMON_HTTP_REQUEST,
    "LLMImageGenerate": LLMImageGenerate,
    "LLMResponseImageParser": LLMResponseImageParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "COMMON_HTTP_REQUEST": "通用HTTP请求",
    "LLMImageGenerate": "LLM 图像生成",
    "LLMResponseImageParser": "LLM响应图片解析",
}