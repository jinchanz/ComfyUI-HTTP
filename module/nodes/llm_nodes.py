"""
LLM 相关节点 - 图片生成、智能生成和响应解析
"""
import io
import json
import base64
import torch
import requests
from typing import Any, Dict, List

from ..models import ChatCompletionRequest, ChatCompletionResponse
from ..utils.apinode import bytesio_to_image_tensor, download_url_to_bytesio, tensor_to_data_uri

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
                "text",
                "image"
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

        # 收集所有 choices 中的图片 URL/base64，支持多张
        image_urls_collected: List[str] = []
        for choice in response.choices:
            msg = choice.message or {}
            urls = self._extract_image_urls(msg.get("content"))
            if urls:
                image_urls_collected.extend(urls)

        # 去掉空值
        image_urls_collected = [u for u in image_urls_collected if u]
        if not image_urls_collected:
            raise ValueError("接口未返回图片内容")

        # 逐张下载/解码并拼成批次
        img_tensors = []
        for url in image_urls_collected:
            if url.startswith("http://") or url.startswith("https://"):
                img_bytesio = await download_url_to_bytesio(url)
            else:
                if url.startswith("data:"):
                    _, _, encoded = url.partition(",")
                else:
                    encoded = url
                try:
                    img_bytes = base64.b64decode(encoded)
                except Exception as exc:
                    raise ValueError("返回的图片内容不是有效的 Base64 数据") from exc
                img_bytesio = io.BytesIO(img_bytes)
            img_tensor = bytesio_to_image_tensor(img_bytesio)
            img_tensors.append(img_tensor)

        batch_tensor = torch.cat(img_tensors, dim=0)
        return (batch_tensor, request_json, json.dumps(response.model_dump(), ensure_ascii=False, indent=2))

    def _extract_image_urls(self, content):
        urls = []
        if content is None:
            return urls

        if isinstance(content, str):
            urls.append(content)
        elif isinstance(content, dict):
            if content.get("type") == "image_url":
                url_val = content.get("image_url", {})
                if isinstance(url_val, dict):
                    maybe = url_val.get("url")
                    if maybe:
                        urls.append(maybe)
                elif isinstance(url_val, str):
                    urls.append(url_val)
            elif "url" in content:
                urls.append(content.get("url"))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "image_url":
                        url_val = item.get("image_url", {})
                        if isinstance(url_val, dict):
                            maybe = url_val.get("url")
                            if maybe:
                                urls.append(maybe)
                        elif isinstance(url_val, str):
                            urls.append(url_val)
                    elif "url" in item:
                        urls.append(item.get("url"))
        return urls


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


class LLMSmartGenerate():
    """
    智能 LLM 生成节点 - 同时支持文本和图片生成

    该节点会自动检测 API 返回的内容类型：
    - 如果返回图片，则输出图片和文本（如果有）
    - 如果只返回文本，则只输出文本
    - 支持混合输出（文本 + 图片）
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
                "video": (
                    "VIDEO",
                    {
                        "default": None,
                        "tooltip": "可选视频输入",
                    },
                ),
                "video_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "可选视频 URL，每行一个，0 个或多个",
                    },
                ),
                "audio": (
                    "AUDIO",
                    {
                        "default": None,
                        "tooltip": "可选音频输入",
                    },
                ),
                "audio_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "可选音频 URL，每行一个，0 个或多个",
                    },
                ),
                "image_mime_type": (
                    "STRING",
                    {
                        "default": "image/png",
                        "tooltip": "上传到接口的图片编码格式",
                    },
                ),
                "video_mime_type": (
                    "STRING",
                    {
                        "default": "video/mp4",
                        "tooltip": "上传到接口的视频编码格式",
                    },
                ),
                "audio_mime_type": (
                    "STRING",
                    {
                        "default": "audio/wav",
                        "tooltip": "上传到接口的音频编码格式",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "gemini-3-pro-preview",
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
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 128000, "tooltip": "最大生成 token 数"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "控制生成随机性，0=确定性，2=最随机"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "核采样参数"}),
                "stream": ("BOOLEAN", {"default": False, "tooltip": "是否启用流式输出（实验性）"}),
                "extendParams": ("STRING", {"default": "", "tooltip": "额外的 JSON 参数"}),
                "output_mode": (
                    ["auto", "text_only", "image_only", "video_only", "audio_only", "multimodal"],
                    {
                        "default": "auto",
                        "tooltip": "输出模式：auto（自动检测），text_only（仅文本），image_only（仅图片），video_only（仅视频），audio_only（仅音频），multimodal（全部）"
                    }
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "VIDEO", "AUDIO", "BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("text", "image", "video", "audio", "has_image", "has_video", "has_audio", "request", "response")
    FUNCTION = "api_call"
    CATEGORY = "api node/llm/smart"
    API_NODE = True

    async def api_call(
        self,
        prompt,
        image=None,
        image_urls="",
        video=None,
        video_urls="",
        audio=None,
        audio_urls="",
        image_mime_type="image/png",
        video_mime_type="video/mp4",
        audio_mime_type="audio/wav",
        model="gemini-3-pro-preview",
        api_base="",
        auth_token="",
        headers="",
        timeout=600,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        stream=False,
        extendParams="",
        output_mode="auto",
        **kwargs,
    ):
        # 构建请求内容
        content_blocks: List[Dict[str, Any]] = []
        if prompt and prompt.strip():
            content_blocks.append({"type": "text", "text": prompt})

        # 处理图片输入
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

        # 处理视频输入
        if video is not None:
            try:
                # 视频可能是 dict 格式: {"video": tensor, "audio": tensor, ...}
                # 或者直接是 URL/路径字符串
                if isinstance(video, dict):
                    video_path = video.get("video") or video.get("path")
                elif isinstance(video, str):
                    video_path = video
                else:
                    video_path = None

                if video_path:
                    content_blocks.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": video_path},
                        }
                    )
            except Exception as e:
                print(f"[LLMSmartGenerate] 视频输入处理警告: {str(e)}")

        if video_urls:
            urls = [
                line.strip()
                for line in video_urls.splitlines()
                if line.strip()
            ]
            for url in urls:
                content_blocks.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": url},
                    }
                )

        # 处理音频输入
        if audio is not None:
            try:
                # 音频可能是 dict 格式: {"waveform": tensor, "sample_rate": int}
                # 或者直接是 URL/路径字符串
                if isinstance(audio, dict):
                    audio_path = audio.get("audio") or audio.get("path")
                elif isinstance(audio, str):
                    audio_path = audio
                else:
                    audio_path = None

                if audio_path:
                    content_blocks.append(
                        {
                            "type": "audio_url",
                            "audio_url": {"url": audio_path},
                        }
                    )
            except Exception as e:
                print(f"[LLMSmartGenerate] 音频输入处理警告: {str(e)}")

        if audio_urls:
            urls = [
                line.strip()
                for line in audio_urls.splitlines()
                if line.strip()
            ]
            for url in urls:
                content_blocks.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": url},
                    }
                )

        if not content_blocks:
            raise ValueError("请至少提供文本、图片、视频或音频中的一种输入")

        # 构建请求
        path = "/v1/chat/completions"

        # 根据输出模式设置 modalities（使用小写，符合 OpenAI API 规范）
        if output_mode == "text_only":
            modalities = ["text"]
        elif output_mode == "image_only":
            modalities = ["image"]
        elif output_mode == "video_only":
            modalities = ["video"]
        elif output_mode == "audio_only":
            modalities = ["audio"]
        elif output_mode == "multimodal":
            modalities = ["text", "image", "video", "audio"]
        else:  # auto - 使用保守设置，避免 API 不支持某些模态
            modalities = ["text", "image"]

        # 构建基础请求参数
        request_params = {
            "model": model or "gemini-3-pro-preview",
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks,
                }
            ],
            "stream": stream,
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # 合并 extendParams（可以覆盖 modalities）
        if extendParams and extendParams.strip():
            try:
                extra_params = json.loads(extendParams)
                if isinstance(extra_params, dict):
                    # extendParams 可以覆盖 modalities
                    request_params.update(extra_params)
            except json.JSONDecodeError as e:
                raise ValueError(f"extendParams JSON 解析失败: {str(e)}")

        request = ChatCompletionRequest(**request_params)

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
            print(f"[LLMSmartGenerate] 请求结果: {response_http.status_code}")
            print(f"[LLMSmartGenerate] 响应内容: {response_http.content[:500]}")  # 打印前500字符以查看响应内容
            response_http.raise_for_status()
            response_json = response_http.json()
            response = ChatCompletionResponse.model_validate(response_json)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"网络请求失败: {str(e)}") from e

        if response.success is False:
            raise ValueError(response.message or "接口返回失败")

        if not response.choices:
            raise ValueError("接口未返回可用的结果")

        # 智能解析响应内容
        text_content = ""
        image_urls_collected: List[str] = []
        video_urls_collected: List[str] = []
        audio_urls_collected: List[str] = []

        for choice in response.choices:
            msg = choice.message or {}
            content = msg.get("content")

            # 提取文本内容
            text_parts = self._extract_text_content(content)
            if text_parts:
                text_content += "\n".join(text_parts)

            # 提取图片 URL
            if output_mode not in ["text_only", "video_only", "audio_only"]:
                urls = self._extract_image_urls(content)
                if urls:
                    image_urls_collected.extend(urls)

            # 提取视频 URL
            if output_mode not in ["text_only", "image_only", "audio_only"]:
                urls = self._extract_video_urls(content)
                if urls:
                    video_urls_collected.extend(urls)

            # 提取音频 URL
            if output_mode not in ["text_only", "image_only", "video_only"]:
                urls = self._extract_audio_urls(content)
                if urls:
                    audio_urls_collected.extend(urls)

        # 处理图片输出
        has_image = False
        batch_tensor = None

        # 去掉空值
        image_urls_collected = [u for u in image_urls_collected if u]

        if image_urls_collected:
            # 逐张下载/解码并拼成批次
            img_tensors = []
            for url in image_urls_collected:
                try:
                    if url.startswith("http://") or url.startswith("https://"):
                        img_bytesio = await download_url_to_bytesio(url)
                    else:
                        if url.startswith("data:"):
                            _, _, encoded = url.partition(",")
                        else:
                            encoded = url
                        try:
                            img_bytes = base64.b64decode(encoded)
                        except Exception as exc:
                            print(f"[LLMSmartGenerate] 图片解码失败: {str(exc)}")
                            continue
                        img_bytesio = io.BytesIO(img_bytes)
                    img_tensor = bytesio_to_image_tensor(img_bytesio)
                    img_tensors.append(img_tensor)
                except Exception as exc:
                    print(f"[LLMSmartGenerate] 图片处理失败: {str(exc)}")
                    continue

            if img_tensors:
                batch_tensor = torch.cat(img_tensors, dim=0)
                has_image = True

        # 如果没有图片，创建一个空白占位图片（1x1 黑色像素）
        if batch_tensor is None:
            batch_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        # 处理视频输出
        has_video = False
        video_output = None
        video_urls_collected = [u for u in video_urls_collected if u]

        if video_urls_collected:
            # 返回视频 URL 字符串列表作为 dict
            video_output = {
                "type": "video_urls",
                "urls": video_urls_collected,
            }
            has_video = True
            print(f"[LLMSmartGenerate] 检测到 {len(video_urls_collected)} 个视频 URL")

        # 如果没有视频，返回空 dict
        if video_output is None:
            video_output = {"type": "none", "urls": []}

        # 处理音频输出
        has_audio = False
        audio_output = None
        audio_urls_collected = [u for u in audio_urls_collected if u]

        if audio_urls_collected:
            # 返回音频 URL 字符串列表作为 dict
            audio_output = {
                "type": "audio_urls",
                "urls": audio_urls_collected,
            }
            has_audio = True
            print(f"[LLMSmartGenerate] 检测到 {len(audio_urls_collected)} 个音频 URL")

        # 如果没有音频，返回空 dict
        if audio_output is None:
            audio_output = {"type": "none", "urls": []}

        # 确保返回文本内容
        if not text_content:
            text_content = ""

        response_json_str = json.dumps(response.model_dump(), ensure_ascii=False, indent=2)

        print(f"[LLMSmartGenerate] 文本: {len(text_content)} 字符, 图片: {has_image}, 视频: {has_video}, 音频: {has_audio}")

        return (text_content, batch_tensor, video_output, audio_output, has_image, has_video, has_audio, request_json, response_json_str)

    def _extract_text_content(self, content):
        """从响应中提取文本内容"""
        text_parts = []

        if content is None:
            return text_parts

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, dict):
            if content.get("type") == "text":
                text_val = content.get("text")
                if text_val:
                    text_parts.append(text_val)
            elif "text" in content:
                text_parts.append(content.get("text"))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        text_val = item.get("text")
                        if text_val:
                            text_parts.append(text_val)
                    elif "text" in item:
                        text_parts.append(item.get("text"))

        return text_parts

    def _extract_image_urls(self, content):
        """从响应中提取图片 URL"""
        urls = []
        if content is None:
            return urls

        if isinstance(content, str):
            # 只有当字符串看起来像 URL 或 base64 数据时才认为是图片
            if (content.startswith("http://") or content.startswith("https://") or
                content.startswith("data:image") or len(content) > 100):
                urls.append(content)
        elif isinstance(content, dict):
            if content.get("type") == "image_url":
                url_val = content.get("image_url", {})
                if isinstance(url_val, dict):
                    maybe = url_val.get("url")
                    if maybe:
                        urls.append(maybe)
                elif isinstance(url_val, str):
                    urls.append(url_val)
            elif "url" in content:
                urls.append(content.get("url"))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    if (item.startswith("http://") or item.startswith("https://") or
                        item.startswith("data:image") or len(item) > 100):
                        urls.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "image_url":
                        url_val = item.get("image_url", {})
                        if isinstance(url_val, dict):
                            maybe = url_val.get("url")
                            if maybe:
                                urls.append(maybe)
                        elif isinstance(url_val, str):
                            urls.append(url_val)
                    elif "url" in item:
                        urls.append(item.get("url"))
        return urls

    def _extract_video_urls(self, content):
        """从响应中提取视频 URL"""
        urls = []
        if content is None:
            return urls

        if isinstance(content, str):
            # 检查是否是视频 URL 或 data URI
            if (content.startswith("http://") or content.startswith("https://") or
                content.startswith("data:video")):
                urls.append(content)
        elif isinstance(content, dict):
            if content.get("type") == "video_url":
                url_val = content.get("video_url", {})
                if isinstance(url_val, dict):
                    maybe = url_val.get("url")
                    if maybe:
                        urls.append(maybe)
                elif isinstance(url_val, str):
                    urls.append(url_val)
            elif content.get("type") == "video":
                if "url" in content:
                    urls.append(content.get("url"))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    if (item.startswith("http://") or item.startswith("https://") or
                        item.startswith("data:video")):
                        urls.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "video_url":
                        url_val = item.get("video_url", {})
                        if isinstance(url_val, dict):
                            maybe = url_val.get("url")
                            if maybe:
                                urls.append(maybe)
                        elif isinstance(url_val, str):
                            urls.append(url_val)
                    elif item.get("type") == "video" and "url" in item:
                        urls.append(item.get("url"))
        return urls

    def _extract_audio_urls(self, content):
        """从响应中提取音频 URL"""
        urls = []
        if content is None:
            return urls

        if isinstance(content, str):
            # 检查是否是音频 URL 或 data URI
            if (content.startswith("http://") or content.startswith("https://") or
                content.startswith("data:audio")):
                urls.append(content)
        elif isinstance(content, dict):
            if content.get("type") == "audio_url":
                url_val = content.get("audio_url", {})
                if isinstance(url_val, dict):
                    maybe = url_val.get("url")
                    if maybe:
                        urls.append(maybe)
                elif isinstance(url_val, str):
                    urls.append(url_val)
            elif content.get("type") == "audio":
                if "url" in content:
                    urls.append(content.get("url"))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    if (item.startswith("http://") or item.startswith("https://") or
                        item.startswith("data:audio")):
                        urls.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "audio_url":
                        url_val = item.get("audio_url", {})
                        if isinstance(url_val, dict):
                            maybe = url_val.get("url")
                            if maybe:
                                urls.append(maybe)
                        elif isinstance(url_val, str):
                            urls.append(url_val)
                    elif item.get("type") == "audio" and "url" in item:
                        urls.append(item.get("url"))
        return urls

