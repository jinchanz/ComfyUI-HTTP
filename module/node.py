import requests
import json

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
        

NODE_CLASS_MAPPINGS = {
    "COMMON_HTTP_REQUEST": COMMON_HTTP_REQUEST,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "COMMON_HTTP_REQUEST": "通用HTTP请求",
}