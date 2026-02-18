"""
HTTP 请求相关节点
"""
import json
import time
import requests


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


class POLLING_HTTP_REQUEST:
    """通用轮询节点 - 周期性执行HTTP请求直到满足条件"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (["POST", "GET", "PUT", "DELETE", "PATCH"], {}),
                "api_endpoint": ("STRING", {"default": ""}),
                "params": ("STRING", {"default": "{}", "multiline": True}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 60, "tooltip": "轮询间隔（秒）"}),
                "max_attempts": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "最大轮询次数"}),
                "success_condition": (["status_field", "custom_jsonpath", "status_code_only"], {
                    "tooltip": "成功条件类型"
                }),
                "condition_field": ("STRING", {"default": "status", "tooltip": "状态字段路径，如: data.status 或 result.state"}),
                "expected_value": ("STRING", {"default": "completed", "tooltip": "期望的状态值"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "headers": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 30, "min": 1, "max": 600}),
                "stop_on_error": ("BOOLEAN", {"default": True, "tooltip": "遇到错误时是否立即停止"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("response", "attempts", "success", "final_status")

    FUNCTION = "poll_request"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def poll_request(self, method, api_endpoint, params, poll_interval, max_attempts,
                     success_condition, condition_field, expected_value,
                     api_key="", headers="", timeout=30, stop_on_error=True):
        """
        执行轮询请求
        """
        try:
            # 验证参数
            if not api_endpoint or api_endpoint.strip() == "":
                raise ValueError("API端点不能为空")

            # 解析params
            try:
                if isinstance(params, str):
                    request_body = json.loads(params) if params.strip() else {}
                else:
                    request_body = params
            except json.JSONDecodeError as e:
                raise ValueError(f"params参数JSON解析失败: {str(e)}")

            # 设置请求头
            real_headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
            }

            if api_key and api_key.strip():
                real_headers["Authorization"] = f"Bearer {api_key.strip()}"

            if headers and headers.strip():
                try:
                    extra_headers = json.loads(headers)
                    if isinstance(extra_headers, dict):
                        real_headers.update(extra_headers)
                except json.JSONDecodeError as e:
                    raise ValueError(f"headers参数JSON解析失败: {str(e)}")

            print(f"[POLLING_HTTP_REQUEST] 开始轮询: {api_endpoint}")
            print(f"[POLLING_HTTP_REQUEST] 最大尝试次数: {max_attempts}, 间隔: {poll_interval}秒")

            attempts = 0
            last_response = None
            last_status = "未开始"
            success = False

            # 开始轮询
            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                print(f"[POLLING_HTTP_REQUEST] 第 {attempt}/{max_attempts} 次轮询...")

                try:
                    # 发送请求
                    response = requests.request(
                        method,
                        api_endpoint.strip(),
                        headers=real_headers,
                        json=request_body if method in ["POST", "PUT", "PATCH"] else None,
                        params=request_body if method in ["GET", "DELETE"] else None,
                        timeout=timeout,
                        verify=False
                    )
                    print(f"[POLLING_HTTP_REQUEST] 响应状态码: {response}")
                    # 检查HTTP状态码
                    if success_condition == "status_code_only":
                        if response.status_code == 200:
                            success = True
                            last_status = f"HTTP {response.status_code}"
                            last_response = response.text
                            print(f"[POLLING_HTTP_REQUEST] ✓ 成功 (HTTP 200)")
                            break
                        else:
                            last_status = f"HTTP {response.status_code}"
                            last_response = response.text
                    else:
                        # 先检查HTTP状态码，不成功的话记录
                        if response.status_code < 200 or response.status_code >= 300:
                            last_status = f"HTTP {response.status_code}"
                            last_response = response.text
                            print(f"[POLLING_HTTP_REQUEST] HTTP {response.status_code}: {response.text[:100]}")
                        else:
                            # 解析响应
                            try:
                                response_data = response.json()
                                last_response = json.dumps(response_data, ensure_ascii=False, indent=2)
                            except json.JSONDecodeError:
                                # 响应不是JSON，直接使用文本
                                last_response = response.text
                                response_data = {"raw_response": response.text}

                        # 检查成功条件（仅在HTTP状态码正常时）
                        if response.status_code >= 200 and response.status_code < 300:
                            if success_condition == "status_field":
                                # 通过字段路径获取值
                                current_value = self._get_nested_value(response_data, condition_field)
                                last_status = str(current_value) if current_value is not None else "无法获取状态"

                                if current_value == expected_value or str(current_value) == expected_value:
                                    success = True
                                    print(f"[POLLING_HTTP_REQUEST] ✓ 成功: {condition_field}={current_value}")
                                    break
                                else:
                                    print(f"[POLLING_HTTP_REQUEST] 当前状态: {condition_field}={current_value}, 期望: {expected_value}")

                            elif success_condition == "custom_jsonpath":
                                # 自定义判断逻辑
                                current_value = self._get_nested_value(response_data, condition_field)
                                last_status = str(current_value) if current_value is not None else "无法获取状态"

                                # 支持多种判断方式
                                if self._check_condition(current_value, expected_value):
                                    success = True
                                    print(f"[POLLING_HTTP_REQUEST] ✓ 满足条件")
                                    break

                        # 检查成功条件
                        if success_condition == "status_field":
                            # 通过字段路径获取值
                            current_value = self._get_nested_value(response_data, condition_field)
                            last_status = str(current_value) if current_value is not None else "无法获取状态"

                            if current_value == expected_value or str(current_value) == expected_value:
                                success = True
                                print(f"[POLLING_HTTP_REQUEST] ✓ 成功: {condition_field}={current_value}")
                                break
                            else:
                                print(f"[POLLING_HTTP_REQUEST] 当前状态: {condition_field}={current_value}, 期望: {expected_value}")

                        elif success_condition == "custom_jsonpath":
                            # 自定义判断逻辑 (简化版)
                            current_value = self._get_nested_value(response_data, condition_field)
                            last_status = str(current_value) if current_value is not None else "无法获取状态"

                            # 支持多种判断方式
                            if self._check_condition(current_value, expected_value):
                                success = True
                                print(f"[POLLING_HTTP_REQUEST] ✓ 满足条件")
                                break

                except requests.exceptions.RequestException as e:
                    error_msg = f"请求失败: {str(e)}"
                    last_status = error_msg
                    print(f"[POLLING_HTTP_REQUEST] ✗ {error_msg}")

                    if stop_on_error:
                        last_response = json.dumps({"error": error_msg}, ensure_ascii=False)
                        break

                except json.JSONDecodeError as e:
                    error_msg = f"JSON解析失败: {str(e)}"
                    last_status = error_msg
                    print(f"[POLLING_HTTP_REQUEST] ✗ {error_msg}")

                    if stop_on_error:
                        last_response = json.dumps({"error": error_msg}, ensure_ascii=False)
                        break

                # 如果不是最后一次尝试，等待后继续
                if attempt < max_attempts and not success:
                    print(f"[POLLING_HTTP_REQUEST] 等待 {poll_interval} 秒后重试...")
                    time.sleep(poll_interval)

            # 输出最终结果
            if success:
                print(f"[POLLING_HTTP_REQUEST] ✓ 轮询成功，共尝试 {attempts} 次")
            else:
                print(f"[POLLING_HTTP_REQUEST] ✗ 轮询失败，已达最大尝试次数 {attempts}")

            final_response = last_response if last_response else json.dumps({"error": "无响应数据"}, ensure_ascii=False)

            return (final_response, attempts, success, last_status)

        except Exception as e:
            error_msg = f"轮询错误: {str(e)}"
            print(f"[POLLING_HTTP_REQUEST] {error_msg}")
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                0,
                False,
                error_msg
            )

    def _get_nested_value(self, data, path):
        """
        获取嵌套字段的值
        支持: data.status, result.state, items[0].value 等
        """
        if not path:
            return data

        keys = path.replace('[', '.').replace(']', '').split('.')
        current = data

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list):
                try:
                    index = int(key)
                    current = current[index] if 0 <= index < len(current) else None
                except (ValueError, IndexError):
                    return None
            else:
                return None

            if current is None:
                return None

        return current

    def _check_condition(self, current_value, expected_value):
        """
        检查条件是否满足
        支持: 相等判断、包含判断、正则等
        """
        if current_value == expected_value or str(current_value) == expected_value:
            return True

        # 支持逗号分隔的多个可能值
        if ',' in expected_value:
            possible_values = [v.strip() for v in expected_value.split(',')]
            return str(current_value) in possible_values

        # 支持简单的包含判断 (contains:xxx)
        if expected_value.startswith("contains:"):
            search_text = expected_value[9:]
            return search_text in str(current_value)

        # 支持否定判断 (not:xxx)
        if expected_value.startswith("not:"):
            not_value = expected_value[4:]
            return str(current_value) != not_value

        return False
