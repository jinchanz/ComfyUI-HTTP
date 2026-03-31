"""
HTTP 请求相关节点
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def _parse_json_input(value, field_name, default=None):
    if value is None:
        return {} if default is None else default

    if isinstance(value, str):
        if not value.strip():
            return {} if default is None else default
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"{field_name}参数JSON解析失败: {str(e)}") from e

    return value


def _build_headers(api_key="", headers=""):
    real_headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
    }

    if api_key and api_key.strip():
        real_headers["Authorization"] = f"Bearer {api_key.strip()}"

    if headers and isinstance(headers, str) and headers.strip():
        extra_headers = _parse_json_input(headers, "headers", default={})
        if isinstance(extra_headers, dict):
            real_headers.update(extra_headers)

    return real_headers


def _format_response_text(response):
    try:
        return json.dumps(response.json(), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return response.text


def _get_nested_value(data, path):
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


def _check_condition(current_value, expected_value):
    """
    检查条件是否满足
    支持: 相等判断、包含判断、逗号分隔多值、否定判断
    """
    if current_value == expected_value or str(current_value) == expected_value:
        return True

    if ',' in expected_value:
        possible_values = [v.strip() for v in expected_value.split(',')]
        return str(current_value) in possible_values

    if expected_value.startswith("contains:"):
        search_text = expected_value[9:]
        return search_text in str(current_value)

    if expected_value.startswith("not:"):
        not_value = expected_value[4:]
        return str(current_value) != not_value

    return False


def _inject_task_id(template_value, task_id, placeholder="{task_id}"):
    if template_value is None:
        return None

    replacement = str(task_id)

    if isinstance(template_value, str):
        if placeholder and placeholder in template_value:
            return template_value.replace(placeholder, replacement)
        if template_value.endswith("="):
            return f"{template_value}{replacement}"
        return template_value
    if isinstance(template_value, list):
        return [_inject_task_id(item, task_id, placeholder) for item in template_value]
    if isinstance(template_value, dict):
        return {
            key: _inject_task_id(value, task_id, placeholder)
            for key, value in template_value.items()
        }

    return template_value


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
                                current_value = _get_nested_value(response_data, condition_field)
                                last_status = str(current_value) if current_value is not None else "无法获取状态"

                                if current_value == expected_value or str(current_value) == expected_value:
                                    success = True
                                    print(f"[POLLING_HTTP_REQUEST] ✓ 成功: {condition_field}={current_value}")
                                    break
                                else:
                                    print(f"[POLLING_HTTP_REQUEST] 当前状态: {condition_field}={current_value}, 期望: {expected_value}")

                            elif success_condition == "custom_jsonpath":
                                current_value = _get_nested_value(response_data, condition_field)
                                last_status = str(current_value) if current_value is not None else "无法获取状态"

                                if _check_condition(current_value, expected_value):
                                    success = True
                                    print(f"[POLLING_HTTP_REQUEST] ✓ 满足条件")
                                    break

                        # 检查成功条件
                        if success_condition == "status_field":
                            current_value = _get_nested_value(response_data, condition_field)
                            last_status = str(current_value) if current_value is not None else "无法获取状态"

                            if current_value == expected_value or str(current_value) == expected_value:
                                success = True
                                print(f"[POLLING_HTTP_REQUEST] ✓ 成功: {condition_field}={current_value}")
                                break
                            else:
                                print(f"[POLLING_HTTP_REQUEST] 当前状态: {condition_field}={current_value}, 期望: {expected_value}")

                        elif success_condition == "custom_jsonpath":
                            current_value = _get_nested_value(response_data, condition_field)
                            last_status = str(current_value) if current_value is not None else "无法获取状态"

                            if _check_condition(current_value, expected_value):
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



class CONCURRENT_HTTP_REQUEST:
    """
    并发HTTP请求编排节点 - 支持每个请求独立配置URL/method/参数，支持异步任务自动轮询。

    requests_json 格式示例:
    [
      {
        "url": "https://api.example.com/generate",
        "method": "POST",
        "params": {"prompt": "画一只猫"},
        "headers": {"X-Custom": "value1"}
      },
      {
        "url": "https://api.other.com/query",
        "method": "GET",
        "params": {"id": 123}
      },
      {
        "url": "https://api.example.com/async-task",
        "method": "POST",
        "params": {"prompt": "生成视频"},
        "async": true,
        "task_id_field": "data.task_id",
        "poll_url": "https://api.example.com/task-status?task_id={task_id}",
        "poll_method": "GET",
        "poll_params": {},
        "poll_interval": 3,
        "max_poll_attempts": 30,
        "condition_field": "data.status",
        "expected_value": "completed"
      }
    ]
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "requests_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": (
                        "JSON数组，每个元素描述一个请求。"
                        "基础字段: url(必填), method(默认POST), params(默认{}), headers(可选)。"
                        "异步轮询字段: async=true, task_id_field, poll_url({task_id}占位), "
                        "poll_method, poll_params, poll_interval, max_poll_attempts, "
                        "condition_field, expected_value"
                    ),
                }),
                "max_concurrency": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "tooltip": "最大并发数，控制同时执行的请求/任务数量",
                }),
            },
            "optional": {
                "default_api_key": ("STRING", {"default": "", "tooltip": "全局默认API密钥，各请求可单独覆盖"}),
                "default_headers": ("STRING", {"default": "", "tooltip": "全局默认请求头JSON，各请求可单独覆盖或追加"}),
                "default_timeout": ("INT", {"default": 60, "min": 1, "max": 3600, "tooltip": "全局默认超时时间（秒）"}),
                "stop_on_error": ("BOOLEAN", {"default": False, "tooltip": "任一请求失败时是否中止剩余请求"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("responses", "success_count", "fail_count", "errors")

    FUNCTION = "concurrent_request"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def concurrent_request(self, requests_json, max_concurrency,
                           default_api_key="", default_headers="",
                           default_timeout=60, stop_on_error=False):
        try:
            request_list = _parse_json_input(requests_json, "requests_json", default=[])
            if not isinstance(request_list, list):
                raise ValueError("requests_json 必须是一个JSON数组")
            if not request_list:
                raise ValueError("requests_json 数组不能为空")

            total_count = len(request_list)
            print(f"[CONCURRENT_HTTP_REQUEST] 开始并发编排: 共 {total_count} 个任务, 最大并发 {max_concurrency}")

            global_headers = _build_headers(default_api_key, default_headers)

            results = [None] * total_count
            errors = {}
            success_count = 0
            fail_count = 0
            should_cancel = False

            def execute_task(index, task_descriptor):
                """执行单个任务：普通请求 或 异步提交+轮询"""
                nonlocal should_cancel
                if should_cancel:
                    return index, None, "已取消"

                try:
                    task_config = self._normalize_task_config(task_descriptor, global_headers, default_timeout)
                    is_async = task_config.get("async", False)

                    if is_async:
                        return self._execute_async_task(index, task_config)
                    else:
                        return self._execute_simple_request(index, task_config)

                except Exception as task_error:
                    error_message = f"任务 #{index} 异常: {str(task_error)}"
                    print(f"[CONCURRENT_HTTP_REQUEST] {error_message}")
                    if stop_on_error:
                        should_cancel = True
                    return index, None, error_message

            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                future_to_index = {
                    executor.submit(execute_task, idx, descriptor): idx
                    for idx, descriptor in enumerate(request_list)
                }

                for future in as_completed(future_to_index):
                    index, response_text, error_message = future.result()
                    if error_message:
                        fail_count += 1
                        errors[str(index)] = error_message
                        results[index] = json.dumps({"error": error_message}, ensure_ascii=False)
                    else:
                        success_count += 1
                        results[index] = response_text

            for idx in range(total_count):
                if results[idx] is None:
                    fail_count += 1
                    errors[str(idx)] = "已取消"
                    results[idx] = json.dumps({"error": "已取消"}, ensure_ascii=False)

            responses_output = json.dumps(results, ensure_ascii=False, indent=2)
            errors_output = json.dumps(errors, ensure_ascii=False, indent=2) if errors else "{}"

            print(f"[CONCURRENT_HTTP_REQUEST] 编排完成: 成功 {success_count}, 失败 {fail_count}")

            return (responses_output, success_count, fail_count, errors_output)

        except Exception as error:
            error_msg = f"并发编排错误: {str(error)}"
            print(f"[CONCURRENT_HTTP_REQUEST] {error_msg}")
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                0,
                0,
                json.dumps({"error": error_msg}, ensure_ascii=False),
            )

    def _normalize_task_config(self, task_descriptor, global_headers, default_timeout):
        """将用户输入的任务描述标准化为内部配置"""
        if not isinstance(task_descriptor, dict):
            raise ValueError("每个任务描述必须是一个JSON对象")

        url = task_descriptor.get("url", "").strip()
        if not url:
            raise ValueError("任务缺少必填字段 'url'")

        merged_headers = dict(global_headers)
        task_api_key = task_descriptor.get("api_key", "")
        if task_api_key and task_api_key.strip():
            merged_headers["Authorization"] = f"Bearer {task_api_key.strip()}"

        task_headers = task_descriptor.get("headers")
        if task_headers:
            if isinstance(task_headers, str):
                task_headers = _parse_json_input(task_headers, "task.headers", default={})
            if isinstance(task_headers, dict):
                merged_headers.update(task_headers)

        return {
            "url": url,
            "method": task_descriptor.get("method", "POST").upper(),
            "params": task_descriptor.get("params", {}),
            "headers": merged_headers,
            "timeout": task_descriptor.get("timeout", default_timeout),
            "async": task_descriptor.get("async", False),
            "task_id_field": task_descriptor.get("task_id_field", "data.task_id"),
            "poll_url": task_descriptor.get("poll_url", ""),
            "poll_method": task_descriptor.get("poll_method", "GET").upper(),
            "poll_params": task_descriptor.get("poll_params", {}),
            "poll_interval": task_descriptor.get("poll_interval", 3),
            "max_poll_attempts": task_descriptor.get("max_poll_attempts", 20),
            "condition_field": task_descriptor.get("condition_field", "data.status"),
            "expected_value": task_descriptor.get("expected_value", "completed"),
            "task_id_placeholder": task_descriptor.get("task_id_placeholder", "{task_id}"),
        }

    def _execute_simple_request(self, index, config):
        """执行普通的一次性HTTP请求"""
        method = config["method"]
        print(f"[CONCURRENT_HTTP_REQUEST] 任务 #{index} 发送 {method} {config['url']}")

        response = requests.request(
            method,
            config["url"],
            headers=config["headers"],
            json=config["params"] if method in ("POST", "PUT", "PATCH") else None,
            params=config["params"] if method in ("GET", "DELETE", "HEAD") else None,
            timeout=config["timeout"],
            verify=False,
        )
        response.raise_for_status()
        response_text = _format_response_text(response)
        print(f"[CONCURRENT_HTTP_REQUEST] 任务 #{index} 成功 (HTTP {response.status_code})")
        return index, response_text, None

    def _execute_async_task(self, index, config):
        """执行异步任务：提交请求 → 提取task_id → 轮询等待结果"""
        submit_method = config["method"]
        print(f"[CONCURRENT_HTTP_REQUEST] 任务 #{index} [异步] 提交 {submit_method} {config['url']}")

        submit_response = requests.request(
            submit_method,
            config["url"],
            headers=config["headers"],
            json=config["params"] if submit_method in ("POST", "PUT", "PATCH") else None,
            params=config["params"] if submit_method in ("GET", "DELETE", "HEAD") else None,
            timeout=config["timeout"],
            verify=False,
        )
        submit_response.raise_for_status()

        try:
            submit_data = submit_response.json()
        except json.JSONDecodeError as parse_error:
            raise ValueError(f"提交响应不是有效JSON: {str(parse_error)}") from parse_error

        task_id = _get_nested_value(submit_data, config["task_id_field"])
        if task_id is None or str(task_id).strip() == "":
            raise ValueError(
                f"无法从提交响应中提取任务ID (字段: {config['task_id_field']}), "
                f"响应: {json.dumps(submit_data, ensure_ascii=False)[:200]}"
            )

        print(f"[CONCURRENT_HTTP_REQUEST] 任务 #{index} [异步] 提交成功, task_id={task_id}")

        placeholder = config["task_id_placeholder"]
        poll_url = _inject_task_id(config["poll_url"], task_id, placeholder)
        poll_params = _inject_task_id(config["poll_params"], task_id, placeholder)
        poll_method = config["poll_method"]
        poll_interval = config["poll_interval"]
        max_poll_attempts = config["max_poll_attempts"]
        condition_field = config["condition_field"]
        expected_value = config["expected_value"]

        if not poll_url:
            raise ValueError("异步任务缺少 poll_url 配置")

        for attempt in range(1, max_poll_attempts + 1):
            if attempt > 1:
                time.sleep(poll_interval)

            print(f"[CONCURRENT_HTTP_REQUEST] 任务 #{index} [异步] 轮询 {attempt}/{max_poll_attempts}")

            poll_response = requests.request(
                poll_method,
                poll_url,
                headers=config["headers"],
                json=poll_params if poll_method in ("POST", "PUT", "PATCH") else None,
                params=poll_params if poll_method in ("GET", "DELETE", "HEAD") else None,
                timeout=config["timeout"],
                verify=False,
            )
            poll_response.raise_for_status()

            try:
                poll_data = poll_response.json()
            except json.JSONDecodeError:
                poll_data = {"raw_response": poll_response.text}

            current_value = _get_nested_value(poll_data, condition_field)

            if current_value is not None and (
                current_value == expected_value or str(current_value) == expected_value
            ):
                result_text = json.dumps(poll_data, ensure_ascii=False, indent=2)
                print(
                    f"[CONCURRENT_HTTP_REQUEST] 任务 #{index} [异步] ✓ 完成 "
                    f"({condition_field}={current_value}), 共轮询 {attempt} 次"
                )
                return index, result_text, None

            print(
                f"[CONCURRENT_HTTP_REQUEST] 任务 #{index} [异步] "
                f"当前 {condition_field}={current_value}, 期望 {expected_value}"
            )

        last_text = _format_response_text(poll_response)
        error_message = (
            f"任务 #{index} 轮询超时: 已达最大尝试次数 {max_poll_attempts}, "
            f"最后状态 {condition_field}={current_value}"
        )
        print(f"[CONCURRENT_HTTP_REQUEST] {error_message}")
        return index, last_text, error_message


class ASYNC_TASK_HTTP_REQUEST(POLLING_HTTP_REQUEST):
    """任务提交 + 轮询一体化节点"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "submit_method": (["POST", "GET", "PUT", "DELETE", "PATCH"], {}),
                "submit_endpoint": ("STRING", {"default": ""}),
                "submit_params": ("STRING", {"default": "{}", "multiline": True}),
                "task_id_field": ("STRING", {"default": "data.task_id", "tooltip": "从提交响应中提取任务ID的字段路径"}),
                "poll_method": (["GET", "POST", "PUT", "DELETE", "PATCH"], {}),
                "poll_endpoint": ("STRING", {"default": "", "tooltip": "轮询地址，支持 {task_id} 占位符"}),
                "poll_params": ("STRING", {"default": "{}", "multiline": True, "tooltip": "轮询参数，支持 {task_id} 占位符"}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 60, "tooltip": "轮询间隔（秒）"}),
                "max_attempts": ("INT", {"default": 20, "min": 1, "max": 200, "tooltip": "最大轮询次数"}),
                "success_condition": (["status_field", "custom_jsonpath", "status_code_only"], {
                    "tooltip": "轮询成功条件类型"
                }),
                "condition_field": ("STRING", {"default": "data.status", "tooltip": "轮询响应中的状态字段路径"}),
                "expected_value": ("STRING", {"default": "completed", "tooltip": "期望状态值"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "headers": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 30, "min": 1, "max": 600}),
                "stop_on_error": ("BOOLEAN", {"default": True, "tooltip": "遇到错误时是否立即停止"}),
                "task_id_placeholder": ("STRING", {"default": "{task_id}", "tooltip": "在轮询地址和参数中替换任务ID的占位符"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("submit_response", "task_id", "poll_response", "attempts", "success", "final_status")

    FUNCTION = "request_and_poll"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def request_and_poll(self, submit_method, submit_endpoint, submit_params, task_id_field,
                         poll_method, poll_endpoint, poll_params, poll_interval, max_attempts,
                         success_condition, condition_field, expected_value,
                         api_key="", headers="", timeout=30, stop_on_error=True,
                         task_id_placeholder="{task_id}"):
        try:
            if not submit_endpoint or submit_endpoint.strip() == "":
                raise ValueError("提交API端点不能为空")

            if not poll_endpoint or poll_endpoint.strip() == "":
                raise ValueError("轮询API端点不能为空")

            submit_body = _parse_json_input(submit_params, "submit_params", default={})
            real_headers = _build_headers(api_key, headers)

            print(f"[ASYNC_TASK_HTTP_REQUEST] 开始提交任务: {submit_endpoint}")
            print(f"[ASYNC_TASK_HTTP_REQUEST] real_headers: {real_headers}")
            print(f"[ASYNC_TASK_HTTP_REQUEST] 提交任务: {submit_endpoint}")
            print(f"[ASYNC_TASK_HTTP_REQUEST] 提交参数: {json.dumps(submit_body, ensure_ascii=False)}")

            submit_response = requests.request(
                submit_method,
                submit_endpoint.strip(),
                headers=real_headers,
                json=submit_body if submit_method in ["POST", "PUT", "PATCH"] else None,
                params=submit_body if submit_method in ["GET", "DELETE"] else None,
                timeout=timeout,
                verify=False,
            )
            print(f"[ASYNC_TASK_HTTP_REQUEST] 提交响应状态码: {submit_response.status_code}")
            print(f"[ASYNC_TASK_HTTP_REQUEST] 提交响应内容: {submit_response.text[:500]}")
            submit_response.raise_for_status()

            submit_response_data = submit_response.json()
            submit_response_text = json.dumps(submit_response_data, ensure_ascii=False, indent=2)
            task_id = _get_nested_value(submit_response_data, task_id_field)

            if task_id is None or str(task_id).strip() == "":
                raise ValueError(f"无法从提交响应中提取任务ID，字段路径: {task_id_field}")

            print(f"[ASYNC_TASK_HTTP_REQUEST] 提交成功，任务ID: {task_id}")

            resolved_poll_endpoint = _inject_task_id(poll_endpoint, task_id, task_id_placeholder)
            resolved_poll_params_obj = _inject_task_id(
                _parse_json_input(poll_params, "poll_params", default={}),
                task_id,
                task_id_placeholder,
            )
            resolved_poll_params = json.dumps(resolved_poll_params_obj, ensure_ascii=False)

            poll_response, attempts, success, final_status = self.poll_request(
                method=poll_method,
                api_endpoint=resolved_poll_endpoint,
                params=resolved_poll_params,
                poll_interval=poll_interval,
                max_attempts=max_attempts,
                success_condition=success_condition,
                condition_field=condition_field,
                expected_value=expected_value,
                api_key=api_key,
                headers=headers,
                timeout=timeout,
                stop_on_error=stop_on_error,
            )

            return (
                submit_response_text,
                str(task_id),
                poll_response,
                attempts,
                success,
                final_status,
            )

        except requests.exceptions.RequestException as e:
            error_msg = f"任务提交失败: {str(e)}"
            print(f"[ASYNC_TASK_HTTP_REQUEST] {error_msg}")
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                "",
                json.dumps({"error": error_msg}, ensure_ascii=False),
                0,
                False,
                error_msg,
            )
        except json.JSONDecodeError as e:
            error_msg = f"提交响应JSON解析失败: {str(e)}"
            print(f"[ASYNC_TASK_HTTP_REQUEST] {error_msg}")
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                "",
                json.dumps({"error": error_msg}, ensure_ascii=False),
                0,
                False,
                error_msg,
            )
        except Exception as e:
            error_msg = f"任务处理失败: {str(e)}"
            print(f"[ASYNC_TASK_HTTP_REQUEST] {error_msg}")
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                "",
                json.dumps({"error": error_msg}, ensure_ascii=False),
                0,
                False,
                error_msg,
            )
