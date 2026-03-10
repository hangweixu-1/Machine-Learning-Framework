#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown数据提取工具
从文件夹中的所有Markdown文件提取结构化数据，使用DeepSeek V3 API，输出到CSV
"""

import os
import glob
import json
import csv
import time
import re
from typing import Dict, List, Optional, Any
from openai import OpenAI

# ==================== 配置区域 ====================
# 硅基流动 DeepSeek V3 API配置
API_KEY = "your_api_key_here"  # 替换为你的API Key
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "deepseek-ai/DeepSeek-V3"  # DeepSeek V3模型

# 文件夹配置
INPUT_MD_FOLDER = "corrected_markdowns"  # 输入的markdown文件夹
OUTPUT_CSV_FILE = "extracted_data.csv"  # 输出的CSV文件名

# API调用配置
MAX_RETRIES = 5
MIN_API_INTERVAL = 0.5  # API调用最小间隔（秒）


# ================================================


class MarkdownDataExtractor:
    """Markdown数据提取器"""

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL, model: str = MODEL):
        """初始化提取器"""
        print("🚀 Markdown数据提取工具启动")

        # 初始化API客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

        # API调用统计
        self.api_success_count = 0
        self.api_failure_count = 0
        self.last_api_call_time = 0

        # 调试模式
        self.debug_mode = True

    def call_api(self, system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
        """
        调用DeepSeek API - 带重试机制和速率控制

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            max_tokens: 最大token数

        Returns:
            API返回的文本内容
        """
        base_delay = 1.0
        max_delay = 60.0

        # 速率控制
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < MIN_API_INTERVAL:
            time.sleep(MIN_API_INTERVAL - time_since_last_call)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    stream=False
                )

                result = response.choices[0].message.content.strip()
                if result:
                    self.api_success_count += 1
                    self.last_api_call_time = time.time()
                    return result

            except Exception as e:
                error_msg = str(e)
                if self.debug_mode:
                    print(f"\n  [API错误 {attempt + 1}/{MAX_RETRIES}]: {error_msg}")

                # 判断延迟时间
                if "rate limit" in error_msg.lower():
                    delay = min(base_delay * (3 ** attempt), max_delay)
                elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    delay = min(base_delay * (2 ** attempt), max_delay)
                else:
                    delay = base_delay * (attempt + 1)

                if attempt < MAX_RETRIES - 1:
                    if self.debug_mode:
                        print(f"  [等待 {delay:.1f} 秒后重试...]")
                    time.sleep(delay)
                    continue

                self.api_failure_count += 1

        return ""

    def parse_json_response(self, response: str) -> Optional[Dict]:
        """
        解析API返回的JSON

        Args:
            response: API返回的原始文本

        Returns:
            解析后的字典，失败返回None
        """
        if not response:
            return None

        response = response.strip()

        # 移除markdown代码块标记
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()

        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON对象
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON数组
        try:
            start = response.find('[')
            end = response.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end + 1]
                return {"data": json.loads(json_str)}
        except json.JSONDecodeError:
            pass

        return None

    def read_markdown_file(self, file_path: str) -> str:
        """
        读取Markdown文件内容

        Args:
            file_path: 文件路径

        Returns:
            文件内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"  [错误] 无法读取文件 {file_path}: {e}")
            return ""

    def extract_data_from_markdown(self, markdown_content: str, filename: str) -> Optional[Dict]:
        """
        从单个Markdown文件中提取数据

        Args:
            markdown_content: Markdown文件内容
            filename: 文件名（用于记录）

        Returns:
            提取的数据字典
        """
        if not markdown_content or len(markdown_content.strip()) < 100:
            print(f"  [跳过] 文件内容太短或为空")
            return None

        # 如果内容太长，进行截断（保留前后部分）
        max_content_length = 15000
        if len(markdown_content) > max_content_length:
            half_len = max_content_length // 2
            markdown_content = (
                    markdown_content[:half_len] +
                    "\n\n... [内容已截断] ...\n\n" +
                    markdown_content[-half_len:]
            )

        # ==================== 自定义提示词区域 ====================
        # 根据你的具体需求修改下面的提示词

        system_prompt = """你是一个专业的化学文献数据提取专家。

你的任务是：从用户提供的化学文献Markdown文本中，提取以下关键信息：

1. **文献基本信息**:
   - title: 文献标题
   - authors: 作者列表
   - journal: 期刊名称
   - year: 发表年份
   - doi: DOI号（如有）

2. **实验信息**:
   - catalysts: 催化剂信息列表（名称、组成、制备方法等）
   - reaction_conditions: 反应条件（温度、压力、时间等）
   - reactants: 反应物
   - products: 产物

3. **性能数据**:
   - conversion: 转化率数据
   - selectivity: 选择性数据
   - yield: 产率数据
   - other_metrics: 其他性能指标

4. **表征信息**:
   - characterization_methods: 使用的表征方法（XRD, TEM, BET等）
   - key_findings: 主要发现和结论

提取规则：
- 如果某个字段在文献中找不到，填写 null 或空字符串
- 数值数据请保留原始单位
- 催化剂名称请保持化学式格式（如 PdZn/ZnO, Cu/ZnO 等）
- 对于列表类型的数据，如果只有一个值，也请用列表格式
- 尽量从文献中提取原始数据，不要推测

输出要求：
- 必须以JSON格式返回
- 确保JSON格式正确，可以被解析"""

        user_prompt = f"""请从以下化学文献Markdown文本中提取数据：

文件名: {filename}

---文献内容开始---
{markdown_content}
---文献内容结束---

请以JSON格式返回提取的数据。"""

        # ==================== 提示词区域结束 ====================

        response = self.call_api(system_prompt, user_prompt)
        if not response:
            return None

        result = self.parse_json_response(response)
        if result:
            # 添加源文件名
            result['source_file'] = filename

        return result

    def flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        将嵌套字典展平为单层字典

        Args:
            d: 嵌套字典
            parent_key: 父键前缀
            sep: 分隔符

        Returns:
            展平后的字典
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # 将列表转换为字符串
                if all(isinstance(item, (str, int, float, bool)) or item is None for item in v):
                    items.append((new_key, '; '.join(str(item) for item in v if item is not None)))
                else:
                    # 对于复杂列表，转换为JSON字符串
                    items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, v))

        return dict(items)

    def process_folder(self, input_folder: str, output_csv: str) -> None:
        """
        处理整个文件夹中的所有Markdown文件

        Args:
            input_folder: 输入文件夹路径
            output_csv: 输出CSV文件路径
        """
        # 获取所有markdown文件
        md_files = glob.glob(os.path.join(input_folder, "*.md"))

        if not md_files:
            print(f"[!] 在 {input_folder} 中未找到任何Markdown文件")
            return

        print(f"\n📁 找到 {len(md_files)} 个Markdown文件")

        all_data = []
        all_keys = set()

        for i, md_file in enumerate(md_files, 1):
            filename = os.path.basename(md_file)
            print(f"\n{'=' * 60}")
            print(f"📄 处理文件 [{i}/{len(md_files)}]: {filename}")
            print(f"{'=' * 60}")

            try:
                # 读取文件
                content = self.read_markdown_file(md_file)
                if not content:
                    continue

                print(f"  文件大小: {len(content)} 字符")

                # 提取数据
                extracted = self.extract_data_from_markdown(content, filename)

                if extracted:
                    # 展平数据
                    flat_data = self.flatten_dict(extracted)
                    all_data.append(flat_data)
                    all_keys.update(flat_data.keys())
                    print(f"  ✓ 成功提取 {len(flat_data)} 个字段")
                else:
                    print(f"  ✗ 提取失败")
                    # 添加一个只有文件名的记录
                    all_data.append({'source_file': filename, 'extraction_status': 'failed'})
                    all_keys.update(['source_file', 'extraction_status'])

            except KeyboardInterrupt:
                print("\n\n[!] 用户中断处理")
                break
            except Exception as e:
                print(f"  ✗ 处理出错: {e}")
                all_data.append({'source_file': filename, 'extraction_status': f'error: {str(e)}'})
                all_keys.update(['source_file', 'extraction_status'])
                continue

        # 写入CSV
        if all_data:
            self.write_to_csv(all_data, all_keys, output_csv)
        else:
            print("\n[!] 没有提取到任何数据")

    def write_to_csv(self, data: List[Dict], keys: set, output_file: str) -> None:
        """
        将提取的数据写入CSV文件

        Args:
            data: 数据列表
            keys: 所有字段名集合
            output_file: 输出文件路径
        """
        # 排序字段，确保source_file在最前面
        sorted_keys = sorted(keys)
        if 'source_file' in sorted_keys:
            sorted_keys.remove('source_file')
            sorted_keys = ['source_file'] + sorted_keys

        try:
            with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=sorted_keys, extrasaction='ignore')
                writer.writeheader()

                for row in data:
                    # 确保所有值都是字符串类型
                    clean_row = {}
                    for k, v in row.items():
                        if v is None:
                            clean_row[k] = ''
                        elif isinstance(v, (dict, list)):
                            clean_row[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            clean_row[k] = str(v)
                    writer.writerow(clean_row)

            print(f"\n{'=' * 60}")
            print(f"✅ 数据已保存到: {output_file}")
            print(f"   共 {len(data)} 条记录, {len(sorted_keys)} 个字段")
            print(f"{'=' * 60}")

        except Exception as e:
            print(f"\n[!] 写入CSV失败: {e}")

    def print_summary(self) -> None:
        """打印处理摘要"""
        print(f"\n{'=' * 60}")
        print("📊 处理摘要")
        print(f"{'=' * 60}")
        print(f"API调用成功: {self.api_success_count}")
        print(f"API调用失败: {self.api_failure_count}")
        if self.api_success_count + self.api_failure_count > 0:
            success_rate = self.api_success_count / (self.api_success_count + self.api_failure_count) * 100
            print(f"成功率: {success_rate:.1f}%")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='从Markdown文件夹提取数据到CSV')
    parser.add_argument('--input-folder', '-i', type=str, default=INPUT_MD_FOLDER,
                        help=f'输入Markdown文件夹路径 (默认: {INPUT_MD_FOLDER})')
    parser.add_argument('--output-csv', '-o', type=str, default=OUTPUT_CSV_FILE,
                        help=f'输出CSV文件路径 (默认: {OUTPUT_CSV_FILE})')
    parser.add_argument('--api-key', type=str, default=API_KEY,
                        help='硅基流动API Key')
    parser.add_argument('--model', type=str, default=MODEL,
                        help=f'使用的模型 (默认: {MODEL})')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')

    args = parser.parse_args()

    # 检查输入文件夹
    if not os.path.exists(args.input_folder):
        print(f"[!] 输入文件夹不存在: {args.input_folder}")
        print(f"    请确保文件夹路径正确，或使用 --input-folder 参数指定")
        return

    # 检查API Key
    if args.api_key == "your_api_key_here":
        print("[!] 请设置正确的API Key")
        print("    方法1: 修改脚本中的 API_KEY 变量")
        print("    方法2: 使用 --api-key 参数指定")
        return

    # 创建提取器并处理
    extractor = MarkdownDataExtractor(
        api_key=args.api_key,
        model=args.model
    )
    extractor.debug_mode = args.debug

    # 处理文件夹
    extractor.process_folder(args.input_folder, args.output_csv)

    # 打印摘要
    extractor.print_summary()


if __name__ == "__main__":
    main()