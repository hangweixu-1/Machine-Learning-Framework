#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF化学文献转换与语义纠错系统 - 增强版v6
主要增强：
1. 添加逐篇文献改进效果散点图
2. 添加配对t检验统计分析
3. 增强可视化和统计报告
"""

import os
import glob
import json
import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, SupportedPdfParseMethod
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from openai import OpenAI
import numpy as np
import traceback
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import seaborn as sns

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# DeepSeek API配置
API_KEY = "your api"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "model"

# 配置参数
input_pdf_folder = "input"
output_md_folder = "output_markdowns2"
corrected_md_folder = "corrected_markdowns"
stats_file = "conversion_correction_stats.json"
correction_reports_folder = "correction_reports"
visualization_folder = "visualizations"


@dataclass
class ErrorStats:
    """错误统计"""
    semantic_errors: int = 0
    semantic_error_chars: int = 0
    semantic_error_examples: List[str] = field(default_factory=list)

    total_characters: int = 0
    actual_content_chars: int = 0
    ocr_accuracy: float = 100.0
    ocr_cer: float = 0.0
    ocr_quality: str = "未评估"

    total_errors: int = 0
    error_rate: float = 0.0


@dataclass
class ConversionCorrectionStats:
    """转换和纠错统计信息"""
    pdf_name: str
    total_pages: int = 0
    total_characters: int = 0
    errors_before: ErrorStats = field(default_factory=ErrorStats)
    errors_after: ErrorStats = field(default_factory=ErrorStats)
    correction_count: int = 0
    corrections_made: List[str] = field(default_factory=list)
    api_calls: int = 0
    api_success: int = 0
    api_failures: int = 0
    processing_time: float = 0.0
    conversion_time: float = 0.0
    correction_time: float = 0.0
    ocr_mode: bool = False

    # 新增字段
    detected_errors_count: int = 0  # API检测到的错误总数
    applied_corrections_count: int = 0  # 实际应用的纠正数


class StructureProtector:
    """保护文档结构元素的类"""

    def __init__(self):
        self.protected_elements = {}
        self.counter = 0
        self.placeholder_prefix = "__STRUCTURE_PROTECTOR_"
        self.placeholder_suffix = "__"

    def protect(self, text: str) -> str:
        """保护文档中的结构元素"""
        # 保护Markdown标题
        text = self._protect_headers(text)

        # 保护整个表格（包括内容）
        text = self._protect_tables_complete(text)

        # 保护图片
        text = self._protect_images(text)

        # 保护代码块
        text = self._protect_code_blocks(text)

        return text

    def unprotect(self, text: str) -> str:
        """恢复被保护的结构元素"""
        for placeholder, original in sorted(self.protected_elements.items(),
                                            key=lambda x: len(x[0]), reverse=True):
            text = text.replace(placeholder, original)
        return text

    def _create_placeholder(self) -> str:
        """创建一个新的占位符"""
        placeholder = f"{self.placeholder_prefix}{self.counter}{self.placeholder_suffix}"
        self.counter += 1
        return placeholder

    def _protect_pattern(self, text: str, pattern: re.Pattern, multiline: bool = True) -> str:
        """通用的保护方法"""
        flags = re.MULTILINE | re.DOTALL if multiline else 0

        def replacer(match):
            original = match.group(0)
            placeholder = self._create_placeholder()
            self.protected_elements[placeholder] = original
            return placeholder

        return pattern.sub(replacer, text)

    def _protect_headers(self, text: str) -> str:
        """保护Markdown标题"""
        # 保护 # 到 ###### 的标题
        pattern = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)
        return self._protect_pattern(text, pattern)

    def _protect_tables_complete(self, text: str) -> str:
        """完全保护表格（包括内容）"""
        # 保护HTML表格
        pattern = re.compile(r'<table.*?>.*?</table>', re.DOTALL | re.IGNORECASE)
        text = self._protect_pattern(text, pattern)

        # 保护Markdown表格
        lines = text.split('\n')
        in_table = False
        table_start = -1

        for i, line in enumerate(lines):
            # 检测表格开始（包含|的行）
            if '|' in line and not in_table:
                in_table = True
                table_start = i
            # 检测表格结束（不包含|的行）
            elif in_table and '|' not in line:
                # 保护整个表格
                if table_start != -1:
                    table_content = '\n'.join(lines[table_start:i])
                    placeholder = self._create_placeholder()
                    self.protected_elements[placeholder] = table_content
                    # 替换表格内容为占位符
                    for j in range(table_start, i):
                        if j == table_start:
                            lines[j] = placeholder
                        else:
                            lines[j] = ''
                in_table = False
                table_start = -1

        # 处理最后一个表格（如果文件以表格结束）
        if in_table and table_start != -1:
            table_content = '\n'.join(lines[table_start:])
            placeholder = self._create_placeholder()
            self.protected_elements[placeholder] = table_content
            for j in range(table_start, len(lines)):
                if j == table_start:
                    lines[j] = placeholder
                else:
                    lines[j] = ''

        return '\n'.join(line for line in lines if line)  # 移除空行

    def _protect_images(self, text: str) -> str:
        """保护图片引用"""
        pattern = re.compile(r'!\[.*?\]\(.*?\)')
        return self._protect_pattern(text, pattern)

    def _protect_code_blocks(self, text: str) -> str:
        """保护代码块"""
        # 保护三个反引号的代码块
        pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        text = self._protect_pattern(text, pattern)

        # 保护单行代码
        pattern = re.compile(r'`[^`\n]+`')
        return self._protect_pattern(text, pattern)


def is_context_nitrogen_gas(text: str, match_start: int, match_end: int) -> bool:
    """
    判断匹配的文本在上下文中是否指氮气（N₂）。
    返回 True 表示应该是氮气 N₂，False 表示应该是单个 N。
    """
    # 获取匹配前后的上下文（前后各50个字符）
    context_before = text[max(0, match_start - 50):match_start].lower()
    context_after = text[match_end:min(len(text), match_end + 50)].lower()

    # 指示氮气的关键词
    nitrogen_gas_indicators = [
        'gas', 'atmosphere', 'flow', 'stream', 'purge', 'inert',
        'ambient', 'blanket', 'protective', 'carrier'
    ]

    # 指示化合物名称的关键词（如硝酸盐、硝酸等）
    compound_indicators = [
        'nitrate', 'nitrite', 'nitric', 'nitrogen(', 'palladium(',
        'tetraammine', 'complex', 'salt', 'precursor', 'compound'
    ]

    # 检查是否在化合物上下文中
    for indicator in compound_indicators:
        if indicator in context_before or indicator in context_after:
            return False  # 是化合物中的 N，不应该是 N₂

    # 检查是否在氮气上下文中
    for indicator in nitrogen_gas_indicators:
        if indicator in context_before or indicator in context_after:
            return True  # 可能是氮气 N₂

    # 默认情况下，如果不确定，保持为 N
    return False


# 增强版规则错误收集函数
def gather_rule_based_errors(text: str) -> List[Dict]:
    """
    只收集基于规则的错误，不修改文本。
    增强了规则匹配的灵活性和覆盖面。
    """
    errors = []

    # 首先处理需要上下文判断的特殊情况
    # 对于 /NU, \NU 等变体，需要根据上下文决定是 N 还是 N₂
    # 使用零宽后行断言 (?<=\W) 和零宽前行断言 (?=\W|$)
    # 这意味着"前面是一个非单词字符（或行首），后面也是一个非单词字符（或行尾）"
    nu_pattern = re.compile(r'(?:^|(?<=\W))([/\\]?\s*N\s*[Uu]\b)')
    for match in nu_pattern.finditer(text):
        # 注意：因为我们加了一层捕获组，实际要替换的内容在 group(1)
        error_text = match.group(1)
        if is_context_nitrogen_gas(text, match.start(1), match.end(1)):
            # 在氮气上下文中，转换为 N₂
            errors.append({
                'text': error_text,
                'correct': 'N₂',
                'confidence': 0.90,
                'type': 'rule_based'
            })
        else:
            # 在化合物上下文中，转换为 N
            errors.append({
                'text': error_text,
                'correct': 'N',
                'confidence': 0.95,
                'type': 'rule_based'
            })

    chemical_rules = [
        # 基础化学式
        (r'\bCO2\b', 'CO₂', 0.95),
        (r'\bH2O\b', 'H₂O', 0.95),
        (r'\bN2\b', 'N₂', 0.95),
        (r'\bO2\b', 'O₂', 0.95),
        (r'\bH2\b', 'H₂', 0.95),
        (r'\bCH4\b', 'CH₄', 0.95),
        (r'\bNH3\b', 'NH₃', 0.95),
        (r'\bH2SO4\b', 'H₂SO₄', 0.95),
        (r'\bNa2CO3\b', 'Na₂CO₃', 0.95),
        (r'\bCaCO3\b', 'CaCO₃', 0.95),
        (r'\bNaHCO3\b', 'NaHCO₃', 0.95),
        (r'\bMgSO4\b', 'MgSO₄', 0.95),
        (r'\bH3PO4\b', 'H₃PO₄', 0.95),
        (r'\bC2H5OH\b', 'C₂H₅OH', 0.95),
        (r'\bC6H12O6\b', 'C₆H₁₂O₆', 0.95),

        # 常见作者名错误
        (r'Dı´ez-Ramı´rez', 'Díez-Ramírez', 0.98),
        (r'Sa´nchez', 'Sánchez', 0.98),
        (r'Rodrı´guez', 'Rodríguez', 0.98),
        (r'Garcı´a', 'García', 0.98),
        (r'Martı´nez', 'Martínez', 0.98),
        (r'Go´mez', 'Gómez', 0.98),
        (r'Pe´rez', 'Pérez', 0.98),
        (r'Herna´ndez', 'Hernández', 0.98),

        # 特殊字符的变体修正
        (r'([a-zA-Z])ı´([a-zA-Z])', r'\1í\2', 0.95),  # 修正任意位置的 ı´ -> í
        (r'([a-zA-Z])a´([a-zA-Z])', r'\1á\2', 0.95),  # 修正任意位置的 a´ -> á
        (r'([a-zA-Z])o´([a-zA-Z])', r'\1ó\2', 0.95),  # 修正任意位置的 o´ -> ó
        (r'([a-zA-Z])e´([a-zA-Z])', r'\1é\2', 0.95),  # 修正任意位置的 e´ -> é
        (r'([a-zA-Z])u´([a-zA-Z])', r'\1ú\2', 0.95),  # 修正任意位置的 u´ -> ú

        # 单位错误（增强版）
        (r'μ\s+m\b', 'μm', 0.98),
        (r'm\s+L\b', 'mL', 0.98),
        (r'n\s+m\b', 'nm', 0.98),
        (r'°\s+C\b', '°C', 0.98),
        (r'K\s*/\s*min\b', 'K/min', 0.95),
        (r'mol\s*/\s*L\b', 'mol/L', 0.95),
        (r'g\s*/\s*cm3\b', 'g/cm³', 0.95),
        (r'cm\s*-\s*1\b', 'cm⁻¹', 0.95),
        (r'm\s*2\s*/\s*g\b', 'm²/g', 0.95),
        (r'cm\s*2\b', 'cm²', 0.95),
        (r'cm\s*3\b', 'cm³', 0.95),

        # 常见OCR错误
        (r'ﬁ', 'fi', 0.95),
        (r'ﬂ', 'fl', 0.95),
        (r'ﬀ', 'ff', 0.95),
        (r'ﬃ', 'ffi', 0.95),
        (r'ﬄ', 'ffl', 0.95),
        (r'\bFFig\b', 'Fig', 0.95),  # 修正 FFig -> Fig
        (r'\bTTable\b', 'Table', 0.95),  # 修正 TTable -> Table

        # 元素符号分离错误（增强版）
        (r'\bC\s+u\b', 'Cu', 0.90),
        (r'\bZ\s+n\b', 'Zn', 0.90),
        (r'\bA\s+g\b', 'Ag', 0.90),
        (r'\bP\s+d\b', 'Pd', 0.90),
        (r'\bP\s+t\b', 'Pt', 0.90),
        (r'\bA\s+u\b', 'Au', 0.90),
        (r'\bF\s+e\b', 'Fe', 0.90),
        (r'\bN\s+i\b', 'Ni', 0.90),
        (r'\bC\s+o\b', 'Co', 0.90),
        (r'\bM\s+n\b', 'Mn', 0.90),
        (r'\bC\s+r\b', 'Cr', 0.90),
        (r'\bT\s+i\b', 'Ti', 0.90),
        (r'\bS\s+i\b', 'Si', 0.90),
        (r'\bA\s+l\b', 'Al', 0.90),
        (r'\bM\s+g\b', 'Mg', 0.90),
        (r'\bC\s+a\b', 'Ca', 0.90),
        (r'\bN\s+a\b', 'Na', 0.90),

        # 化学术语常见拼写错误
        (r'\bcatalystt\b', 'catalyst', 0.95),
        (r'\bmethanool\b', 'methanol', 0.95),
        (r'\bethanool\b', 'ethanol', 0.95),
        (r'\btemperture\b', 'temperature', 0.95),
        (r'\bpressur\b', 'pressure', 0.95),
        (r'\breacction\b', 'reaction', 0.95),
        (r'\bconcentracion\b', 'concentration', 0.95),
        (r'\bsolubilty\b', 'solubility', 0.95),
    ]

    for pattern, replacement, confidence in chemical_rules:
        # 对于包含捕获组的模式，使用不同的处理方式
        if '(' in pattern and ')' in pattern and r'\1' in replacement:
            for match in re.finditer(pattern, text):
                # 使用 re.sub 来应用捕获组替换
                original_text = match.group(0)
                corrected_text = re.sub(pattern, replacement, original_text)
                errors.append({
                    'text': original_text,
                    'correct': corrected_text,
                    'confidence': confidence,
                    'type': 'rule_based'
                })
        else:
            # 普通模式匹配
            for match in re.finditer(pattern, text):
                errors.append({
                    'text': match.group(0),
                    'correct': replacement,
                    'confidence': confidence,
                    'type': 'rule_based'
                })

    return errors


def sanitize_llm_errors(llm_errors: List[Dict]) -> List[Dict]:
    """
    对LLM返回的错误进行理智检查，过滤掉危险的修正。
    """
    sanitized_errors = []
    for err in llm_errors:
        # 危险操作：将一个纯数字字符串修正为一个不完全是数字的字符串
        is_dangerous_numeric_change = (
                err.get('text', '').replace('.', '', 1).isdigit() and
                not err.get('correct', '').replace('.', '', 1).isdigit()
        )

        if is_dangerous_numeric_change:
            # 不再允许任何数字到非数字的转换（包括'1'到'-'）
            print(f"\n  [理智检查] 过滤掉危险修正: {err.get('text')} → {err.get('correct')}")
        else:
            # 对于所有非危险操作，直接接受
            sanitized_errors.append(err)

    return sanitized_errors


class OptimizedChemicalCorrector:
    """优化后的化学文献纠错器"""

    def __init__(self):
        """初始化纠错器"""
        print("🚀 智能纠错系统启动（增强版v6）")

        # 初始化DeepSeek客户端
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        # API调用统计
        self.api_success_count = 0
        self.api_failure_count = 0

        # 调试模式
        self.debug_mode = True

        # 纠正映射记录
        self.correction_map = {}  # 存储错误文本到正确文本的映射

        # 降低置信度门槛
        self.confidence_threshold = 0.7  # 从0.8降低到0.7

        # API调用速率控制
        self.last_api_call_time = 0
        self.min_api_interval = 0.5  # 最小API调用间隔（秒）

        # 占位符前缀
        self.placeholder_prefix = "__STRUCTURE_PROTECTOR_"

    def preprocess_text_for_api(self, text: str) -> str:
        """预处理文本以避免API解析问题"""
        # 移除或替换可能导致问题的特殊字符
        text = text.replace('\x00', '')
        text = text.replace('\u0000', '')

        # 规范化换行符
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')

        # 移除不可见的控制字符（保留换行和制表符）
        import string
        allowed_chars = string.printable + '\n\t' + ''.join(chr(i) for i in range(0x80, 0x10000))
        text = ''.join(char for char in text if char in allowed_chars)

        return text

    def call_deepseek_api(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """调用DeepSeek API - 带指数退避和速率控制"""
        max_retries = 5
        base_delay = 1.0  # 基础延迟（秒）
        max_delay = 60.0  # 最大延迟（秒）

        # 速率控制：确保API调用间隔
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.min_api_interval:
            sleep_time = self.min_api_interval - time_since_last_call
            time.sleep(sleep_time)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
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
                    print(f"\n  [API错误 {attempt + 1}/{max_retries}]: {error_msg}")

                # 判断是否应该重试
                if "rate limit" in error_msg.lower():
                    # 速率限制错误，使用更长的延迟
                    delay = min(base_delay * (3 ** attempt), max_delay)
                elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    # 网络错误，使用指数退避
                    delay = min(base_delay * (2 ** attempt), max_delay)
                else:
                    # 其他错误，使用较短延迟
                    delay = base_delay * (attempt + 1)

                if attempt < max_retries - 1:
                    if self.debug_mode:
                        print(f"  [等待 {delay:.1f} 秒后重试...]")
                    time.sleep(delay)
                    continue

                self.api_failure_count += 1

        return ""

    def parse_api_json_response(self, response: str) -> Optional[Dict]:
        """解析API返回的JSON"""
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

        # 尝试解析JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON对象（查找最外层的大括号）
        try:
            # 使用栈来找到匹配的大括号
            start = -1
            bracket_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(response):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        if start == -1:
                            start = i
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0 and start != -1:
                            json_part = response[start:i + 1]
                            return json.loads(json_part)
        except Exception:
            pass

        return None

    def apply_corrections_to_text_robust(self, text: str, errors: List[Dict]) -> Tuple[str, int]:
        """(最终清理和修复版) 更稳健的文本替换方法"""
        if not errors:
            return text, 0

        corrected_text = text
        applied_count = 0

        # 按错误文本长度降序排序
        sorted_errors = sorted(errors, key=lambda x: len(x.get('text', '')), reverse=True)

        # 去重，建立唯一的替换映射
        replacement_map = {}
        for error in sorted_errors:
            if isinstance(error, dict) and 'text' in error and 'correct' in error:
                error_text = str(error['text'])
                correct_text = str(error['correct'])
                if error_text and error_text != correct_text and error_text not in replacement_map:
                    replacement_map[error_text] = correct_text

        # 执行唯一的替换循环
        for error_text, correct_text in replacement_map.items():
            try:
                # 为搜索模式转义特殊字符
                pattern = re.escape(error_text)

                # 使用 finditer 查找所有不重叠的匹配项
                matches = list(re.finditer(pattern, corrected_text))
                occurrences = len(matches)

                if occurrences > 0:
                    # 从后往前替换，避免影响前面匹配项的索引
                    # 这是一个比 re.sub 更安全、更可控的替换方法
                    for match in reversed(matches):
                        start, end = match.span()
                        corrected_text = corrected_text[:start] + correct_text + corrected_text[end:]

                    applied_count += occurrences

                    if self.debug_mode:
                        print(f"    应用纠正: '{error_text}' → '{correct_text}' (×{occurrences})")
            except re.error as e:
                if self.debug_mode:
                    print(f"  [正则替换错误] 跳过: '{error_text}' -> '{correct_text}'. 错误: {e}")

        return corrected_text, applied_count

    def detect_and_correct_chunk(self, text: str) -> List[Dict]:
        """
        (重构后) 只调用LLM来检测错误，并返回错误列表。
        """
        if len(text.strip()) < 50 or text.startswith(self.placeholder_prefix):
            return []

        text_to_check = self.preprocess_text_for_api(text)
        if len(text_to_check) > 3000:
            text_to_check = text_to_check[:3000] + "..."

        # 优化后的系统提示 - 更智能地处理LaTeX公式，并注意N/NU的上下文
        system_prompt = """你是化学文献OCR错误纠正专家。

你的任务是：仔细检查用户提供的文本，找出其中100%确定的OCR识别错误和化学术语格式错误。
所有的错误需要联系上下文仔细核对
常见OCR错误类型（包括但不限于）：
1. **字符识别错误**: 
   - `ı´` 应为 `í`
   - `ﬁ` 应为 `fi`
   - `Z n` 应为 `Zn`
   - `C u` 应为 `Cu`
   - `FFig` 应为 `Fig`
   - `TTable` 应为 `Table`
   - **特别注意**: `NU` 或 `/NU` 的纠正需要看上下文：
     * 在化合物名称中（如 nitrate, palladium(II) nitrate）：`NU` → `N`
     * 在气体环境描述中（如 under NU atmosphere）：`NU` → `N₂`

2. **化学式格式错误**: 
   - `N2` 应为 `N₂`
   - `H2O` 应为 `H₂O`
   - `CO2` 应为 `CO₂`
   - `CH3OH` 中的数字应为下标
   - `P d Z n O 1 - x / Z n O` 应为 `PdZnO₁₋ₓ/ZnO`

3. **明显拼写错误**: 
   - `catalystt` 应为 `catalyst`
   - `methanool` 应为 `methanol`
   - `temperture` 应为 `temperature`
   - `Dı´ez-Ramı´rez` 应为 `Díez-Ramírez`
   - `Sa´nchez` 应为 `Sánchez`

4. **单位和符号错误**: 
   - `μ m` 应为 `μm`
   - `K / min` 应为 `K/min`
   - `m L` 应为 `mL`
   - `° C` 应为 `°C`
   - `cm - 1` 应为 `cm⁻¹`
   - `m 2 / g` 应为 `m²/g`

5. **LaTeX公式处理（更智能的规则）**:
   - **谨慎处理LaTeX公式**（`$...$` 和 `$$...$$`）：
     * **可以修正**公式内部明显的OCR错误（如 `C O2` -> `CO₂`, `μ m` -> `μm`）
     * **可以删除**明显与化学文献无关的、由OCR错误产生的**无意义符号组合**
       例如：`$\mathbf { \nabla } \cdot \mathbf { \varepsilon }$` 或 `$\odot \otimes \oplus$` 
       这些很可能是对装饰性符号、边框或其他非文本元素的误识别，应当删除
     * **可以简化**格式混乱的LaTeX表达式，如去除多余的空格和花括号
       例如：`ca. $0 . 5 { \mathrm { ~ g } } { \dot { } } { \mathrm { ~ } }$` -> `ca. $0.5~\mathrm{g}$`
     * **不要修改**结构完整、看起来是正确的数学或物理公式
     * 如果不确定，请保持原样

重要规则：
- **绝对不要修改表格中的任何内容**，即使看起来像错误。
- **只报告你100%确定的错误**。
- **不要将数字修改为其他符号**（例如，不要将'1'修改为'-'）。
- 保持原文的格式、标点、大小写不变（除非是明显的OCR错误）。
- 不要修改已被保护的占位符（如 `__STRUCTURE_PROTECTOR_...__`）。
- 对于化学式、单位等格式问题，如果你很确定，请给出 >= 0.9 的高置信度。

输出要求：
- 必须以JSON格式返回
- JSON中只包含一个名为 "errors" 的列表
- 每个错误对象包含：
  * "text": 原始错误文本
  * "correct": 建议的正确文本（如果是要删除的内容，correct应为空字符串""）
  * "confidence": 置信度（0.0到1.0）

示例输出：
{
  "errors": [
    {"text": "Dı´ez-Ramı´rez", "correct": "Díez-Ramírez", "confidence": 0.95},
    {"text": "P d Z n O 1 - x / Z n O", "correct": "PdZnO₁₋ₓ/ZnO", "confidence": 0.90},
    {"text": "CO2", "correct": "CO₂", "confidence": 0.95},
    {"text": "μ m", "correct": "μm", "confidence": 0.98},
    {"text": "$\\mathbf { \\nabla } \\cdot \\mathbf { \\varepsilon }$", "correct": "", "confidence": 0.85},
    {"text": "FFig", "correct": "Fig", "confidence": 0.95}
  ]
}"""

        # 用户提示
        user_prompt = f"""请检测以下文本中的OCR错误：

{text_to_check}

只返回JSON格式的错误列表。"""

        response = self.call_deepseek_api(system_prompt, user_prompt, 2000)
        if not response:
            return []

        result = self.parse_api_json_response(response)
        if not result or 'errors' not in result:
            return []

        return result.get('errors', [])

    def identify_corrections(self, original: str, corrected: str) -> List[str]:
        """识别具体的修改"""
        corrections = []

        try:
            matcher = SequenceMatcher(None, original, corrected, autojunk=False)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    old_text = original[i1:i2].strip()
                    new_text = corrected[j1:j2].strip()

                    if old_text and new_text and old_text != new_text:
                        if len(old_text) < 50 and len(new_text) < 50:
                            corrections.append(f"{old_text} → {new_text}")
                elif tag == 'delete':
                    # 处理删除操作（如删除无意义的LaTeX公式）
                    old_text = original[i1:i2].strip()
                    if old_text and len(old_text) < 100:
                        corrections.append(f"{old_text} → [删除]")
        except Exception:
            pass

        return corrections[:20]

    def calculate_ocr_accuracy(self, total_chars: int, error_chars: int) -> Tuple[float, float, str]:
        """计算OCR准确率"""
        if total_chars <= 0:
            return 100.0, 0.0, "无有效内容"

        accuracy = ((total_chars - error_chars) / total_chars) * 100
        accuracy = max(0, min(100, accuracy))

        cer = (error_chars / total_chars) * 100

        # 质量评级
        if cer <= 0.5:
            quality = "优秀 (CER ≤ 0.5%)"
        elif cer <= 1:
            quality = "良好 (CER 0.5-1%)"
        elif cer <= 2:
            quality = "较好 (CER 1-2%)"
        elif cer <= 5:
            quality = "一般 (CER 2-5%)"
        elif cer <= 10:
            quality = "较差 (CER 5-10%)"
        else:
            quality = "很差 (CER > 10%)"

        return accuracy, cer, quality

    def evaluate_text_quality(self, text: str) -> Tuple[List[Dict], int]:
        """
        评估文本质量，检测错误但不修正
        返回：(错误列表, 错误字符总数)
        """
        # 收集规则错误
        rule_errors = gather_rule_based_errors(text)

        # 收集LLM错误
        protector = StructureProtector()
        protected_text = protector.protect(text)
        segments = self.split_content_optimized(protected_text)

        llm_errors = []
        for segment in segments:
            segment_errors = self.detect_and_correct_chunk(segment)
            llm_errors.extend(segment_errors)

        llm_errors = sanitize_llm_errors(llm_errors)

        # 合并所有错误
        all_errors = rule_errors + llm_errors

        # 过滤和去重
        valid_errors = [
            err for err in all_errors
            if isinstance(err, dict) and
               'confidence' in err and
               err['confidence'] >= self.confidence_threshold
        ]

        # 去重
        unique_errors_map = {}
        for err in valid_errors:
            key = (err.get('text', ''), err.get('correct', ''))
            if key not in unique_errors_map:
                unique_errors_map[key] = err

        final_errors = list(unique_errors_map.values())

        # 计算错误字符数
        error_chars = sum(len(err['text']) for err in final_errors)

        return final_errors, error_chars

    def correct_document(self, content: str) -> Tuple[str, List[str], ConversionCorrectionStats]:
        """
        (带调试信息的重构版) 纠正整个文档，采用"收集-应用"模式。
        """
        start_time = time.time()
        stats = ConversionCorrectionStats(pdf_name="current", total_characters=len(content))
        original_content = content

        print("\n🔍 [阶段1/3] 收集确定性错误...")
        rule_errors = gather_rule_based_errors(original_content)
        print(f"  - 规则检测到 {len(rule_errors)} 个错误")

        print("\n🔍 [阶段2/3] 收集LLM模糊错误...")
        protector = StructureProtector()
        print("  - 🛡️  保护文档结构...")
        protected_content = protector.protect(original_content)
        segments = self.split_content_optimized(protected_content)
        print(f"  - 📊 文档分割: 共{len(segments)}段")

        llm_errors = []
        print(f"  - ⚡ 调用LLM... ", end='', flush=True)
        for idx, segment in enumerate(segments):
            if idx % 5 == 0: print(f"{idx}/{len(segments)} ", end='', flush=True)
            segment_errors = self.detect_and_correct_chunk(segment)
            llm_errors.extend(segment_errors)
        print(f"\n  - ✅ LLM处理完成，初步检测到 {len(llm_errors)} 个错误")

        llm_errors = sanitize_llm_errors(llm_errors)
        print(f"  - 🧠 理智检查后，保留 {len(llm_errors)} 个LLM错误")

        print("\n🔍 [阶段3/3] 合并并应用所有修正...")
        all_errors = rule_errors + llm_errors

        # 过滤低置信度错误
        valid_errors = [
            err for err in all_errors
            if isinstance(err, dict) and
               'confidence' in err and
               err['confidence'] >= self.confidence_threshold
        ]
        print(f"  - 💎 置信度过滤: {len(all_errors)} → {len(valid_errors)} 个错误")

        # 去重
        unique_errors_map = {}
        for err in valid_errors:
            key = (err.get('text', ''), err.get('correct', ''))
            if key not in unique_errors_map:
                unique_errors_map[key] = err

        final_errors = list(unique_errors_map.values())
        print(f"  - 🔄 去重后: {len(final_errors)} 个唯一错误")

        # 应用修正
        corrected_content, applied_count = self.apply_corrections_to_text_robust(original_content, final_errors)
        print(f"  - ✅ 成功应用 {applied_count} 处修正")

        # 生成修改日志
        corrections_log = []
        for err in final_errors:
            if err.get('correct', '') == '':
                log_entry = f"{err['text']} → [删除]"
            else:
                log_entry = f"{err['text']} → {err['correct']}"
            err_type = err.get('type', 'llm')
            log_entry += f" ({err_type})"
            corrections_log.append(log_entry)

        # --- 统计和报告生成 ---
        stats.api_calls = len(segments)
        stats.api_success = self.api_success_count
        stats.api_failures = self.api_failure_count
        stats.detected_errors_count = len(final_errors)
        stats.applied_corrections_count = applied_count
        stats.correction_count = applied_count
        stats.corrections_made = corrections_log

        # 纠正前的错误统计
        total_error_chars_before = sum(len(err['text']) for err in final_errors)
        error_examples_before = []
        for err in final_errors[:10]:
            if err.get('correct', '') == '':
                example_text = f"{err['text']} → [删除]"
            else:
                example_text = f"{err['text']} → {err['correct']}"
            if err.get('type') == 'rule_based':
                example_text += " (规则)"
            else:
                example_text += " (LLM)"
            error_examples_before.append(example_text)

        stats.errors_before.semantic_errors = len(final_errors)
        stats.errors_before.semantic_error_chars = total_error_chars_before
        stats.errors_before.semantic_error_examples = error_examples_before

        # 计算纠正前的准确率
        accuracy_before, cer_before, quality_before = self.calculate_ocr_accuracy(
            len(original_content), total_error_chars_before
        )
        stats.errors_before.ocr_accuracy = accuracy_before
        stats.errors_before.ocr_cer = cer_before
        stats.errors_before.ocr_quality = quality_before

        # 重新评估纠正后的文本质量（不是假设100%）
        print("\n🔍 [阶段4/4] 评估纠正后的文本质量...")
        errors_after, error_chars_after = self.evaluate_text_quality(corrected_content)

        # 纠正后的统计
        error_examples_after = []
        for err in errors_after[:10]:
            if err.get('correct', '') == '':
                example_text = f"{err['text']} → [删除]"
            else:
                example_text = f"{err['text']} → {err['correct']}"
            error_examples_after.append(example_text)

        stats.errors_after.semantic_errors = len(errors_after)
        stats.errors_after.semantic_error_chars = error_chars_after
        stats.errors_after.semantic_error_examples = error_examples_after

        # 计算纠正后的准确率
        accuracy_after, cer_after, quality_after = self.calculate_ocr_accuracy(
            len(corrected_content), error_chars_after
        )
        stats.errors_after.ocr_accuracy = accuracy_after
        stats.errors_after.ocr_cer = cer_after
        stats.errors_after.ocr_quality = quality_after

        print(f"\n📊 处理结果:")
        print(f"  - 总字符数: {len(content):,}")
        print(f"  - 检测到的错误: {stats.detected_errors_count}处")
        print(f"  - 应用的纠正: {stats.applied_corrections_count}处")
        print(f"  - API调用次数: {stats.api_calls}")
        print(f"  - 结构保护元素: {len(protector.protected_elements)}个")

        # 显示纠正应用率
        if stats.detected_errors_count > 0:
            application_rate = (stats.applied_corrections_count / stats.detected_errors_count) * 100
            print(f"  - 纠正应用率: {application_rate:.1f}%")

        print(f"\n📈 OCR质量评估:")
        print(f"  - 纠正前: 准确率 {accuracy_before:.2f}%, CER {cer_before:.2f}%, {quality_before}")
        print(f"  - 纠正后: 准确率 {accuracy_after:.2f}%, CER {cer_after:.2f}%, {quality_after}")
        print(f"  - 剩余错误: {len(errors_after)}个")

        stats.correction_time = time.time() - start_time
        stats.processing_time = stats.correction_time

        return corrected_content, corrections_log, stats

    def split_content_optimized(self, content: str, target_length: int = 8000) -> List[str]:
        """优化的内容分割 - 使用更大的分段，考虑占位符"""
        segments = []

        # 按双换行分割（段落）
        paragraphs = content.split('\n\n')
        current_segment = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # 跳过过短的段落
            if para_length < 20:
                continue

            # 如果是占位符段落，单独作为一段
            if para.startswith('__STRUCTURE_PROTECTOR_') and para.endswith('__'):
                if current_segment:
                    segments.append('\n\n'.join(current_segment))
                    current_segment = []
                    current_length = 0
                segments.append(para)
                continue

            # 如果单个段落太长，需要分割
            if para_length > target_length * 1.5:
                # 先保存当前段
                if current_segment:
                    segments.append('\n\n'.join(current_segment))
                    current_segment = []
                    current_length = 0

                # 按句子分割长段落
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_segment = []
                temp_length = 0

                for sent in sentences:
                    if temp_length + len(sent) > target_length and temp_segment:
                        segments.append(' '.join(temp_segment))
                        temp_segment = [sent]
                        temp_length = len(sent)
                    else:
                        temp_segment.append(sent)
                        temp_length += len(sent)

                if temp_segment:
                    segments.append(' '.join(temp_segment))

            # 正常累积段落
            elif current_length + para_length > target_length and current_segment:
                segments.append('\n\n'.join(current_segment))
                current_segment = [para]
                current_length = para_length
            else:
                current_segment.append(para)
                current_length += para_length

        # 添加最后的段落
        if current_segment:
            segments.append('\n\n'.join(current_segment))

        # 合并过短的段（第二轮优化）
        optimized_segments = []
        temp_segment = ""
        min_segment_length = 1500

        for seg in segments:
            seg_length = len(seg.strip())

            # 占位符段落不合并
            if seg.startswith('__STRUCTURE_PROTECTOR_') and seg.endswith('__'):
                if temp_segment:
                    optimized_segments.append(temp_segment)
                    temp_segment = ""
                optimized_segments.append(seg)
                continue

            if seg_length < min_segment_length and temp_segment:
                # 合并短段
                temp_segment += "\n\n" + seg
            elif temp_segment and len(temp_segment) + seg_length < target_length * 1.2:
                # 合并到临时段
                temp_segment += "\n\n" + seg
            else:
                # 保存临时段并开始新段
                if temp_segment:
                    optimized_segments.append(temp_segment)
                temp_segment = seg

        if temp_segment:
            optimized_segments.append(temp_segment)

        # 过滤掉太短的段（但保留占位符）
        optimized_segments = [seg for seg in optimized_segments
                              if len(seg.strip()) > 200 or
                              (seg.startswith('__STRUCTURE_PROTECTOR_') and seg.endswith('__'))]

        # 第三轮优化：如果段落数还是太多，再次合并
        if len(optimized_segments) > 15:
            final_segments = []
            temp_segment = ""
            max_segment_length = target_length * 1.5

            for seg in optimized_segments:
                # 占位符段落不合并
                if seg.startswith('__STRUCTURE_PROTECTOR_') and seg.endswith('__'):
                    if temp_segment:
                        final_segments.append(temp_segment)
                        temp_segment = ""
                    final_segments.append(seg)
                    continue

                if temp_segment and len(temp_segment) + len(seg) < max_segment_length:
                    temp_segment += "\n\n" + seg
                else:
                    if temp_segment:
                        final_segments.append(temp_segment)
                    temp_segment = seg

            if temp_segment:
                final_segments.append(temp_segment)

            return final_segments

        return optimized_segments


def pdf_to_markdown_with_correction(pdf_path: str, output_md: str, corrected_md: str,
                                    corrector: OptimizedChemicalCorrector) -> ConversionCorrectionStats:
    """转换PDF到Markdown并进行纠错"""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    tmp_md_dir = os.path.abspath("mineru_tmp_md")
    tmp_img_dir = os.path.abspath("mineru_tmp_img")
    os.makedirs(tmp_md_dir, exist_ok=True)
    os.makedirs(tmp_img_dir, exist_ok=True)

    stats = ConversionCorrectionStats(pdf_name=pdf_name)
    overall_start_time = time.time()

    try:
        print(f"\n[步骤1] PDF转换")
        conversion_start = time.time()

        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)
        ds = PymuDocDataset(pdf_bytes)

        ocr_needed = (ds.classify() == SupportedPdfParseMethod.OCR)
        stats.ocr_mode = ocr_needed

        if hasattr(ds, 'page_num'):
            stats.total_pages = ds.page_num
        elif hasattr(ds, '__len__'):
            stats.total_pages = len(ds)

        md_writer = FileBasedDataWriter(tmp_md_dir)
        img_writer = FileBasedDataWriter(tmp_img_dir)

        kwargs = dict(
            ocr=ocr_needed,
            formula_enable=True,
            table_enable=True,
            lang='en'
        )
        pipeline = ds.apply(doc_analyze, **kwargs)

        if ocr_needed:
            pipeline = pipeline.pipe_ocr_mode(img_writer)
        else:
            pipeline = pipeline.pipe_txt_mode(img_writer)

        pipeline.dump_md(md_writer, f"{pdf_name}.md", tmp_img_dir)

        stats.conversion_time = time.time() - conversion_start
        print(f"✅ 转换完成 ({stats.conversion_time:.1f}秒)")

        md_file = os.path.join(tmp_md_dir, f"{pdf_name}.md")
        if not os.path.exists(md_file):
            raise FileNotFoundError(f"未找到Markdown文件：{md_file}")

        with open(md_file, encoding='utf-8') as f:
            content = f.read()

        # 保留原始内容，包括图片
        lines = content.splitlines()
        content_for_stats = '\n'.join([line for line in lines if not line.startswith('![')])

        stats.total_characters = len(content_for_stats)

        # 保存原始Markdown
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[步骤2] 智能纠错 (增强版v6)")

        # 执行纠错
        corrected_content, corrections_log, correction_stats = corrector.correct_document(content)

        # 更新统计信息
        stats.errors_before = correction_stats.errors_before
        stats.errors_after = correction_stats.errors_after
        stats.correction_count = correction_stats.correction_count
        stats.corrections_made = corrections_log
        stats.api_calls = correction_stats.api_calls
        stats.api_success = correction_stats.api_success
        stats.api_failures = correction_stats.api_failures
        stats.correction_time = correction_stats.correction_time
        stats.detected_errors_count = correction_stats.detected_errors_count
        stats.applied_corrections_count = correction_stats.applied_corrections_count

        # 保存纠正后的Markdown
        with open(corrected_md, 'w', encoding='utf-8') as f:
            f.write(corrected_content)

        stats.processing_time = time.time() - overall_start_time

        # 生成报告
        report_path = os.path.join(correction_reports_folder, f"{pdf_name}_report.json")
        report = {
            'file': pdf_name,
            'pages': stats.total_pages,
            'characters': stats.total_characters,
            'processing_mode': 'OCR' if stats.ocr_mode else 'Text',
            'ocr_quality': {
                'before': {
                    'accuracy': round(stats.errors_before.ocr_accuracy, 2),
                    'cer': round(stats.errors_before.ocr_cer, 2),
                    'quality': stats.errors_before.ocr_quality,
                    'errors': stats.errors_before.semantic_errors,
                    'error_examples': stats.errors_before.semantic_error_examples[:10]
                },
                'after': {
                    'accuracy': round(stats.errors_after.ocr_accuracy, 2),
                    'cer': round(stats.errors_after.ocr_cer, 2),
                    'quality': stats.errors_after.ocr_quality,
                    'errors': stats.errors_after.semantic_errors,
                    'error_examples': stats.errors_after.semantic_error_examples[:10]
                }
            },
            'corrections': {
                'detected_errors': stats.detected_errors_count,
                'applied_corrections': stats.applied_corrections_count,
                'actual_changes': stats.correction_count,
                'application_rate': round((stats.applied_corrections_count / stats.detected_errors_count * 100),
                                          1) if stats.detected_errors_count > 0 else 0,
                'api_calls': stats.api_calls,
                'api_success': stats.api_success,
                'api_failures': stats.api_failures,
                'samples': stats.corrections_made[:20]
            },
            'performance': {
                'total_time': round(stats.processing_time, 1),
                'conversion_time': round(stats.conversion_time, 1),
                'correction_time': round(stats.correction_time, 1)
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        if corrections_log:
            print(f"\n纠错示例（前10个）:")
            for i, correction in enumerate(corrections_log[:10], 1):
                print(f"  {i}. {correction}")

    except Exception as e:
        print(f"\n[错误详情]:")
        traceback.print_exc()
        stats.processing_time = time.time() - overall_start_time
        raise

    return stats


def create_before_after_scatter_plot(all_stats: List[ConversionCorrectionStats], save_path: str):
    """
    创建符合学术期刊标准的逐篇文献改进效果散点图（修复版）
    """
    # 设置学术风格
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    # 准备数据
    cer_before = [s.errors_before.ocr_cer for s in all_stats]
    cer_after = [s.errors_after.ocr_cer for s in all_stats]
    pdf_names = [s.pdf_name for s in all_stats]

    # 调试输出
    print(f"\n[调试] CER数据:")
    print(f"  纠正前: {cer_before}")
    print(f"  纠正后: {cer_after}")

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 7))  # 稍微增大尺寸

    # 设置白色背景
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 数据范围（根据实际数据动态调整）
    all_values = cer_before + cer_after
    data_min = 0  # CER从0开始
    data_max = max(all_values) if all_values else 10

    # 为了美观，确保范围合理
    if data_max < 1:
        data_max = 1
    elif data_max < 5:
        data_max = 5
    elif data_max < 10:
        data_max = 10
    else:
        data_max = int(data_max * 1.2)  # 留20%边距

    # 绘制y=x对角线
    ax.plot([data_min, data_max], [data_min, data_max],
            'k--', linewidth=1, alpha=0.5, zorder=1,
            label='No improvement (y = x)')

    # 根据样本数量选择marker大小
    n_samples = len(all_stats)
    if n_samples == 1:
        marker_size = 120
        edge_width = 2
    elif n_samples < 10:
        marker_size = 80
        edge_width = 1.5
    else:
        marker_size = 50
        edge_width = 1

    # 绘制散点
    scatter = ax.scatter(cer_before, cer_after,
                         s=marker_size,
                         c='#1f77b4',
                         alpha=0.7,
                         edgecolors='black',
                         linewidth=edge_width,
                         zorder=3)

    # 设置轴标签
    ax.set_xlabel('CER Before Correction (%)', fontsize=12)
    ax.set_ylabel('CER After Correction (%)', fontsize=12)

    # 设置坐标轴范围
    ax.set_xlim(data_min - 0.5, data_max + 0.5)
    ax.set_ylim(data_min - 0.5, data_max + 0.5)

    # 添加网格
    ax.grid(True, which='major', linestyle='-', linewidth=0.5,
            color='gray', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3,
            color='gray', alpha=0.2)

    # 设置刻度
    from matplotlib.ticker import MultipleLocator

    # 根据数据范围设置合适的主刻度间隔
    if data_max <= 2:
        major_interval = 0.5
        minor_interval = 0.1
    elif data_max <= 5:
        major_interval = 1
        minor_interval = 0.5
    elif data_max <= 10:
        major_interval = 2
        minor_interval = 0.5
    elif data_max <= 20:
        major_interval = 5
        minor_interval = 1
    else:
        major_interval = 10
        minor_interval = 2

    ax.xaxis.set_major_locator(MultipleLocator(major_interval))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_interval))
    ax.yaxis.set_major_locator(MultipleLocator(major_interval))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_interval))

    # 计算统计信息（确保计算正确）
    improved_count = sum(1 for b, a in zip(cer_before, cer_after) if a < b)
    improvement_rate = (improved_count / n_samples) * 100 if n_samples > 0 else 0

    # 使用实际的数值计算
    if cer_before and cer_after:
        mean_before = np.mean(cer_before)
        mean_after = np.mean(cer_after)
        avg_improvement = mean_before - mean_after
        relative_improvement = (avg_improvement / mean_before) * 100 if mean_before > 0 else 0
    else:
        avg_improvement = 0
        relative_improvement = 0

    # 添加统计信息文本框
    textstr = f'n = {n_samples}\n'
    textstr += f'Improved: {improved_count}/{n_samples} ({improvement_rate:.0f}%)\n'
    textstr += f'Mean reduction: {avg_improvement:.2f}%\n'
    textstr += f'Relative improvement: {relative_improvement:.1f}%'

    props = dict(boxstyle='round,pad=0.5',
                 facecolor='white',
                 edgecolor='gray',
                 linewidth=0.8,
                 alpha=0.9)

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=props)

    # 添加改进区域着色（对角线下方）
    # 创建一个多边形表示改进区域
    improvement_x = [data_min, data_max, data_max, data_min]
    improvement_y = [data_min, data_min, data_max, data_min]
    ax.fill(improvement_x, improvement_y,
            color='green', alpha=0.05, zorder=0)

    # 为显著改进的点添加标签
    for i, (x, y, name) in enumerate(zip(cer_before, cer_after, pdf_names)):
        if x - y > 1.0:  # CER降低超过1%
            # 对长文件名进行智能截断
            if len(name) > 30:
                # 保留开头和结尾
                label = name[:15] + '...' + name[-12:]
            else:
                label = name

            # 调整标签位置，避免被截断
            # 根据点的位置决定标签方向
            if x > data_max * 0.7:  # 如果点靠右
                ha = 'right'
                xytext = (-10, 10)
            else:
                ha = 'left'
                xytext = (10, 10)

            ax.annotate(label, (x, y),
                        xytext=xytext,
                        textcoords='offset points',
                        fontsize=8,
                        ha=ha,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='yellow',
                                  alpha=0.5,
                                  edgecolor='none'),
                        arrowprops=dict(arrowstyle='->',
                                        connectionstyle='arc3,rad=0.3',
                                        alpha=0.5))

    # 添加图例
    ax.legend(loc='lower right', frameon=True,
              fancybox=False, shadow=False,
              edgecolor='gray', framealpha=0.9)

    # 设置等比例轴（确保x和y轴比例相同）
    ax.set_aspect('equal', adjustable='box')

    # 去除顶部和右侧边框（学术风格）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 调整布局，留出更多空间
    plt.tight_layout(pad=1.5)

    # 保存高分辨率图片
    plt.savefig(save_path,
                dpi=600,  # 高分辨率用于发表
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')

    # 同时保存矢量格式
    vector_path = save_path.replace('.png', '.pdf')
    plt.savefig(vector_path,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='pdf')

    plt.close()

    print(f"\n📊 散点图已保存至:")
    print(f"   - PNG格式: {save_path} (600 DPI)")
    print(f"   - PDF格式: {vector_path} (矢量图)")
    print(f"\n📈 统计摘要:")
    print(f"   - 平均CER降低: {avg_improvement:.2f}%")
    print(f"   - 相对改进: {relative_improvement:.1f}%")

def perform_paired_t_test(all_stats: List[ConversionCorrectionStats]) -> Dict:
    """
    执行配对t检验，验证改进的统计显著性
    """
    # 准备数据
    cer_before = np.array([s.errors_before.ocr_cer for s in all_stats])
    cer_after = np.array([s.errors_after.ocr_cer for s in all_stats])

    # 检查样本数量
    sample_size = len(all_stats)

    if sample_size < 2:
        # 样本太少，无法进行t检验
        diff = cer_before - cer_after
        result = {
            't_statistic': None,
            'p_value': None,
            'cohen_d': None,
            'mean_improvement': float(np.mean(diff)) if sample_size > 0 else 0.0,
            'std_improvement': None,
            'confidence_interval': [None, None],
            'is_significant': False,
            'significance_level': "n.a.",
            'sample_size': sample_size
        }

        print("\n" + "=" * 60)
        print("📊 统计分析结果")
        print("=" * 60)
        print(f"样本数量: {sample_size} (需要至少2个样本进行配对t检验)")
        if sample_size == 1:
            print(f"单个样本的CER改进: {result['mean_improvement']:.3f}%")
            print("\n注意: 配对t检验需要至少2个样本。请处理更多PDF文件以获得统计学结论。")

        return result

    # 执行配对t检验
    t_statistic, p_value = stats.ttest_rel(cer_before, cer_after)

    # 计算效应量 (Cohen's d)
    diff = cer_before - cer_after
    cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.inf

    # 计算置信区间
    confidence_level = 0.95
    degrees_of_freedom = len(diff) - 1
    confidence_interval = stats.t.interval(confidence_level, degrees_of_freedom,
                                           loc=np.mean(diff),
                                           scale=stats.sem(diff))

    # 判断显著性
    is_significant = bool(p_value < 0.05)  # 转换为Python原生bool
    significance_level = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."

    result = {
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'cohen_d': float(cohen_d) if not np.isinf(cohen_d) else None,
        'mean_improvement': float(np.mean(diff)),
        'std_improvement': float(np.std(diff, ddof=1)),
        'confidence_interval': [float(confidence_interval[0]), float(confidence_interval[1])],
        'is_significant': is_significant,
        'significance_level': significance_level,
        'sample_size': sample_size
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("📊 配对t检验结果 (Paired t-test Results)")
    print("=" * 60)
    print(f"样本数量: {result['sample_size']}")
    print(f"平均CER改进: {result['mean_improvement']:.3f}% ± {result['std_improvement']:.3f}%")
    print(f"t统计量: {result['t_statistic']:.3f}")
    print(f"p值: {result['p_value']:.6f} {result['significance_level']}")
    if result['cohen_d'] is not None:
        print(f"Cohen's d效应量: {result['cohen_d']:.3f}")
    print(f"95%置信区间: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
    print(f"\n统计显著性: {'是 ✓' if is_significant else '否 ✗'}")

    if is_significant:
        print(f"结论: 纠错系统带来的性能提升在统计学上是显著的 (p < 0.05)")

    return result


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PDF化学文献转换与语义纠错系统（增强版v6）')
    parser.add_argument('--input-folder', type=str, default=input_pdf_folder)
    parser.add_argument('--output-folder', type=str, default=output_md_folder)
    parser.add_argument('--corrected-folder', type=str, default=corrected_md_folder)
    parser.add_argument('--single-file', type=str, help='只处理单个PDF文件')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')

    args = parser.parse_args()

    # 创建必要的文件夹
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.corrected_folder, exist_ok=True)
    os.makedirs(correction_reports_folder, exist_ok=True)
    os.makedirs(visualization_folder, exist_ok=True)

    # 初始化纠错器
    corrector = OptimizedChemicalCorrector()
    if args.debug:
        corrector.debug_mode = True

    # 获取待处理的PDF文件
    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"[!] 文件不存在: {args.single_file}")
            return
        pdf_files = [args.single_file]
    else:
        pdf_files = glob.glob(os.path.join(args.input_folder, "*.pdf"))

    if not pdf_files:
        print(f"[!] 未找到PDF文件")
        return

    print(f"\n📁 找到 {len(pdf_files)} 个PDF文件")

    all_stats = []

    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_md_path = os.path.join(args.output_folder, f"{pdf_name}.md")
        corrected_md_path = os.path.join(args.corrected_folder, f"{pdf_name}_corrected.md")

        print(f"\n{'=' * 60}")
        print(f"📄 处理文件 [{i}/{len(pdf_files)}]: {pdf_name}.pdf")
        print(f"{'=' * 60}")

        try:
            stats = pdf_to_markdown_with_correction(
                pdf_path, output_md_path, corrected_md_path, corrector
            )
            all_stats.append(stats)

        except KeyboardInterrupt:
            print("\n\n[!] 用户中断处理")
            break
        except Exception as e:
            print(f"\n[✗] 处理 {pdf_name}.pdf 时出错: {str(e)}")
            continue

    # 打印汇总
    if all_stats:
        print("\n" + "=" * 80)
        print("📊 处理汇总")
        print("=" * 80)

        total_pdfs = len(all_stats)
        total_chars = sum(s.total_characters for s in all_stats)
        total_corrections = sum(s.correction_count for s in all_stats)
        total_detected = sum(s.detected_errors_count for s in all_stats)
        total_applied = sum(s.applied_corrections_count for s in all_stats)
        total_api_calls = sum(s.api_calls for s in all_stats)
        total_api_success = sum(s.api_success for s in all_stats)

        print(f"处理文件数: {total_pdfs}")
        print(f"总字符数: {total_chars:,}")
        print(f"检测到的错误: {total_detected}")
        print(f"应用的纠正: {total_applied}")
        print(f"实际修改: {total_corrections}")
        print(f"API调用: {total_api_success}/{total_api_calls} 成功")

        if total_detected > 0:
            overall_application_rate = (total_applied / total_detected) * 100
            print(f"整体纠正应用率: {overall_application_rate:.1f}%")
        else:
            overall_application_rate = 0.0

        # OCR质量统计
        print("\n" + "-" * 60)
        print("📈 整体质量分析")
        print("-" * 60)

        all_accuracy_before = np.mean([s.errors_before.ocr_accuracy for s in all_stats])
        all_accuracy_after = np.mean([s.errors_after.ocr_accuracy for s in all_stats])
        all_cer_before = np.mean([s.errors_before.ocr_cer for s in all_stats])
        all_cer_after = np.mean([s.errors_after.ocr_cer for s in all_stats])

        print(f"\n平均值:")
        print(f"  纠错前: 准确率 {all_accuracy_before:.2f}%, CER {all_cer_before:.2f}%")
        print(f"  纠错后: 准确率 {all_accuracy_after:.2f}%, CER {all_cer_after:.2f}%")
        print(f"  改进: 准确率提升 {all_accuracy_after - all_accuracy_before:.2f}%")

        # 创建散点图
        scatter_plot_path = os.path.join(visualization_folder, 'before_after_scatter_plot.png')
        create_before_after_scatter_plot(all_stats, scatter_plot_path)

        # 执行配对t检验
        t_test_results = perform_paired_t_test(all_stats)

        # 保存综合报告（包含统计检验结果）
        summary_report = {
            'summary': {
                'total_files': total_pdfs,
                'total_characters': total_chars,
                'detected_errors': total_detected,
                'applied_corrections': total_applied,
                'actual_changes': total_corrections,
                'overall_application_rate': round(overall_application_rate, 1),
                'processing_time': round(sum(s.processing_time for s in all_stats), 1),
                'api_calls': total_api_calls,
                'api_success': total_api_success
            },
            'overall_average': {
                'accuracy_before': round(all_accuracy_before, 2),
                'accuracy_after': round(all_accuracy_after, 2),
                'cer_before': round(all_cer_before, 2),
                'cer_after': round(all_cer_after, 2),
                'improvement': round(all_accuracy_after - all_accuracy_before, 2)
            },
            'statistical_analysis': {
                't_statistic': round(t_test_results['t_statistic'], 3) if t_test_results[
                                                                              't_statistic'] is not None else None,
                'p_value': t_test_results['p_value'] if t_test_results['p_value'] is not None else None,
                'cohen_d': round(t_test_results['cohen_d'], 3) if t_test_results['cohen_d'] is not None else None,
                'mean_improvement': round(t_test_results['mean_improvement'], 3) if t_test_results[
                                                                                        'mean_improvement'] is not None else None,
                'confidence_interval': [
                    round(t_test_results['confidence_interval'][0], 3) if t_test_results['confidence_interval'][
                                                                              0] is not None else None,
                    round(t_test_results['confidence_interval'][1], 3) if t_test_results['confidence_interval'][
                                                                              1] is not None else None
                ],
                'is_significant': bool(t_test_results['is_significant']),  # 确保是Python原生bool
                'significance_level': t_test_results['significance_level'],
                'sample_size': t_test_results['sample_size']
            },
            'files': [
                {
                    'name': s.pdf_name,
                    'detected_errors': s.detected_errors_count,
                    'applied_corrections': s.applied_corrections_count,
                    'actual_changes': s.correction_count,
                    'accuracy_before': round(s.errors_before.ocr_accuracy, 2),
                    'accuracy_after': round(s.errors_after.ocr_accuracy, 2),
                    'cer_before': round(s.errors_before.ocr_cer, 2),
                    'cer_after': round(s.errors_after.ocr_cer, 2),
                    'time': round(s.processing_time, 1)
                }
                for s in all_stats
            ]
        }

        with open(os.path.join(correction_reports_folder, 'summary_report.json'), 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
            print(f"\n📄 报告已保存")
            print(f"📊 可视化图表已保存至: {visualization_folder}")


if __name__ == "__main__":
    main()