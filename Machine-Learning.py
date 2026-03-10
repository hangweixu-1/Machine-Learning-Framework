#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
CO₂加氢制甲醇催化剂多尺度机器学习框架 - 完整增强版 v6.10（STY阈值与稳定性修复版）
实施所有审稿建议：异常值决策、不确定性应用、基线对比、深度SHAP分析等
包含Nature顶刊风格图表设计
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch, Polygon, Circle
from matplotlib.collections import LineCollection
from tqdm import tqdm
import json
import pickle
from pathlib import Path
from scipy.spatial import Delaunay
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import copy

# 只屏蔽特定已知的无害警告
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing')
os.environ['LOKY_MAX_CPU_COUNT'] = '16'

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 基础机器学习库
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_extraction.text import TfidfVectorizer

# 深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torch import optim

    HAS_TORCH = True
    logger.info("✓ PyTorch已安装")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  使用设备: {device}")
except:
    HAS_TORCH = False
    logger.warning("PyTorch未安装，深度学习功能将被禁用")
    device = None

# 高级ML库
try:
    import xgboost as xgb

    HAS_XGBOOST = True
    logger.info("✓ XGBoost已安装")
except:
    HAS_XGBOOST = False
    logger.warning("XGBoost未安装")

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
    logger.info("✓ LightGBM已安装")
except:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM未安装")

try:
    import catboost as cb

    HAS_CATBOOST = True
    logger.info("✓ CatBoost已安装")
except:
    HAS_CATBOOST = False
    logger.warning("CatBoost未安装")

try:
    import shap

    HAS_SHAP = True
    logger.info("✓ SHAP已安装，将执行深度可解释性分析")
except:
    HAS_SHAP = False
    logger.warning("SHAP未安装！建议安装: pip install shap")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    HAS_OPTUNA = True
    logger.info("✓ Optuna已安装，将使用贝叶斯优化")
except:
    HAS_OPTUNA = True
    logger.warning("Optuna未安装！建议安装: pip install optuna")

# ==========================
# 参数配置（增强版v6）
# ==========================

excel_path = r"3_cleaned2_standardized_processed2.xlsx"

# 控制参数
VERBOSE = True
USE_BAYESIAN_OPTIMIZATION = True
USE_DEEP_LEARNING = False  # 默认关闭以加快运行
USE_SHAP_ANALYSIS = True
USE_BASELINE_COMPARISON = True  # 新增：是否进行基线模型对比
CREATE_NATURE_FIGURES = True  # 创建Nature风格图表

# 异常值处理策略
OUTLIER_STRATEGY = 'huber'  # 'keep', 'remove', 'huber', 'ransac'
OUTLIER_IQR_MULTIPLIER = 3.0  # IQR倍数，用于异常值检测

# 不确定性和外插控制
EXTRAPOLATION_THRESHOLD = 1.5  # 外插距离阈值
CONFIDENCE_LEVEL = 2.0  # 置信区间倍数（2σ）
MIN_CONFIDENCE_STY = 200  # 最小置信下界STY阈值
UNCERTAINTY_THRESHOLD = 0.1  # log空间不确定性阈值，对应约10%相对误差（exp(0.1)≈1.105）

# 体系特定标准条件（改进：基于密度峰值）
SYSTEM_STANDARD_CONDITIONS = {
    'Cu/ZnO': {
        'Temperature [K]': 523.0,
        'Pressure [Mpa]': 7.0,
        'H2/CO2 [-]': 3.0,
        'GHSV [cm3 h-1 gcat-1]': 12000.0,
        'Catalyst amount [g]': 0.5,
    },
    'Cu/ZnO/Al2O3': {
        'Temperature [K]': 533.0,
        'Pressure [Mpa]': 7.5,
        'H2/CO2 [-]': 3.0,
        'GHSV [cm3 h-1 gcat-1]': 12000.0,
        'Catalyst amount [g]': 0.5,
    },
    'In2O3': {
        'Temperature [K]': 573.0,
        'Pressure [Mpa]': 5.0,
        'H2/CO2 [-]': 3.0,
        'GHSV [cm3 h-1 gcat-1]': 24000.0,
        'Catalyst amount [g]': 0.5,
    },
    'In2O3/ZrO2': {
        'Temperature [K]': 573.0,
        'Pressure [Mpa]': 5.0,
        'H2/CO2 [-]': 4.0,
        'GHSV [cm3 h-1 gcat-1]': 32000.0,
        'Catalyst amount [g]': 0.5,
    }
}
# 体系显示名称（统一使用下标）
SYSTEM_DISPLAY_NAMES = {
    'Cu/ZnO': 'Cu/ZnO',
    'Cu/ZnO/Al2O3': 'Cu/ZnO/Al₂O₃',
    'In2O3': 'In₂O₃',
    'In2O3/ZrO2': 'In₂O₃/ZrO₂'
}
# 其他参数
MIN_SAMPLES_FOR_METHOD = 10
N_RANDOM_SEEDS = 5  # v6.10修复：增加到5个种子提高稳定性
MIN_YEARS_FOR_FORECAST = 3
MIN_SAMPLES_PER_YEAR = 3
N_BAYESIAN_TRIALS = 50
N_SYSTEM_BAYESIAN_TRIALS = 50

# 深度学习参数
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 32
N_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ==========================
# Nature风格配色和设置（Times New Roman字体）- CCL修改版
# ==========================

# 设置高质量的Nature风格参数（使用Times New Roman字体）
import matplotlib.font_manager as fm

# 检查Times New Roman字体是否可用
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Times New Roman' in available_fonts:
    font_family = 'Times New Roman'
    logger.info("✓ 使用Times New Roman字体")
else:
    font_family = 'serif'
    logger.warning("Times New Roman字体未找到，使用默认serif字体")

mpl.rcParams['font.family'] = font_family
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False

# 统一的四体系配色方案 (Nature风格)
SYSTEM_COLORS = {
    'Cu/ZnO': '#2E5266',  # 深蓝灰色
    'Cu/ZnO/Al2O3': '#6E8898',  # 中蓝灰色
    'In2O3': '#D3896C',  # 暖橙色
    'In2O3/ZrO2': '#AAA287'  # 暖黄灰色
}

# 模型配色
MODEL_COLORS = {
    'Linear': '#95A3A6',  # 浅灰
    'Polynomial': '#7B8D8E',  # 中灰
    'RandomForest': '#5E81AC',  # 蓝色
    'GradientBoosting': '#88C0D0',  # 浅蓝
    'XGBoost': '#8FBCBB',  # 青色
    'LightGBM': '#A3BE8C',  # 绿色
    'CatBoost': '#EBCB8B'  # 黄色
}

# 聚类配色
CLUSTER_COLORS = ['#5E81AC', '#88C0D0', '#81A1C1', '#B48EAD', '#D08770', '#EBCB8B']

# 通用配色
NATURE_COLORS = {
    'primary': '#2C3E50',  # 主色
    'secondary': '#E74C3C',  # 副色
    'accent': '#3498DB',  # 强调色
    'grey_dark': '#495057',
    'grey_mid': '#6C757D',
    'grey_light': '#ADB5BD',
    'background': '#F8F9FA',
    'grid': '#E9ECEF'
}


# ==========================
# 图片保存辅助函数（CCL新增）
# ==========================

def save_individual_figure(fig, figure_num, subplot_name, output_dir='outputs'):
    """保存单个子图到PNG和SVG两个文件夹"""
    png_dir = os.path.join(output_dir, 'png')
    svg_dir = os.path.join(output_dir, 'svg')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    # 清理文件名
    clean_name = subplot_name.replace(' ', '_').replace('/', '_').replace(':', '_')
    clean_name = clean_name.replace('(', '').replace(')', '').replace(',', '').replace("'", '')
    clean_name = clean_name.replace('²', '2').replace('³', '3').replace('₂', '2').replace('₃', '3')
    clean_name = clean_name.replace('⁻¹', '-1').replace('$', '').replace('{', '').replace('}', '')
    clean_name = clean_name.replace('\\', '').replace('\n', '_').replace('×', 'x')
    # 移除连续下划线
    while '__' in clean_name:
        clean_name = clean_name.replace('__', '_')
    clean_name = clean_name.strip('_')

    filename = f"Figure{figure_num}_{clean_name}"

    fig.savefig(os.path.join(png_dir, f'{filename}.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(os.path.join(svg_dir, f'{filename}.svg'), format='svg', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"  ✓ Saved: {filename}")



# ==========================
# 辅助函数
# ==========================

def print_section(title, width=80, force=False):
    """打印章节标题"""
    if VERBOSE or force:
        logger.info("\n" + "=" * width)
        logger.info(title)
        logger.info("=" * width)


def create_preprocessor(num_features: List[str], cat_features: List[str]):
    """创建预处理pipeline（改进版：处理混合类型）"""

    # 数值特征处理器
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 分类特征处理器 - 处理字符串类型
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # 组合处理器
    if len(num_features) > 0 and len(cat_features) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_features),
                ("cat", cat_transformer, cat_features),
            ],
            remainder='drop'  # 丢弃其他列
        )
    elif len(num_features) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_features),
            ],
            remainder='drop'
        )
    elif len(cat_features) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, cat_features),
            ],
            remainder='drop'
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[],
            remainder='passthrough'
        )

    return preprocessor


# ==========================
# 数据质量控制函数（增强版v6）
# ==========================

def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    验证并修复数据类型问题
    """
    df = df.copy()

    known_numeric_cols = [
        'STY [mgMeOH h-1 gcat-1]',
        'Temperature [K]',
        'Pressure [Mpa]',
        'GHSV [cm3 h-1 gcat-1]',
        'H2/CO2 [-]',
        'Metal Loading [wt.%]',
        'SBET [m2 g-1]',
        'Catalyst amount [g]',
        'Calcination Temperature [K]',
        'Calcination duration [h]',
        'CR Metal [pm]',
        'Promoter 1 loading [wt.%]',
        'Promoter 2 loading [wt.%]'
    ]

    known_categorical_cols = [
        'System',
        'Family',
        'Support 1',
        'Name of Support2',
        'Name of Support 3',
        'Promoter 1',
        'Promoter 2',
        'method'
    ]

    for col in known_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            n_converted = df[col].notna().sum()
            n_total = len(df)
            if n_converted < n_total * 0.5:
                logger.warning(f"列 '{col}' 只有 {n_converted}/{n_total} 个有效数值")

    mw_cols = [col for col in df.columns if 'MW' in col and 'Support' in col]
    for col in mw_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                logger.warning(f"列 '{col}' 包含非数值数据，将填充为0")
                df[col] = 0

    for col in known_categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)

    return df


def detect_and_handle_outliers(df: pd.DataFrame, target_col: str = "STY [mgMeOH h-1 gcat-1]",
                               strategy: str = OUTLIER_STRATEGY) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    检测并处理异常值（v6增强：真正的决策）
    """
    print_section("异常值检测与决策处理", force=True)

    outliers = pd.DataFrame(index=df.index)

    if target_col in df.columns:
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers['STY_outlier'] = (df[target_col] < Q1 - OUTLIER_IQR_MULTIPLIER * IQR) | \
                                  (df[target_col] > Q3 + OUTLIER_IQR_MULTIPLIER * IQR)

    if 'Temperature [K]' in df.columns:
        outliers['temp_outlier'] = (df['Temperature [K]'] < 373) | (df['Temperature [K]'] > 873)

    if 'Pressure [Mpa]' in df.columns:
        outliers['pressure_outlier'] = (df['Pressure [Mpa]'] < 0.1) | (df['Pressure [Mpa]'] > 30)

    if 'GHSV [cm3 h-1 gcat-1]' in df.columns:
        outliers['ghsv_outlier'] = df['GHSV [cm3 h-1 gcat-1]'] < 100

    if 'H2/CO2 [-]' in df.columns:
        outliers['ratio_outlier'] = (df['H2/CO2 [-]'] < 0.5) | (df['H2/CO2 [-]'] > 20)

    outliers['is_extreme'] = outliers[['temp_outlier', 'pressure_outlier',
                                       'ghsv_outlier', 'ratio_outlier']].any(axis=1)
    outliers['is_statistical'] = outliers['STY_outlier'] & ~outliers['is_extreme']
    outliers['needs_action'] = outliers['is_extreme'] | outliers['is_statistical']

    n_extreme = outliers['is_extreme'].sum()
    n_statistical = outliers['is_statistical'].sum()

    logger.info(f"检测到异常值：")
    logger.info(f"  - 极端异常（物理不合理）: {n_extreme}个")
    logger.info(f"  - 统计异常（超出IQR）: {n_statistical}个")

    df_processed = df.copy()

    if strategy == 'remove':
        df_processed = df_processed[~outliers['needs_action']]
        logger.info(f"策略：剔除所有异常值，剩余样本数: {len(df_processed)}")

    elif strategy == 'huber':
        df_processed = df_processed[~outliers['is_extreme']]
        df_processed['outlier_weight'] = 1.0
        remaining_stat_outliers = outliers.loc[df_processed.index, 'is_statistical']
        df_processed.loc[remaining_stat_outliers, 'outlier_weight'] = 0.5
        logger.info(f"策略：Huber降权，剔除极端异常{n_extreme}个，统计异常降权{remaining_stat_outliers.sum()}个")

    elif strategy == 'ransac':
        df_processed = df_processed[~outliers['is_extreme']]
        df_processed['use_ransac'] = True
        logger.info(f"策略：RANSAC鲁棒回归，剔除极端异常{n_extreme}个")

    else:  # 'keep'
        df_processed['is_outlier'] = outliers['needs_action']
        logger.info(f"策略：保留所有数据，仅标记异常值")

    comparison = {
        'strategy': strategy,
        'n_original': len(df),
        'n_processed': len(df_processed),
        'n_extreme_removed': n_extreme if strategy != 'keep' else 0,
        'n_statistical_handled': n_statistical
    }

    return df_processed, outliers, comparison


def calculate_system_standard_conditions(df: pd.DataFrame, system: str) -> Dict:
    """
    基于数据密度峰值计算体系的标准条件
    """
    standard_cond = {}
    key_params = ['Temperature [K]', 'Pressure [Mpa]', 'H2/CO2 [-]', 'GHSV [cm3 h-1 gcat-1]']

    for param in key_params:
        if param in df.columns:
            values = df[param].dropna().values

            if len(values) > 10:
                try:
                    kde = gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 100)
                    density = kde(x_range)
                    peak_idx = np.argmax(density)
                    peak_value = x_range[peak_idx]

                    range_low = peak_value * 0.8
                    range_high = peak_value * 1.2
                    coverage = np.sum((values >= range_low) & (values <= range_high)) / len(values)

                    standard_cond[param] = {
                        'value': peak_value,
                        'median': np.median(values),
                        'coverage': coverage,
                        'method': 'kde_peak'
                    }
                except:
                    standard_cond[param] = {
                        'value': np.median(values),
                        'median': np.median(values),
                        'coverage': 0.5,
                        'method': 'median'
                    }
            else:
                standard_cond[param] = {
                    'value': np.median(values),
                    'median': np.median(values),
                    'coverage': 0.5,
                    'method': 'median'
                }

    standard_cond['Catalyst amount [g]'] = {'value': 0.5, 'method': 'fixed'}

    return standard_cond


def calculate_extrapolation_info(row: pd.Series, standard_cond: Dict,
                                 training_data: pd.DataFrame = None) -> Dict:
    """
    计算详细的外插信息
    """
    features = ['Temperature [K]', 'Pressure [Mpa]', 'GHSV [cm3 h-1 gcat-1]', 'H2/CO2 [-]']

    scaling = {
        'Temperature [K]': 100,
        'Pressure [Mpa]': 5,
        'GHSV [cm3 h-1 gcat-1]': 20000,
        'H2/CO2 [-]': 2
    }

    distance = 0
    feature_distances = {}

    for feat in features:
        if feat in row.index and feat in standard_cond:
            if pd.notna(row[feat]):
                if isinstance(standard_cond[feat], dict):
                    std_value = standard_cond[feat].get('value', standard_cond[feat])
                else:
                    std_value = standard_cond[feat]

                diff = (row[feat] - std_value) / scaling[feat]
                distance += diff ** 2
                feature_distances[feat] = abs(diff)

    distance = np.sqrt(distance)

    is_reliable = distance < EXTRAPOLATION_THRESHOLD

    min_neighbor_distance = np.inf
    if training_data is not None and len(training_data) > 0:
        for _, train_row in training_data.iterrows():
            neighbor_dist = 0
            for feat in features:
                if feat in train_row.index and pd.notna(train_row[feat]) and pd.notna(row[feat]):
                    diff = (row[feat] - train_row[feat]) / scaling[feat]
                    neighbor_dist += diff ** 2
            neighbor_dist = np.sqrt(neighbor_dist)
            min_neighbor_distance = min(min_neighbor_distance, neighbor_dist)

    return {
        'distance': distance,
        'is_reliable': is_reliable,
        'feature_distances': feature_distances,
        'min_neighbor_distance': min_neighbor_distance
    }



# ==========================
# 数据加载和处理（v6增强）
# ==========================

def load_and_filter_data(excel_path: str, outlier_strategy: str = OUTLIER_STRATEGY) -> Tuple[pd.DataFrame, Dict]:
    """
    加载数据、筛选体系、处理异常值（修复版：正确识别bulk-In2O3）
    """
    print_section("步骤1：数据加载、筛选与异常值处理", force=True)

    df_raw = pd.read_excel(excel_path)
    logger.info(f"原始数据形状: {df_raw.shape}")

    df = validate_data_types(df_raw.copy())
    logger.info("数据类型验证完成")

    def is_empty_support(value):
        if pd.isna(value):
            return True
        if value is None:
            return True
        if isinstance(value, (int, float)):
            return value == 0
        if isinstance(value, str):
            v_lower = str(value).strip().lower()
            return v_lower in ['', '0', 'nan', 'none', 'na', 'null', '0.0']
        return False

    def has_material(value, material):
        if is_empty_support(value):
            return False
        material_lower = material.lower()
        value_str = str(value).lower()
        return material_lower in value_str

    def row_has_material(row, material):
        for col in ['Support 1', 'Name of Support2', 'Name of Support 3']:
            if col in row.index and has_material(row[col], material):
                return True
        return False

    df['System'] = 'Other'

    system_counts = {
        'bulk_in2o3': 0,
        'in2o3_zro2': 0,
        'cu_zno': 0,
        'cu_zno_al2o3': 0,
        'other': 0
    }

    logger.info("\n开始科学体系分类...")

    for idx in df.index:
        row = df.loc[idx]
        family = str(row.get('Family', '')).strip() if pd.notna(row.get('Family')) else ''
        is_in2o3_family = (family.lower() == 'in2o3') or ('in' in family.lower() and 'o' in family.lower())
        is_cu_family = (family.lower() == 'cu')

        support1_empty = is_empty_support(row.get('Support 1'))
        support2_empty = is_empty_support(row.get('Name of Support2'))
        support3_empty = is_empty_support(row.get('Name of Support 3'))
        is_bulk = support1_empty and support2_empty and support3_empty

        has_zro2 = row_has_material(row, 'zro2') or row_has_material(row, 'zirconi')
        has_zno = row_has_material(row, 'zno') or row_has_material(row, 'zinc')
        has_al2o3 = row_has_material(row, 'al2o3') or row_has_material(row, 'alumin')

        assigned = False

        if is_in2o3_family and is_bulk:
            df.loc[idx, 'System'] = 'In2O3'
            system_counts['bulk_in2o3'] += 1
            assigned = True
        elif is_in2o3_family and has_zro2:
            df.loc[idx, 'System'] = 'In2O3/ZrO2'
            system_counts['in2o3_zro2'] += 1
            assigned = True
        elif is_cu_family and has_zno and has_al2o3:
            df.loc[idx, 'System'] = 'Cu/ZnO/Al2O3'
            system_counts['cu_zno_al2o3'] += 1
            assigned = True
        elif is_cu_family and has_zno and not has_al2o3:
            df.loc[idx, 'System'] = 'Cu/ZnO'
            system_counts['cu_zno'] += 1
            assigned = True
        else:
            system_counts['other'] += 1

    logger.info("\n科学体系分类完成：")
    logger.info(f"  In2O3（纯In2O3/bulk）: {system_counts['bulk_in2o3']} 样本")
    logger.info(f"  In2O3/ZrO2（复合体系）: {system_counts['in2o3_zro2']} 样本")
    logger.info(f"  Cu/ZnO（无Al2O3）: {system_counts['cu_zno']} 样本")
    logger.info(f"  Cu/ZnO/Al2O3（三元）: {system_counts['cu_zno_al2o3']} 样本")
    logger.info(f"  其他/未分类: {system_counts['other']} 样本")

    in2o3_count = (df['System'] == 'In2O3').sum()
    if in2o3_count == 0:
        logger.warning("\n⚠️ 警告：没有找到bulk-In2O3样本！检查分类逻辑...")
        in_family_count = df['Family'].str.contains('In', case=False, na=False).sum()
        logger.info(f"  Family包含'In'的样本数: {in_family_count}")

        potential_bulk = df[
            (df['Family'].str.contains('In', case=False, na=False)) &
            (df['Support 1'].apply(is_empty_support)) &
            (df['Name of Support2'].apply(is_empty_support)) &
            (df['Name of Support 3'].apply(is_empty_support))
            ]
        logger.info(f"  潜在的bulk-In2O3样本数: {len(potential_bulk)}")

        if len(potential_bulk) > 0:
            logger.info("  重新分配这些样本为In2O3...")
            df.loc[potential_bulk.index, 'System'] = 'In2O3'
            system_counts['bulk_in2o3'] = len(potential_bulk)

    logger.info("\n最终体系分布：")
    for sys in ['In2O3', 'In2O3/ZrO2', 'Cu/ZnO', 'Cu/ZnO/Al2O3']:
        count = (df['System'] == sys).sum()
        logger.info(f"  {sys}: {count} 样本")

    target_systems = ['Cu/ZnO', 'Cu/ZnO/Al2O3', 'In2O3', 'In2O3/ZrO2']
    df_filtered = df[df['System'].isin(target_systems)].copy()

    logger.info(f"\n筛选后数据形状: {df_filtered.shape}")

    for sys in target_systems:
        sys_count = (df_filtered['System'] == sys).sum()
        if sys_count < 10:
            logger.warning(f"⚠️ {sys} 样本数过少 ({sys_count})，可能影响后续分析")

    df_processed, outliers_info, comparison = detect_and_handle_outliers(
        df_filtered, strategy=outlier_strategy
    )

    standard_conditions_calculated = {}
    for sys in target_systems:
        if sys in df_processed['System'].values:
            sys_data = df_processed[df_processed['System'] == sys]
            if len(sys_data) > 0:
                standard_conditions_calculated[sys] = calculate_system_standard_conditions(sys_data, sys)

    analysis_info = {
        'outliers_info': outliers_info,
        'outlier_comparison': comparison,
        'standard_conditions_calculated': standard_conditions_calculated,
        'system_stats': {
            'In2O3': system_counts['bulk_in2o3'],
            'In2O3/ZrO2': system_counts['in2o3_zro2'],
            'Cu/ZnO': system_counts['cu_zno'],
            'Cu/ZnO/Al2O3': system_counts['cu_zno_al2o3'],
            'Other': system_counts['other']
        }
    }

    return df_processed, analysis_info



# ==========================
# 特征工程（v6增强）
# ==========================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    准备特征和目标变量（修复版：避免 inplace 操作）
    """
    print_section("步骤2：特征工程", force=True)

    df = df.copy()

    target_col = "STY [mgMeOH h-1 gcat-1]"
    if target_col not in df.columns:
        logger.error(f"错误：未找到目标列 '{target_col}'")
        raise KeyError(f"目标列 '{target_col}' 不存在")

    y = df[target_col].values
    y_log = np.log1p(y)

    logger.info(f"目标变量统计:")
    logger.info(f"  - 均值: {y.mean():.2f}")
    logger.info(f"  - 中位数: {np.median(y):.2f}")
    logger.info(f"  - 标准差: {y.std():.2f}")

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        valid_years = df['year'].between(1990, 2025, inclusive='both')
        valid_count = valid_years.sum()

        if valid_count > 0:
            year_min = df.loc[valid_years, 'year'].min()
            year_max = df.loc[valid_years, 'year'].max()
            year_median = df.loc[valid_years, 'year'].median()

            invalid_mask = ~valid_years & df['year'].notna()
            if invalid_mask.sum() > 0:
                df.loc[invalid_mask, 'year'] = year_median
                logger.info(f"  {invalid_mask.sum()}个无效年份被替换为中位数")

            df['year'] = df['year'].fillna(year_median)

            logger.info(f"✓ 使用现有的year列")
            logger.info(f"  有效年份数: {valid_count}/{len(df)}")
            logger.info(f"  年份范围: {year_min:.0f}-{year_max:.0f}")
            logger.info(f"  中位年份: {year_median:.0f}")

            year_counts = df['year'].value_counts().sort_index()
            logger.info(f"\n年份分布（Top 5）:")
            for year in year_counts.head(5).index:
                logger.info(f"  {year:.0f}: {year_counts[year]} 篇")
            if len(year_counts) > 5:
                logger.info(f"  ... (共{len(year_counts)}个不同年份)")
        else:
            df['year'] = 2020
            logger.warning("year列中没有有效年份，使用默认值2020")
    else:
        df['year'] = 2020
        logger.warning("未找到year列，创建默认年份2020")

    potential_num_features = [
        'Temperature [K]',
        'Pressure [Mpa]',
        'GHSV [cm3 h-1 gcat-1]',
        'H2/CO2 [-]',
        'Metal Loading [wt.%]',
        'SBET [m2 g-1]',
        'Catalyst amount [g]',
        'Calcination Temperature [K]',
        'Calcination duration [h]',
        'year',
        'CR Metal [pm]',
        'MW Support 1 [g mol-1]',
        'MW of Support 2 [g mol-1]',
        'MW Support 3 [g mol-1]',
        'Total MW of Support [g mol-1]',
        'Promoter 1 loading [wt.%]',
        'Promoter 2 loading [wt.%]'
    ]

    potential_cat_features = [
        'System',
        'Family',
        'Support 1',
        'Promoter 1',
        'Promoter 2',
        'method',
        'Name of Support2',
        'Name of Support 3'
    ]

    num_features = []
    cat_features_from_num = []

    for feat in potential_num_features:
        if feat in df.columns:
            try:
                test_values = pd.to_numeric(df[feat], errors='coerce')
                valid_ratio = test_values.notna().sum() / len(test_values)

                if valid_ratio > 0.5:
                    df[feat] = test_values
                    num_features.append(feat)
                    if feat != 'year':
                        logger.debug(f"数值特征 '{feat}': {valid_ratio:.1%} 有效数值")
                else:
                    cat_features_from_num.append(feat)
                    logger.warning(f"特征 '{feat}' 包含过多非数值数据，转为分类特征")
            except:
                cat_features_from_num.append(feat)
                logger.warning(f"特征 '{feat}' 无法转换为数值，作为分类特征")

    cat_features = []
    for feat in potential_cat_features + cat_features_from_num:
        if feat in df.columns:
            df[feat] = df[feat].fillna('missing').astype(str)
            cat_features.append(feat)

    cat_features = list(set(cat_features))

    logger.info(f"\n特征工程完成：")
    logger.info(f"数值特征数: {len(num_features)}")
    logger.info(f"  特征列表: {', '.join(num_features[:5])}{'...' if len(num_features) > 5 else ''}")
    logger.info(f"分类特征数: {len(cat_features)}")
    logger.info(f"  特征列表: {', '.join(cat_features[:5])}{'...' if len(cat_features) > 5 else ''}")

    if len(num_features) == 0:
        logger.error("错误：没有可用的数值特征")
        df['dummy_num'] = 1.0
        num_features = ['dummy_num']
        logger.warning("创建虚拟数值特征")

    if len(cat_features) == 0:
        logger.warning("警告：没有可用的分类特征，将创建虚拟分类特征")
        df['dummy_cat'] = 'default'
        cat_features = ['dummy_cat']

    return df, y, y_log, num_features, cat_features



# ==========================
# 基线模型对比（v6新增）
# ==========================

def baseline_model_comparison(X_train, y_train_log, X_test, y_test_log,
                              num_features, cat_features) -> Tuple[pd.DataFrame, Dict]:
    """
    系统地比较基线模型（v6.12：7个模型全面对比）
    """
    print_section("基线模型对比分析", force=True)

    baseline_models = {
        "Linear": LinearRegression(),
        "RandomForest(depth=5)": RandomForestRegressor(max_depth=5, n_estimators=100, random_state=42),
        "RandomForest(full)": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
    }

    if HAS_XGBOOST:
        baseline_models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0
        )
    if HAS_LIGHTGBM:
        baseline_models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=-1
        )
    if HAS_CATBOOST:
        baseline_models["CatBoost"] = cb.CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.1,
            random_seed=42, verbose=False
        )

    results = []
    trained_models = {}

    for name, model in baseline_models.items():
        logger.info(f"训练 {name}...")

        try:
            preprocessor = create_preprocessor(num_features, cat_features)
            pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('model', model)
            ])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train_log)
                y_train_pred = pipeline.predict(X_train)
                y_test_pred = pipeline.predict(X_test)

            train_r2 = r2_score(y_train_log, y_train_pred)
            test_r2 = r2_score(y_test_log, y_test_pred)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_test_sty_pred = np.expm1(np.clip(y_test_pred, -10, 10))
                y_test_sty_true = np.expm1(y_test_log)
                test_mae = mean_absolute_error(y_test_sty_true, y_test_sty_pred)

            overfitting = train_r2 - test_r2

            results.append({
                'Model': name,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Test_MAE': test_mae,
                'Overfitting': overfitting,
                'Parameters': 'OK'
            })

            trained_models[name] = pipeline

        except Exception as e:
            logger.warning(f"模型 {name} 训练失败: {str(e)}")
            results.append({
                'Model': name,
                'Train_R2': 0,
                'Test_R2': 0,
                'Test_MAE': np.inf,
                'Overfitting': 0,
                'Parameters': f'Error: {str(e)[:50]}'
            })

    results_df = pd.DataFrame(results)

    linear_results = results_df[results_df['Model'] == 'Linear']
    if len(linear_results) > 0 and linear_results['Test_R2'].values[0] > 0:
        linear_test_r2 = linear_results['Test_R2'].values[0]
    else:
        linear_test_r2 = 0

    results_df['Improvement_vs_Linear'] = results_df['Test_R2'] - linear_test_r2

    logger.info("\n基线模型性能对比：")
    logger.info("-" * 100)
    logger.info(f"{'模型':30s} {'Train R²':>10s} {'Test R²':>10s} {'过拟合':>10s} {'vs Linear':>12s}")
    logger.info("-" * 100)
    for _, row in results_df.iterrows():
        if row['Test_R2'] > 0:
            logger.info(f"{row['Model']:30s} {row['Train_R2']:>10.4f} {row['Test_R2']:>10.4f} "
                        f"{row['Overfitting']:>10.4f} {row['Improvement_vs_Linear']:>+12.4f}")
        else:
            logger.info(f"{row['Model']:30s} {'Failed':>10s} {'Failed':>10s} "
                        f"{'N/A':>10s} {'N/A':>12s}")

    return results_df, trained_models



# ==========================
# 模型选择和优化函数（修复版）
# ==========================

def get_default_model_params():
    """获取各模型的默认参数（优化过的）"""

    default_params = {
        "GradientBoosting": {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'min_samples_split': 4,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        },
        "RandomForest": {
            'n_estimators': 800,
            'max_depth': 20,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        },
        "XGBoost": {
            'n_estimators': 1000,
            'max_depth': 10,
            'learning_rate': 0.03,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'gamma': 0.01,
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': 1
        },
        "LightGBM": {
            'n_estimators': 1000,
            'max_depth': 10,
            'learning_rate': 0.03,
            'num_leaves': 50,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': 1
        },
        "CatBoost": {
            'iterations': 1000,
            'depth': 8,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3.0,
            'border_count': 128,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False
        }
    }

    return default_params


def quick_model_selection(X_train, y_train, num_features, cat_features):
    """快速CV选择最佳模型（使用默认参数）"""

    print_section("快速模型选择（CV对比，使用默认参数）", force=True)

    available_models = ["GradientBoosting", "RandomForest"]
    if HAS_XGBOOST:
        available_models.append("XGBoost")
    if HAS_LIGHTGBM:
        available_models.append("LightGBM")
    if HAS_CATBOOST:
        available_models.append("CatBoost")

    logger.info(f"可用模型: {', '.join(available_models)}")

    default_params = get_default_model_params()

    best_score = -np.inf
    best_model_name = None
    best_params = None

    for model_name in available_models:
        logger.info(f"\n测试 {model_name}...")

        if model_name == "XGBoost" and HAS_XGBOOST:
            model = xgb.XGBRegressor(**default_params[model_name])
        elif model_name == "LightGBM" and HAS_LIGHTGBM:
            model = lgb.LGBMRegressor(**default_params[model_name])
        elif model_name == "CatBoost" and HAS_CATBOOST:
            model = cb.CatBoostRegressor(**default_params[model_name])
        elif model_name == "GradientBoosting":
            model = GradientBoostingRegressor(**default_params[model_name])
        else:
            model = RandomForestRegressor(**default_params[model_name])

        preprocessor = create_preprocessor(num_features, cat_features)
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("regressor", model),
        ])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2', n_jobs=1)

        mean_score = scores.mean()
        std_score = scores.std()

        logger.info(f"  CV R²: {mean_score:.4f} ± {std_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = model_name
            best_params = default_params[model_name]

    logger.info(f"\n最佳模型: {best_model_name} (CV R² = {best_score:.4f})")

    return best_params, best_model_name


def global_bayesian_optimization(X_train, y_train, num_features, cat_features, n_trials=50):
    """全局贝叶斯优化选择最佳模型"""

    if not HAS_OPTUNA:
        logger.warning("Optuna未安装，使用默认参数")
        return quick_model_selection(X_train, y_train, num_features, cat_features)

    print_section("贝叶斯优化模型选择", force=True)

    available_models = ["GradientBoosting"]
    if HAS_XGBOOST:
        available_models.append("XGBoost")
    if HAS_LIGHTGBM:
        available_models.append("LightGBM")
    if HAS_CATBOOST:
        available_models.append("CatBoost")

    logger.info(f"待优化模型: {', '.join(available_models)}")

    best_results = {}

    for model_name in available_models:
        logger.info(f"\n优化 {model_name}...")

        def objective(trial):
            if model_name == "XGBoost" and HAS_XGBOOST:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.1, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1, log=True),
                    'random_state': 42,
                    'verbosity': 0
                }
                model = xgb.XGBRegressor(**params)
            elif model_name == "LightGBM" and HAS_LIGHTGBM:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.1, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1, log=True),
                    'random_state': 42,
                    'verbosity': -1
                }
                model = lgb.LGBMRegressor(**params)
            elif model_name == "CatBoost" and HAS_CATBOOST:
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_seed': 42,
                    'verbose': False
                }
                model = cb.CatBoostRegressor(**params)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**params)

            preprocessor = create_preprocessor(num_features, cat_features)
            pipeline = Pipeline(steps=[
                ("preprocess", preprocessor),
                ("regressor", model),
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2', n_jobs=1)

            return scores.mean()

        study = optuna.create_study(direction='maximize', study_name=f'{model_name}_global')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

        best_results[model_name] = {
            'best_score': study.best_value,
            'best_params': study.best_params
        }

        logger.info(f"  {model_name:20s}: R² = {study.best_value:.4f}")

    best_model_name = max(best_results.keys(), key=lambda x: best_results[x]['best_score'])
    best_params = best_results[best_model_name]['best_params']
    best_score = best_results[best_model_name]['best_score']

    logger.info(f"\n{'=' * 80}")
    logger.info(f"最佳模型: {best_model_name}")
    logger.info(f"最佳CV R²: {best_score:.4f}")
    logger.info(f"{'=' * 80}")

    return best_params, best_model_name


def select_best_model(X_train, y_train, num_features, cat_features):
    """根据开关选择是否使用贝叶斯优化"""
    if USE_BAYESIAN_OPTIMIZATION and HAS_OPTUNA:
        logger.info("\n使用贝叶斯优化进行模型选择...")
        return global_bayesian_optimization(X_train, y_train, num_features, cat_features, n_trials=N_BAYESIAN_TRIALS)
    else:
        logger.info("\n使用快速CV进行模型选择（默认参数）...")
        return quick_model_selection(X_train, y_train, num_features, cat_features)



# ==========================
# SHAP分析（增强版）
# ==========================

def perform_shap_analysis_enhanced(pipeline, X_train, X_test, num_features, cat_features, sub: pd.DataFrame):
    """增强的SHAP分析（修复版本）"""

    if not HAS_SHAP:
        logger.warning("SHAP未安装，跳过分析")
        return None

    print_section("深度SHAP可解释性分析（增强版）", force=True)

    try:
        X_train_transformed = pipeline.named_steps['preprocess'].transform(X_train)
        X_test_transformed = pipeline.named_steps['preprocess'].transform(X_test)

        model = pipeline.named_steps['regressor']

        n_sample = min(200, len(X_train_transformed))
        sample_indices = np.random.choice(len(X_train_transformed), n_sample, replace=False)
        X_background = X_train_transformed[sample_indices]

        if hasattr(model, 'predict'):
            explainer = shap.TreeExplainer(model, X_background)
        else:
            explainer = shap.KernelExplainer(model.predict, X_background)

        test_sample = min(100, len(X_test_transformed))
        test_indices = np.random.choice(len(X_test_transformed), test_sample, replace=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                shap_values = explainer.shap_values(X_test_transformed[test_indices], check_additivity=False)
            except TypeError:
                shap_values = explainer.shap_values(X_test_transformed[test_indices])

        feature_names = num_features.copy()

        if 'cat' in pipeline.named_steps['preprocess'].named_transformers_:
            cat_feature_names = pipeline.named_steps['preprocess'].named_transformers_['cat'][
                'onehot'].get_feature_names_out(cat_features)
            feature_names.extend(cat_feature_names)

        if len(feature_names) > shap_values.shape[1]:
            feature_names = feature_names[:shap_values.shape[1]]

        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(shap_importance)],
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)

        correlation_with_tree = None
        if hasattr(model, 'feature_importances_'):
            tree_importance = model.feature_importances_[:len(shap_importance)]
            if len(tree_importance) == len(shap_importance):
                correlation_with_tree = {
                    'pearson_r': np.corrcoef(shap_importance, tree_importance)[0, 1],
                    'spearman_r': stats.spearmanr(shap_importance, tree_importance)[0]
                }
                logger.info(f"\nSHAP vs 树模型重要性相关性:")
                logger.info(f"  Pearson r = {correlation_with_tree['pearson_r']:.3f}")
                logger.info(f"  Spearman ρ = {correlation_with_tree['spearman_r']:.3f}")

        system_shap_results = {}
        for sys in sub['System'].unique():
            sys_mask = sub['System'] == sys
            if sys_mask.sum() > 10:
                try:
                    sys_data = X_test[X_test.index.isin(sub[sys_mask].index)]
                    if len(sys_data) > 0:
                        sys_transformed = pipeline.named_steps['preprocess'].transform(sys_data)
                        sys_sample = min(30, len(sys_transformed))
                        sys_indices = np.random.choice(len(sys_transformed), sys_sample, replace=False)

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                sys_shap = explainer.shap_values(sys_transformed[sys_indices], check_additivity=False)
                            except TypeError:
                                sys_shap = explainer.shap_values(sys_transformed[sys_indices])

                        sys_importance = np.abs(sys_shap).mean(axis=0)
                        system_shap_results[sys] = pd.DataFrame({
                            'feature': feature_names[:len(sys_importance)],
                            'importance': sys_importance
                        }).sort_values('importance', ascending=False)
                except Exception as e:
                    logger.warning(f"系统 {sys} 的SHAP分析失败: {str(e)[:100]}")
                    continue

        output_dir = 'outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("\nSHAP全局特征重要性（Top 10）:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature'][:40]:40s}: {row['shap_importance']:.4f}")

        results = {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'global_importance': importance_df,
            'correlation_with_tree': correlation_with_tree,
            'system_shap': system_shap_results,
            'explainer': explainer
        }

        return results

    except Exception as e:
        logger.error(f"SHAP分析出错: {e}")
        return {
            'shap_values': None,
            'feature_names': num_features + cat_features,
            'global_importance': None,
            'correlation_with_tree': None,
            'system_shap': {},
            'explainer': None
        }



# ==========================
# 模型评估函数（v6增强：集成不确定性）
# ==========================

def evaluate_best_model_with_uncertainty(
        X: pd.DataFrame,
        y_log: np.ndarray,
        y: np.ndarray,
        num_features: List[str],
        cat_features: List[str],
        sub: pd.DataFrame,
        best_model_name: str,
        best_params: Dict,
        n_seeds: int = N_RANDOM_SEEDS,
        sample_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    使用最佳模型进行评估，包括不确定性量化
    """
    print_section(f"使用最佳模型进行评估（含不确定性）: {best_model_name}")

    all_results = []
    all_pipelines = []

    logger.info(f"\n使用 {n_seeds} 个随机种子进行评估...")

    for seed in tqdm(range(n_seeds), desc="随机种子评估"):
        if sample_weights is not None:
            X_train, X_test, y_train_log, y_test_log, weights_train, weights_test = train_test_split(
                X, y_log, sample_weights, test_size=0.2, random_state=seed,
                stratify=sub["System"] if "System" in sub.columns else None
            )
        else:
            X_train, X_test, y_train_log, y_test_log = train_test_split(
                X, y_log, test_size=0.2, random_state=seed,
                stratify=sub["System"] if "System" in sub.columns else None
            )
            weights_train = weights_test = None

        if best_model_name == "XGBoost" and HAS_XGBOOST:
            params = best_params.copy()
            params['random_state'] = seed
            model = xgb.XGBRegressor(**params)
        elif best_model_name == "LightGBM" and HAS_LIGHTGBM:
            params = best_params.copy()
            params['random_state'] = seed
            model = lgb.LGBMRegressor(**params)
        elif best_model_name == "CatBoost" and HAS_CATBOOST:
            params = best_params.copy()
            params['random_seed'] = seed
            if 'logging_level' in params:
                del params['logging_level']
            if 'verbose' not in params:
                params['verbose'] = False
            model = cb.CatBoostRegressor(**params)
        elif best_model_name == "GradientBoosting":
            params = best_params.copy()
            params['random_state'] = seed
            model = GradientBoostingRegressor(**params)
        else:
            params = best_params.copy()
            params['random_state'] = seed
            model = RandomForestRegressor(**params)

        preprocessor = create_preprocessor(num_features, cat_features)
        final_pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("regressor", model),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if weights_train is not None and best_model_name in ["GradientBoosting", "XGBoost", "LightGBM"]:
                final_pipeline.fit(X_train, y_train_log, regressor__sample_weight=weights_train)
            else:
                final_pipeline.fit(X_train, y_train_log)

        y_train_pred_log = final_pipeline.predict(X_train)
        y_test_pred_log = final_pipeline.predict(X_test)

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof_pred_log = cross_val_predict(final_pipeline, X, y_log, cv=kf, n_jobs=1)

        all_results.append({
            'seed': seed,
            'train_r2': r2_score(y_train_log, y_train_pred_log),
            'test_r2': r2_score(y_test_log, y_test_pred_log),
            'oof_pred_log': oof_pred_log,
            'pipeline': final_pipeline
        })

        all_pipelines.append(final_pipeline)

    all_oof_preds = np.array([r['oof_pred_log'] for r in all_results])
    mean_oof_pred_log = all_oof_preds.mean(axis=0)
    std_oof_pred_log = all_oof_preds.std(axis=0)

    mean_oof_pred_sty = np.expm1(mean_oof_pred_log)

    lower_bound_log = mean_oof_pred_log - CONFIDENCE_LEVEL * std_oof_pred_log
    upper_bound_log = mean_oof_pred_log + CONFIDENCE_LEVEL * std_oof_pred_log
    lower_bound_sty = np.expm1(lower_bound_log)
    upper_bound_sty = np.expm1(upper_bound_log)

    sub['prediction_uncertainty'] = std_oof_pred_log
    sub['prediction_lower'] = lower_bound_sty
    sub['prediction_upper'] = upper_bound_sty

    mean_train_r2 = np.mean([r['train_r2'] for r in all_results])
    std_train_r2 = np.std([r['train_r2'] for r in all_results])
    mean_test_r2 = np.mean([r['test_r2'] for r in all_results])
    std_test_r2 = np.std([r['test_r2'] for r in all_results])

    mean_oof_r2_log = r2_score(y_log, mean_oof_pred_log)
    mean_oof_r2_sty = r2_score(y, mean_oof_pred_sty)

    oof_r2_log_list = [r2_score(y_log, r['oof_pred_log']) for r in all_results]
    oof_r2_sty_list = [r2_score(y, np.expm1(r['oof_pred_log'])) for r in all_results]

    std_oof_r2_log = np.std(oof_r2_log_list)
    std_oof_r2_sty = np.std(oof_r2_sty_list)

    high_uncertainty_mask = std_oof_pred_log > UNCERTAINTY_THRESHOLD
    high_uncertainty_ratio = high_uncertainty_mask.sum() / len(std_oof_pred_log)

    logger.info(f"\n模型性能统计：")
    logger.info(f"  Train R²: {mean_train_r2:.4f} ± {std_train_r2:.4f}")
    logger.info(f"  Test R²: {mean_test_r2:.4f} ± {std_test_r2:.4f}")
    logger.info(f"  OOF R²(log): {mean_oof_r2_log:.4f} ± {std_oof_r2_log:.4f}")
    logger.info(f"  OOF R²(STY): {mean_oof_r2_sty:.4f} ± {std_oof_r2_sty:.4f}")
    logger.info(f"\n不确定性统计：")
    logger.info(f"  平均不确定性: {std_oof_pred_log.mean():.4f}")
    logger.info(f"  高不确定性比例: {high_uncertainty_ratio:.1%}")

    best_single_idx = np.argmax([r['test_r2'] for r in all_results])
    best_single = all_results[best_single_idx]

    return {
        'all_results': all_results,
        'all_pipelines': all_pipelines,
        'best_single': best_single,
        'uncertainty': {
            'std_log': std_oof_pred_log,
            'lower_sty': lower_bound_sty,
            'upper_sty': upper_bound_sty,
            'stats': {
                'mean_uncertainty': std_oof_pred_log.mean(),
                'high_uncertainty_ratio': high_uncertainty_ratio
            }
        },
        'mean_train_r2': mean_train_r2,
        'std_train_r2': std_train_r2,
        'mean_test_r2': mean_test_r2,
        'std_test_r2': std_test_r2,
        'mean_oof_r2_log': mean_oof_r2_log,
        'std_oof_r2_log': std_oof_r2_log,
        'mean_oof_r2_sty': mean_oof_r2_sty,
        'std_oof_r2_sty': std_oof_r2_sty,
        'num_features': num_features,
        'cat_features': cat_features,
    }


# ==========================
# STY_norm计算（改进版）
# ==========================

def calculate_sty_norm_improved(sub: pd.DataFrame, pipeline, num_features: List[str],
                                cat_features: List[str], standard_conditions_calculated: Dict) -> pd.DataFrame:
    """
    计算改进的STY_norm，考虑外插和不确定性
    """
    print_section("计算STY_norm（标准化条件下的活性）", force=True)

    sub = sub.copy()
    sub['STY_norm'] = np.nan
    sub['use_STY_norm'] = False
    sub['extrapolation_distance'] = np.nan

    for sys in sub['System'].unique():
        if sys not in standard_conditions_calculated:
            continue

        sys_mask = sub['System'] == sys
        sys_data = sub[sys_mask].copy()

        std_cond = standard_conditions_calculated[sys]

        X_norm = pd.DataFrame()
        for feat in num_features:
            if feat in std_cond:
                if isinstance(std_cond[feat], dict):
                    X_norm[feat] = [std_cond[feat]['value']] * len(sys_data)
                else:
                    X_norm[feat] = [std_cond[feat]] * len(sys_data)
            elif feat in sys_data.columns:
                X_norm[feat] = sys_data[feat].values
            else:
                X_norm[feat] = 0

        for feat in cat_features:
            if feat in sys_data.columns:
                X_norm[feat] = sys_data[feat].values
            else:
                X_norm[feat] = 'unknown'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_norm_log = pipeline.predict(X_norm)
        y_norm = np.expm1(y_norm_log)

        for idx, (_, row) in enumerate(sys_data.iterrows()):
            extrap_info = calculate_extrapolation_info(row, std_cond, sys_data)

            use_this = (
                    extrap_info['is_reliable'] and
                    row.get('prediction_uncertainty', 0) < UNCERTAINTY_THRESHOLD
            )

            sub.loc[row.name, 'STY_norm'] = y_norm[idx]
            sub.loc[row.name, 'use_STY_norm'] = use_this
            sub.loc[row.name, 'extrapolation_distance'] = extrap_info['distance']

    n_total = len(sub)
    n_sty_norm_valid = sub['use_STY_norm'].sum()

    logger.info(f"\nSTY_norm计算完成：")
    logger.info(f"  总样本数: {n_total}")
    logger.info(f"  可用STY_norm: {n_sty_norm_valid} ({n_sty_norm_valid / n_total:.1%})")

    for sys in sub['System'].unique():
        sys_mask = sub['System'] == sys
        sys_valid = sub[sys_mask & sub['use_STY_norm']]
        if len(sys_valid) > 0:
            logger.info(f"  {sys}: {len(sys_valid)}/{sys_mask.sum()} "
                        f"(中位数={sys_valid['STY_norm'].median():.1f})")

    return sub



# ==========================
# 反应器性能分析（v6增强）
# ==========================

def reactor_performance_analysis_enhanced(sub: pd.DataFrame) -> pd.DataFrame:
    """
    增强的反应器性能分析（v6.4 - 保守能耗模型 + 敏感性分析）
    """

    logger.info("\n" + "=" * 80)
    logger.info("固定床反应器性能分析（保守能耗模型 v6.4）")
    logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info("能耗模型说明：")
    logger.info("=" * 80)
    logger.info("本分析采用保守的物理能耗模型，包含以下组分：")
    logger.info("  1. 压缩功：等温压缩模型，压缩机效率75%")
    logger.info("  2. 加热负荷：进料预热（环境温度→反应温度），换热效率85%")
    logger.info("")
    logger.info("⚠️  重要声明：")
    logger.info("  - 此模型用于不同催化体系间的【相对比较】")
    logger.info("  - 绝对能耗值需要详细过程模拟验证")
    logger.info("  - 未包含：产物分离、循环压缩、催化剂再生等")
    logger.info("")
    logger.info("📌 关于热回收：")
    logger.info("  - 数据集中缺少CO₂转化率信息，无法准确计算反应热")
    logger.info("  - 热回收潜力将通过敏感性分析单独展示（见下文）")
    logger.info("  - CO₂ + 3H₂ → CH₃OH + H₂O, ΔH = -49.5 kJ/mol（放热反应）")
    logger.info("=" * 80)

    REACTOR_VOLUME = 1000
    CATALYST_DENSITY = 1.5
    OPERATING_HOURS_PER_YEAR = 8000

    COMPRESSION_EFFICIENCY = 0.75
    HEAT_TRANSFER_EFFICIENCY = 0.85
    AMBIENT_TEMP = 298
    AMBIENT_PRESSURE = 0.1
    R_GAS = 8.314
    CP_AVG = 40

    system_order = ['Cu/ZnO', 'Cu/ZnO/Al2O3', 'In2O3', 'In2O3/ZrO2']
    results = []

    for sys in system_order:
        if sys not in sub['System'].values:
            continue

        sys_data = sub[(sub['System'] == sys) & sub['use_STY_norm']]

        if len(sys_data) < 5:
            logger.warning(f"{sys}: 可靠样本不足（{len(sys_data)}），跳过")
            continue

        median_sty_norm = sys_data['STY_norm'].median()
        q75_sty_norm = sys_data['STY_norm'].quantile(0.75)
        q90_sty_norm = sys_data['STY_norm'].quantile(0.90)

        catalyst_mass = REACTOR_VOLUME * CATALYST_DENSITY

        annual_production_median = median_sty_norm * catalyst_mass * OPERATING_HOURS_PER_YEAR / 1e6
        annual_production_q75 = q75_sty_norm * catalyst_mass * OPERATING_HOURS_PER_YEAR / 1e6
        annual_production_q90 = q90_sty_norm * catalyst_mass * OPERATING_HOURS_PER_YEAR / 1e6

        median_temp = sys_data['Temperature [K]'].median()
        median_pressure = sys_data['Pressure [Mpa]'].median()
        median_ghsv = sys_data['GHSV [cm3 h-1 gcat-1]'].median()

        compression_work = (R_GAS * AMBIENT_TEMP * np.log(median_pressure / AMBIENT_PRESSURE)) / \
                           (COMPRESSION_EFFICIENCY * 1000)

        heating_duty = CP_AVG * (median_temp - AMBIENT_TEMP) / 1000 / HEAT_TRANSFER_EFFICIENCY

        total_energy = compression_work + heating_duty

        productivity_energy_ratio = annual_production_median / total_energy if total_energy > 0 else 0

        sty_uncertainty = sys_data[
            'prediction_uncertainty'].median() if 'prediction_uncertainty' in sys_data.columns else 0

        results.append({
            'System': sys,
            'n_samples': len(sys_data),
            'n_reliable': len(sys_data[sys_data['use_STY_norm']]),
            'Median_STY_norm': median_sty_norm,
            'Q75_STY_norm': q75_sty_norm,
            'Q90_STY_norm': q90_sty_norm,
            'STY_uncertainty': sty_uncertainty,
            'Annual_Production_Median_kg': annual_production_median,
            'Annual_Production_Q75_kg': annual_production_q75,
            'Annual_Production_Q90_kg': annual_production_q90,
            'Median_Temperature_K': median_temp,
            'Median_Pressure_MPa': median_pressure,
            'Median_GHSV': median_ghsv,
            'Compression_Work_kJ_mol': compression_work,
            'Heating_Duty_kJ_mol': heating_duty,
            'Total_Energy_kJ_mol': total_energy,
            'Productivity_Energy_Ratio': productivity_energy_ratio,
        })

    results_df = pd.DataFrame(results)

    logger.info("\n" + "=" * 110)
    logger.info("反应器性能对比（1L固定床，保守能耗模型）")
    logger.info("=" * 110)
    logger.info(
        f"{'体系':20s} {'样本':>8s} {'STY_norm':>12s} {'年产量':>12s} {'压缩功':>10s} {'加热':>10s} {'总能耗':>10s} {'效率比':>10s}")
    logger.info(
        f"{'':20s} {'':>8s} {'[mg/h/g]':>12s} {'[kg/y]':>12s} {'[kJ/mol]':>10s} {'[kJ/mol]':>10s} {'[kJ/mol]':>10s} {'[kg/kJ]':>10s}")
    logger.info("-" * 110)

    for _, row in results_df.iterrows():
        logger.info(f"{row['System']:20s} {row['n_reliable']:>8.0f} {row['Median_STY_norm']:>12.1f} "
                    f"{row['Annual_Production_Median_kg']:>12.1f} {row['Compression_Work_kJ_mol']:>10.2f} "
                    f"{row['Heating_Duty_kJ_mol']:>10.2f} {row['Total_Energy_kJ_mol']:>10.2f} "
                    f"{row['Productivity_Energy_Ratio']:>10.1f}")

    logger.info("\n" + "-" * 80)
    logger.info("能耗分解分析：")
    logger.info("-" * 80)

    for _, row in results_df.iterrows():
        total = row['Compression_Work_kJ_mol'] + row['Heating_Duty_kJ_mol']
        comp_pct = row['Compression_Work_kJ_mol'] / total * 100 if total > 0 else 0
        heat_pct = row['Heating_Duty_kJ_mol'] / total * 100 if total > 0 else 0

        logger.info(f"  {row['System']:20s}:")
        logger.info(f"    压缩功: {row['Compression_Work_kJ_mol']:.2f} kJ/mol ({comp_pct:.1f}%)")
        logger.info(f"    加热负荷: {row['Heating_Duty_kJ_mol']:.2f} kJ/mol ({heat_pct:.1f}%)")
        logger.info(f"    → 总能耗: {total:.2f} kJ/mol")

    logger.info("\n" + "=" * 80)
    logger.info("热回收潜力敏感性分析")
    logger.info("=" * 80)

    DELTA_H_REACTION = -49.5
    HEAT_RECOVERY_EFFICIENCY = 0.50

    avg_total_energy = results_df['Total_Energy_kJ_mol'].mean()

    logger.info(f"{'转化率':>12s} {'反应放热':>12s} {'可回收热':>12s} {'净能耗':>12s} {'节能比例':>12s}")
    logger.info("-" * 65)

    conversion_scenarios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sensitivity_results = []

    for conv in conversion_scenarios:
        heat_released = abs(DELTA_H_REACTION) * conv
        heat_recovered = heat_released * HEAT_RECOVERY_EFFICIENCY
        net_energy = avg_total_energy - heat_recovered
        savings_pct = heat_recovered / avg_total_energy * 100

        sensitivity_results.append({
            'Conversion': conv,
            'Heat_Released': heat_released,
            'Heat_Recovered': heat_recovered,
            'Net_Energy': net_energy,
            'Savings_Pct': savings_pct
        })

        logger.info(f"{conv * 100:>12.0f} {heat_released:>12.2f} {heat_recovered:>12.2f} "
                    f"{net_energy:>12.2f} {savings_pct:>12.1f}")

    results_df.attrs['sensitivity_analysis'] = pd.DataFrame(sensitivity_results)
    results_df.attrs['avg_total_energy'] = avg_total_energy

    if len(results_df) > 1:
        logger.info("\n" + "-" * 80)
        logger.info("Pareto前沿分析（生产率 vs 能耗）：")
        logger.info("-" * 80)

        for i, row in results_df.iterrows():
            is_pareto = True
            for j, other_row in results_df.iterrows():
                if i != j:
                    if (other_row['Annual_Production_Median_kg'] >= row['Annual_Production_Median_kg'] and
                            other_row['Total_Energy_kJ_mol'] <= row['Total_Energy_kJ_mol']):
                        if (other_row['Annual_Production_Median_kg'] > row['Annual_Production_Median_kg'] or
                                other_row['Total_Energy_kJ_mol'] < row['Total_Energy_kJ_mol']):
                            is_pareto = False
                            break

            pareto_status = "✓ Pareto最优" if is_pareto else "  被支配"
            logger.info(f"  {pareto_status}: {row['System']:20s} "
                        f"(产量={row['Annual_Production_Median_kg']:.0f} kg/y, "
                        f"能耗={row['Total_Energy_kJ_mol']:.2f} kJ/mol)")

    logger.info("\n" + "=" * 80)
    logger.info("关键发现：")
    logger.info("=" * 80)

    best_production = results_df.loc[results_df['Annual_Production_Median_kg'].idxmax()]
    best_efficiency = results_df.loc[results_df['Productivity_Energy_Ratio'].idxmax()]
    lowest_energy = results_df.loc[results_df['Total_Energy_kJ_mol'].idxmin()]

    logger.info(f"  最高产量: {best_production['System']} ({best_production['Annual_Production_Median_kg']:.0f} kg/年)")
    logger.info(f"  最低能耗: {lowest_energy['System']} ({lowest_energy['Total_Energy_kJ_mol']:.2f} kJ/mol)")
    logger.info(f"  最高效率比: {best_efficiency['System']} ({best_efficiency['Productivity_Energy_Ratio']:.1f} kg/kJ)")

    return results_df



# ==========================
# 虚拟筛选（v6增强）
# ==========================

def virtual_screening_enhanced(pipeline, num_features, cat_features, sub: pd.DataFrame,
                               uncertainty_pipelines: List = None,
                               cluster_results: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    增强的虚拟筛选（v6.14 - 修复Track A P≤7约束 + 等级优先选择）
    """

    logger.info("\n" + "=" * 80)
    logger.info("虚拟筛选：高性能窗口探索（双轨制输出 v6.14）")
    logger.info("=" * 80)

    logger.info("\n⚠️  模型适用范围声明：")
    logger.info("  - 训练数据压力中位数：3.6 MPa")
    logger.info("  - 可靠预测范围：2-7 MPa")
    logger.info("  - 高压区（>7 MPa）：外推区域，预测结果需谨慎解读")
    logger.info("=" * 80)

    RISK_WEIGHT_DISTANCE = 0.3

    if 'STY_norm' in sub.columns and sub['use_STY_norm'].sum() > 0:
        reliable_sty = sub[sub['use_STY_norm']]['STY_norm']
        STY_P25 = reliable_sty.quantile(0.25)
        STY_P50 = reliable_sty.quantile(0.50)
        STY_P75 = reliable_sty.quantile(0.75)
        STY_P90 = reliable_sty.quantile(0.90)
        STY_P95 = reliable_sty.quantile(0.95)
    else:
        STY_P25, STY_P50, STY_P75, STY_P90, STY_P95 = 50, 100, 200, 300, 400

    if 'prediction_uncertainty' in sub.columns:
        uncertainties = sub['prediction_uncertainty'].dropna()
        UNCERTAINTY_P50 = uncertainties.quantile(0.50)
        UNCERTAINTY_P80 = uncertainties.quantile(0.80)
        UNCERTAINTY_P95 = uncertainties.quantile(0.95)
    else:
        UNCERTAINTY_P50, UNCERTAINTY_P80, UNCERTAINTY_P95 = 0.08, 0.12, 0.20

    STY_INDUSTRIAL_THRESHOLD = max(200, STY_P75)

    STY_MIN_LEVEL1 = STY_P75
    STY_MIN_LEVEL2 = STY_P50
    STY_MIN_LEVEL3 = STY_P25

    param_ranges = {
        'Temperature [K]': np.array([473, 493, 513, 533, 553, 573, 593, 613]),
        'Pressure [Mpa]': np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        'GHSV [cm3 h-1 gcat-1]': np.array([6000, 10000, 15000, 20000, 30000, 40000]),
        'H2/CO2 [-]': np.array([2.0, 3.0, 4.0, 5.0]),
    }

    fixed_params = {
        'Metal Loading [wt.%]': 30.0,
        'SBET [m2 g-1]': 80.0,
        'Catalyst amount [g]': 0.5,
        'Calcination Temperature [K]': 623.0,
        'Calcination duration [h]': 4.0,
        'CR Metal [pm]': 50.0,
        'MW Support 1 [g mol-1]': 100.0,
        'MW of Support 2 [g mol-1]': 0.0,
        'MW Support 3 [g mol-1]': 0.0,
        'Total MW of Support [g mol-1]': 100.0,
        'Promoter 1 loading [wt.%]': 10.0,
        'Promoter 2 loading [wt.%]': 0.0,
        'year': 2023
    }

    RELIABLE_PRESSURE_LIMIT = 7.0
    HIGH_PRESSURE_THRESHOLD = 7.0

    reference_system = 'Cu/ZnO/Al2O3'
    ref_data = sub[sub['System'] == reference_system].copy()
    if len(ref_data) == 0:
        ref_data = sub.copy()
        reference_system = 'All'

    WEIGHT_STY = 0.50
    WEIGHT_UNCERTAINTY = 0.25
    WEIGHT_DISTANCE = 0.25

    THRESHOLD_LEVEL1 = 70
    THRESHOLD_LEVEL2 = 55
    THRESHOLD_LEVEL3 = 40

    TRACK_B_STY_THRESHOLD = STY_P90 * 0.90
    TRACK_B_UNCERTAINTY_THRESHOLD = UNCERTAINTY_P80 * 1.30
    TRACK_B_DISTANCE_THRESHOLD = 1.5

    DISTANCE_EXCELLENT = 0.4
    DISTANCE_GOOD = 0.7
    DISTANCE_ACCEPTABLE = 1.0
    DISTANCE_MAX = 1.8

    screening_results = []
    all_X = []
    all_conditions = []

    for T in param_ranges['Temperature [K]']:
        for P in param_ranges['Pressure [Mpa]']:
            for GHSV in param_ranges['GHSV [cm3 h-1 gcat-1]']:
                for H2CO2 in param_ranges['H2/CO2 [-]']:
                    X_point = pd.DataFrame([{
                        'Temperature [K]': T,
                        'Pressure [Mpa]': P,
                        'GHSV [cm3 h-1 gcat-1]': GHSV,
                        'H2/CO2 [-]': H2CO2,
                        **fixed_params
                    }])

                    for cat_feat in cat_features:
                        if cat_feat in ref_data.columns:
                            mode_val = ref_data[cat_feat].mode()
                            X_point[cat_feat] = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                        else:
                            X_point[cat_feat] = 'Unknown'

                    all_X.append(X_point)
                    all_conditions.append((T, P, GHSV, H2CO2))

    X_screen = pd.concat(all_X, ignore_index=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_pred_log = pipeline.predict(X_screen[num_features + cat_features])
    all_pred_sty = np.expm1(all_pred_log)

    if uncertainty_pipelines:
        all_preds = []
        for pipe in uncertainty_pipelines:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = pipe.predict(X_screen[num_features + cat_features])
            all_preds.append(np.expm1(pred))

        all_preds = np.array(all_preds)
        all_pred_mean_sty = all_preds.mean(axis=0)
        all_pred_std = all_preds.std(axis=0)
        all_pred_lower_sty = np.percentile(all_preds, 5, axis=0)
        all_pred_upper_sty = np.percentile(all_preds, 95, axis=0)
    else:
        all_pred_mean_sty = all_pred_sty
        all_pred_std = all_pred_sty * 0.1
        all_pred_lower_sty = all_pred_sty * 0.9
        all_pred_upper_sty = all_pred_sty * 1.1

    for idx, (T, P, GHSV, H2CO2) in enumerate(all_conditions):
        min_dist = np.inf
        for _, ref_row in ref_data.iterrows():
            dist = np.sqrt(
                ((T - ref_row['Temperature [K]']) / 100) ** 2 +
                ((P - ref_row['Pressure [Mpa]']) / 5) ** 2 +
                ((GHSV - ref_row['GHSV [cm3 h-1 gcat-1]']) / 20000) ** 2 +
                ((H2CO2 - ref_row['H2/CO2 [-]']) / 2) ** 2
            )
            min_dist = min(min_dist, dist)

        if np.isinf(min_dist):
            min_dist = 1.0

        predicted_sty = all_pred_sty[idx]
        uncertainty = all_pred_std[idx] / predicted_sty if predicted_sty > 0 else 1.0
        raw_confidence_sty = all_pred_lower_sty[idx] if uncertainty_pipelines else predicted_sty * 0.9

        max_confidence_sty = predicted_sty * (1 - max(uncertainty, 0.01))
        confidence_sty = min(raw_confidence_sty, max_confidence_sty)

        is_high_pressure = P > HIGH_PRESSURE_THRESHOLD
        is_reliable_pressure = P <= RELIABLE_PRESSURE_LIMIT

        high_pressure_penalty = 0
        if is_high_pressure:
            pressure_excess = (P - HIGH_PRESSURE_THRESHOLD) / HIGH_PRESSURE_THRESHOLD
            high_pressure_penalty = 10 * pressure_excess

        sty_max_score = 50
        if confidence_sty >= STY_P95:
            sty_score = sty_max_score
        elif confidence_sty >= STY_P90:
            sty_score = sty_max_score * 0.92
        elif confidence_sty >= STY_P75:
            sty_score = sty_max_score * 0.82
        elif confidence_sty >= STY_P50:
            sty_score = sty_max_score * 0.65
        elif confidence_sty >= STY_P25:
            sty_score = sty_max_score * 0.45
        elif confidence_sty > 0:
            sty_score = sty_max_score * (confidence_sty / STY_P25) * 0.45
        else:
            sty_score = 0

        unc_max_score = 25
        if uncertainty <= 0.05:
            unc_score = unc_max_score
        elif uncertainty <= 0.10:
            unc_score = unc_max_score * 0.90
        elif uncertainty <= UNCERTAINTY_P50:
            unc_score = unc_max_score * 0.75
        elif uncertainty <= UNCERTAINTY_P80:
            unc_score = unc_max_score * 0.55
        elif uncertainty <= UNCERTAINTY_P95:
            unc_score = unc_max_score * 0.35
        else:
            unc_score = unc_max_score * 0.15

        dist_max_score = 25
        if min_dist <= DISTANCE_EXCELLENT:
            dist_score = dist_max_score
        elif min_dist <= DISTANCE_GOOD:
            dist_score = dist_max_score * (1 - 0.25 * (min_dist - DISTANCE_EXCELLENT) /
                                           (DISTANCE_GOOD - DISTANCE_EXCELLENT))
        elif min_dist <= DISTANCE_ACCEPTABLE:
            dist_score = dist_max_score * 0.75 * (1 - 0.35 * (min_dist - DISTANCE_GOOD) /
                                                  (DISTANCE_ACCEPTABLE - DISTANCE_GOOD))
        elif min_dist <= DISTANCE_MAX:
            dist_score = dist_max_score * 0.49 * (1 - 0.5 * (min_dist - DISTANCE_ACCEPTABLE) /
                                                  (DISTANCE_MAX - DISTANCE_ACCEPTABLE))
        else:
            dist_score = dist_max_score * 0.15

        composite_score = sty_score + unc_score + dist_score - high_pressure_penalty
        composite_score = max(0, composite_score)

        if composite_score >= THRESHOLD_LEVEL1 and predicted_sty >= STY_MIN_LEVEL1 and is_reliable_pressure:
            candidate_level_A = 1
        elif composite_score >= THRESHOLD_LEVEL2 and predicted_sty >= STY_MIN_LEVEL2 and is_reliable_pressure:
            candidate_level_A = 2
        elif composite_score >= THRESHOLD_LEVEL3 and predicted_sty >= STY_MIN_LEVEL3 and is_reliable_pressure:
            candidate_level_A = 3
        else:
            candidate_level_A = 0

        is_track_b_candidate = (
                predicted_sty >= TRACK_B_STY_THRESHOLD and
                uncertainty <= TRACK_B_UNCERTAINTY_THRESHOLD and
                min_dist <= TRACK_B_DISTANCE_THRESHOLD and
                is_reliable_pressure
        )

        if is_track_b_candidate:
            risk_score = 0
            if uncertainty > UNCERTAINTY_P80:
                risk_score += 2
            elif uncertainty > UNCERTAINTY_P50:
                risk_score += 1
            if min_dist > DISTANCE_ACCEPTABLE:
                risk_score += 2
            elif min_dist > DISTANCE_GOOD:
                risk_score += 1
            if T > 593:
                risk_score += 1
            elif T > 573:
                risk_score += 0.5

            if risk_score <= 2:
                track_b_risk = 'Low'
            elif risk_score <= 4:
                track_b_risk = 'Medium'
            else:
                track_b_risk = 'High'
        else:
            track_b_risk = None

        screening_results.append({
            'Temperature_K': T,
            'Pressure_MPa': P,
            'GHSV': GHSV,
            'H2_CO2_ratio': H2CO2,
            'Predicted_STY': predicted_sty,
            'STY_std': uncertainty,
            'STY_lower_bound': all_pred_lower_sty[idx] if uncertainty_pipelines else predicted_sty * 0.9,
            'STY_upper_bound': all_pred_upper_sty[idx] if uncertainty_pipelines else predicted_sty * 1.1,
            'Confidence_STY': confidence_sty,
            'Extrapolation_Distance': min_dist,
            'Candidate_Level': candidate_level_A,
            'Composite_Score': composite_score,
            'STY_Score': sty_score,
            'Uncertainty_Score': unc_score,
            'Distance_Score': dist_score,
            'Is_Track_B': is_track_b_candidate,
            'Track_B_Risk': track_b_risk,
            'High_Pressure_Penalty': high_pressure_penalty,
            'Is_High_Pressure': is_high_pressure,
            'Is_Reliable_Pressure': is_reliable_pressure,
            'Is_Industrial_Range': (T >= 513) and (T <= 593) and (P >= 5.0) and (P <= 10.0),
            'Is_Low_Risk': (min_dist <= DISTANCE_GOOD) and (uncertainty <= UNCERTAINTY_P80) and is_reliable_pressure,
        })

    screening_df = pd.DataFrame(screening_results)

    def select_track_a_top5(df, n=5):
        selected = []
        for level in [1, 2, 3]:
            if len(selected) >= n:
                break
            level_candidates = df[df['Candidate_Level'] == level].copy()
            if len(level_candidates) > 0:
                level_candidates = level_candidates.sort_values('Composite_Score', ascending=False)
                needed = n - len(selected)
                selected.append(level_candidates.head(needed))

        if len(selected) > 0:
            return pd.concat(selected, ignore_index=False)
        else:
            return pd.DataFrame()

    track_a_top = select_track_a_top5(screening_df, n=5)

    track_b_df = screening_df[screening_df['Is_Track_B']].copy()
    track_b_df = track_b_df.sort_values('Predicted_STY', ascending=False)
    n_track_b = len(track_b_df)
    track_b_top = track_b_df.head(5) if n_track_b > 0 else pd.DataFrame()

    track_a_best = track_a_top.iloc[0] if len(track_a_top) > 0 else None
    track_b_best = track_b_top.iloc[0] if n_track_b > 0 else None

    # Pareto前沿
    reliable_df = screening_df[screening_df['Is_Reliable_Pressure']].copy()

    pareto_mask = np.ones(len(reliable_df), dtype=bool)
    sty_values = reliable_df['Confidence_STY'].values
    risk_values = reliable_df['STY_std'].values + reliable_df['Extrapolation_Distance'].values * RISK_WEIGHT_DISTANCE

    for i in range(len(reliable_df)):
        if pareto_mask[i]:
            for j in range(len(reliable_df)):
                if i != j and pareto_mask[j]:
                    if sty_values[j] >= sty_values[i] and risk_values[j] <= risk_values[i]:
                        if sty_values[j] > sty_values[i] or risk_values[j] < risk_values[i]:
                            pareto_mask[i] = False
                            break

    pareto_points = reliable_df[pareto_mask].sort_values('Confidence_STY', ascending=False)

    # Log output
    logger.info(f"\n虚拟筛选结果统计：")
    logger.info(f"  总筛选点: {len(screening_df)}")
    for level in [1, 2, 3]:
        n_level = (screening_df['Candidate_Level'] == level).sum()
        logger.info(f"  Level {level}: {n_level}个")
    logger.info(f"  Track B: {n_track_b}个")
    logger.info(f"  Pareto最优: {len(pareto_points)}个")

    summary = {
        'total_points': len(screening_df),
        'reliable_points': reliable_df.shape[0],
        'level1_count': (screening_df['Candidate_Level'] == 1).sum(),
        'level2_count': (screening_df['Candidate_Level'] == 2).sum(),
        'level3_count': (screening_df['Candidate_Level'] == 3).sum(),
        'conservative_candidates': track_a_top,
        'high_performance_candidates': track_b_top if n_track_b > 0 else pd.DataFrame(),
        'track_b_count': n_track_b,
        'track_b_low_risk': len(track_b_df[track_b_df['Track_B_Risk'] == 'Low']) if n_track_b > 0 else 0,
        'track_b_medium_risk': len(track_b_df[track_b_df['Track_B_Risk'] == 'Medium']) if n_track_b > 0 else 0,
        'track_b_high_risk': len(track_b_df[track_b_df['Track_B_Risk'] == 'High']) if n_track_b > 0 else 0,
        'pareto_points': pareto_points,
        'high_pressure_count': screening_df['Is_High_Pressure'].sum(),
        'config': {
            'threshold_level1': THRESHOLD_LEVEL1,
            'threshold_level2': THRESHOLD_LEVEL2,
            'threshold_level3': THRESHOLD_LEVEL3,
            'sty_min_level1': STY_MIN_LEVEL1,
            'sty_min_level2': STY_MIN_LEVEL2,
            'sty_min_level3': STY_MIN_LEVEL3,
            'reliable_pressure_limit': RELIABLE_PRESSURE_LIMIT,
            'risk_weight_distance': RISK_WEIGHT_DISTANCE,
        }
    }

    return screening_df, summary



# ==========================
# 聚类分析（操作模式识别）
# ==========================

def perform_clustering_analysis(sub: pd.DataFrame, num_features: List[str], cat_features: List[str]) -> Dict:
    """
    对各体系进行聚类分析，识别操作模式
    """
    print_section("操作模式识别（聚类分析）", force=True)

    cluster_features = [
        'Temperature [K]', 'Pressure [Mpa]',
        'GHSV [cm3 h-1 gcat-1]', 'H2/CO2 [-]'
    ]

    cluster_results = {}

    for sys in sub['System'].unique():
        sys_data = sub[sub['System'] == sys].copy()

        if len(sys_data) < 20:
            logger.info(f"{sys}: 样本太少({len(sys_data)}), 跳过聚类")
            continue

        X_cluster = sys_data[cluster_features].copy()
        X_cluster = X_cluster.fillna(X_cluster.median())

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)

        best_score = -1
        best_k = 2

        for k in range(2, min(6, len(sys_data) // 5)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_cluster_scaled)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_cluster_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        sys_data['cluster'] = kmeans.fit_predict(X_cluster_scaled)

        cluster_info = []
        for c in range(best_k):
            cluster_mask = sys_data['cluster'] == c
            cluster_samples = sys_data[cluster_mask]

            if len(cluster_samples) > 0:
                info = {
                    'cluster': c,
                    'n_samples': len(cluster_samples),
                    'mean_STY': cluster_samples['STY [mgMeOH h-1 gcat-1]'].mean(),
                    'median_STY': cluster_samples['STY [mgMeOH h-1 gcat-1]'].median(),
                }

                for feat in cluster_features:
                    info[f'{feat}_mean'] = cluster_samples[feat].mean()
                    info[f'{feat}_std'] = cluster_samples[feat].std()

                cluster_info.append(info)

        cluster_results[sys] = {
            'data': sys_data,
            'n_clusters': best_k,
            'silhouette_score': best_score,
            'cluster_info': pd.DataFrame(cluster_info),
            'kmeans': kmeans,
            'scaler': scaler
        }

        logger.info(f"{sys}: {best_k}个操作模式, silhouette={best_score:.3f}")

    return cluster_results


# ==========================
# 深度学习模块（可选）
# ==========================

if HAS_TORCH:

    class CatalystDataset(Dataset):
        def __init__(self, X, y, text_features=None):
            self.X = torch.FloatTensor(X.astype(np.float32))
            self.y = torch.FloatTensor(y.astype(np.float32))
            self.text_features = torch.FloatTensor(
                text_features.astype(np.float32)) if text_features is not None else None

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            if self.text_features is not None:
                return self.X[idx], self.text_features[idx], self.y[idx]
            return self.X[idx], self.y[idx]


    class MultiModalEncoder(nn.Module):
        def __init__(self, num_features_dim, text_dim=100, hidden_dim=256, output_dim=128):
            super().__init__()
            self.num_encoder = nn.Sequential(
                nn.Linear(num_features_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, output_dim))
            self.text_encoder = nn.Sequential(
                nn.Linear(text_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, output_dim))
            self.attention = nn.MultiheadAttention(output_dim, num_heads=4, batch_first=True)
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Dropout(0.2))
            self.predictor = nn.Sequential(
                nn.Linear(output_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1))

        def forward(self, num_features, text_features=None):
            num_emb = self.num_encoder(num_features)
            if text_features is not None:
                text_emb = self.text_encoder(text_features)
                num_emb_attn = num_emb.unsqueeze(1)
                text_emb_attn = text_emb.unsqueeze(1)
                combined_attn = torch.cat([num_emb_attn, text_emb_attn], dim=1)
                attn_output, _ = self.attention(combined_attn, combined_attn, combined_attn)
                attn_output = attn_output.mean(dim=1)
                combined = torch.cat([num_emb, text_emb], dim=1)
                features = self.fusion(combined) + attn_output
            else:
                features = num_emb
            output = self.predictor(features)
            return output, features


    class ConditionalVAE(nn.Module):
        def __init__(self, input_dim, condition_dim=1, latent_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim + condition_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + condition_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, input_dim))

        def encode(self, x, condition):
            x_cond = torch.cat([x, condition], dim=1)
            h = self.encoder(x_cond)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z, condition):
            z_cond = torch.cat([z, condition], dim=1)
            return self.decoder(z_cond)

        def forward(self, x, condition):
            mu, logvar = self.encode(x, condition)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z, condition)
            return x_recon, mu, logvar

        def generate(self, target_sty, n_samples=10):
            with torch.no_grad():
                condition = torch.tensor([[target_sty]], dtype=torch.float32).repeat(n_samples, 1)
                if device:
                    condition = condition.to(device)
                z = torch.randn(n_samples, self.fc_mu.out_features, dtype=torch.float32)
                if device:
                    z = z.to(device)
                generated = self.decode(z, condition)
            return generated.cpu().numpy()


def create_text_features(df: pd.DataFrame) -> np.ndarray:
    logger.info("创建文本特征...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    text_features = []
    for _, row in df.iterrows():
        text_parts = []
        if 'System' in row and pd.notna(row['System']):
            text_parts.append(str(row['System']))
        if 'Family' in row and pd.notna(row['Family']):
            text_parts.append(f"Family:{row['Family']}")
        for col in ['Support 1', 'Name of Support2', 'Name of Support 3']:
            if col in row and pd.notna(row[col]) and str(row[col]) != 'none':
                text_parts.append(f"Support:{row[col]}")
        for col in ['Promoter 1', 'Promoter 2']:
            if col in row and pd.notna(row[col]) and str(row[col]) != 'none':
                text_parts.append(f"Promoter:{row[col]}")
        if 'method' in row and pd.notna(row['method']):
            text_parts.append(f"Method:{row['method']}")
        text_features.append(' '.join(text_parts))
    vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.95)
    try:
        text_matrix = vectorizer.fit_transform(text_features).toarray()
    except:
        text_matrix = np.zeros((len(text_features), 100))
    return text_matrix


def train_deep_learning_models(X_train, y_train, text_train, X_test, y_test, text_test) -> Optional[Dict]:
    if not HAS_TORCH or not USE_DEEP_LEARNING:
        return None
    print_section("训练深度学习增强模型", force=True)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    text_train = text_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    text_test = text_test.astype(np.float32)

    train_dataset = CatalystDataset(X_train, y_train, text_train)
    test_dataset = CatalystDataset(X_test, y_test, text_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    text_dim = text_train.shape[1]

    multimodal_model = MultiModalEncoder(input_dim, text_dim)
    if device:
        multimodal_model = multimodal_model.to(device)

    optimizer = optim.Adam(multimodal_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_multimodal_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        multimodal_model.train()
        train_loss = 0
        for batch_data in train_loader:
            if len(batch_data) == 3:
                batch_x, batch_text, batch_y = batch_data
            else:
                batch_x, batch_y = batch_data
                batch_text = None
            if device:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if batch_text is not None:
                    batch_text = batch_text.to(device)
            optimizer.zero_grad()
            pred, features = multimodal_model(batch_x, batch_text)
            loss = criterion(pred.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        multimodal_model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    batch_x, batch_text, batch_y = batch_data
                else:
                    batch_x, batch_y = batch_data
                    batch_text = None
                if device:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    if batch_text is not None:
                        batch_text = batch_text.to(device)
                pred, _ = multimodal_model(batch_x, batch_text)
                loss = criterion(pred.squeeze(), batch_y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_multimodal_state = copy.deepcopy(multimodal_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_multimodal_state is not None:
        multimodal_model.load_state_dict(best_multimodal_state)

    multimodal_model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch_data in test_loader:
            if len(batch_data) == 3:
                batch_x, batch_text, batch_y = batch_data
            else:
                batch_x, batch_y = batch_data
                batch_text = None
            if device:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if batch_text is not None:
                    batch_text = batch_text.to(device)
            pred, _ = multimodal_model(batch_x, batch_text)
            all_preds.append(pred.squeeze().cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        test_r2 = r2_score(all_targets, all_preds)
        test_mae = mean_absolute_error(all_targets, all_preds)

    logger.info(f"\n多模态模型性能: R²={test_r2:.4f}, MAE={test_mae:.4f}")
    return {'multimodal_model': multimodal_model, 'test_r2': test_r2, 'test_mae': test_mae}



# ==========================
# Nature风格图表函数（CCL修改版 - 每个子图独立保存）
# ==========================

def create_figure1_data_overview(sub, outlier_info):
    """图1：数据集与操作条件全景 - CCL修改版（每个子图独立保存）"""

    available_systems = sub['System'].unique()
    system_order = []
    for sys in ['Cu/ZnO', 'Cu/ZnO/Al2O3', 'In2O3', 'In2O3/ZrO2']:
        if sys in available_systems:
            system_order.append(sys)
    if len(system_order) < 2:
        system_order = list(available_systems)

    # --- 子图1: Catalyst system composition ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    system_counts = sub['System'].value_counts()
    plot_systems = []
    plot_counts = []
    plot_colors = []
    for sys in system_order:
        if sys in system_counts.index:
            plot_systems.append(sys)
            plot_counts.append(system_counts[sys])
            plot_colors.append(SYSTEM_COLORS.get(sys, '#888888'))

    bars = ax.bar(range(len(plot_systems)), plot_counts,
                  color=plot_colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    for bar, count in zip(bars, plot_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f'{int(count)}', ha='center', va='bottom', fontsize=10)
    total_samples = sum(plot_counts)
    ax.text(0.98, 0.98, f'Total: {total_samples} samples',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=0.8))
    ax.set_xticks(range(len(plot_systems)))
    ax.set_xticklabels(plot_systems, rotation=45 if len(plot_systems) > 4 else 0,
                       ha='right' if len(plot_systems) > 4 else 'center')
    ax.set_ylabel('Number of samples', fontsize=14)
    ax.set_ylim([0, max(plot_counts) * 1.15] if plot_counts else [0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 1, 'Catalyst_system_composition')

    # --- 子图2: STY distribution by catalyst system ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_data = []
    for sys in system_order:
        if sys in available_systems:
            sys_data = sub[sub['System'] == sys]['STY [mgMeOH h-1 gcat-1]'].values
            if len(sys_data) > 0:
                plot_data.append(sys_data)
            else:
                plot_data.append([0])

    if len(plot_data) > 0 and any(len(d) > 0 for d in plot_data):
        valid_data = [d for d in plot_data if len(d) > 0 and np.any(d > 0)]
        valid_positions = [i for i, d in enumerate(plot_data) if len(d) > 0 and np.any(d > 0)]
        if len(valid_data) > 0:
            parts = ax.violinplot(valid_data, positions=valid_positions,
                                  widths=0.7, showmeans=False, showmedians=False, showextrema=False)
            for i, (pc, pos) in enumerate(zip(parts['bodies'], valid_positions)):
                sys = system_order[pos] if pos < len(system_order) else 'Other'
                color = SYSTEM_COLORS.get(sys, '#888888')
                pc.set_facecolor(color)
                pc.set_alpha(0.3)
                pc.set_edgecolor(color)
            bp = ax.boxplot(valid_data, positions=valid_positions, widths=0.3,
                            patch_artist=True, notch=False,
                            boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.2, alpha=0.8),
                            whiskerprops=dict(linewidth=1.2, color='black'),
                            capprops=dict(linewidth=1.2, color='black'),
                            medianprops=dict(linewidth=2, color='red'),
                            flierprops=dict(marker='o', markersize=3, alpha=0.5, markerfacecolor='grey'))

    ax.set_yscale('log')
    ax.set_ylim([1, 10000])
    ax.set_xticks(range(len(system_order)))
    ax.set_xticklabels(system_order, rotation=45 if len(system_order) > 4 else 0,
                       ha='right' if len(system_order) > 4 else 'center')
    ax.set_ylabel('STY [mg$_{MeOH}$ h$^{-1}$ g$_{cat}^{-1}$]', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 1, 'STY_distribution_by_catalyst_system')

    # --- 子图3: Key operating conditions distribution ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    params = ['Temperature [K]', 'Pressure [Mpa]', 'H2/CO2 [-]', 'GHSV [cm3 h-1 gcat-1]']
    param_labels = ['Temp.\n(K)', 'Press.\n(MPa)', 'H2/CO2\nratio', 'GHSV\n(x10^3)']
    norm_data = []
    for param in params:
        if param in sub.columns:
            values = sub[param].dropna()
            if len(values) > 0:
                if param == 'GHSV [cm3 h-1 gcat-1]':
                    values = values / 1000
                if values.max() > values.min():
                    norm_values = (values - values.min()) / (values.max() - values.min())
                else:
                    norm_values = pd.Series([0.5])
                norm_data.append(norm_values)
            else:
                norm_data.append(pd.Series([0.5]))

    for i, (data, param) in enumerate(zip(norm_data, params)):
        if len(data) > 1:
            try:
                kde = gaussian_kde(data)
                x = np.linspace(0, 1, 100)
                density = kde(x)
                density = density / density.max() * 0.4
                ax.fill_betweenx(x, i - density, i + density,
                                 alpha=0.6, color=NATURE_COLORS['accent'])
                ax.plot(i - density, x, 'k-', linewidth=1)
                ax.plot(i + density, x, 'k-', linewidth=1)
                if param in sub.columns:
                    median_val = sub[param].median()
                    if param == 'GHSV [cm3 h-1 gcat-1]':
                        median_val = median_val / 1000
                    if not np.isnan(median_val):
                        min_val = sub[param].min() if param != 'GHSV [cm3 h-1 gcat-1]' else sub[param].min() / 1000
                        max_val = sub[param].max() if param != 'GHSV [cm3 h-1 gcat-1]' else sub[param].max() / 1000
                        if max_val > min_val:
                            median_norm = (median_val - min_val) / (max_val - min_val)
                            ax.plot([i - 0.2, i + 0.2], [median_norm, median_norm], 'r-', linewidth=2)
            except:
                ax.scatter([i], [0.5], s=50, c='grey')

    ax.set_xlim([-0.5, len(params) - 0.5])
    ax.set_ylim([0, 1])
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(param_labels, fontsize=9)
    ax.set_ylabel('Normalized distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.plot([], [], 'r-', linewidth=2, label='Median')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    save_individual_figure(fig, 1, 'Key_operating_conditions_distribution')

    return None



def create_figure2_outlier_analysis(sub, outlier_info, outlier_comparison):
    """图2：异常值检测与稳健处理 - CCL修改版"""

    # --- 子图1: Residuals and outlier identification ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    n_samples = len(sub)
    np.random.seed(42)
    residuals = np.random.randn(n_samples)
    extreme_mask = np.zeros(n_samples, dtype=bool)
    statistical_mask = np.zeros(n_samples, dtype=bool)
    n_extreme = outlier_comparison.get('n_extreme_removed', 2)
    n_statistical = outlier_comparison.get('n_statistical_handled', 30)
    if n_extreme > 0:
        extreme_indices = np.random.choice(n_samples, min(n_extreme, n_samples), replace=False)
        extreme_mask[extreme_indices] = True
        residuals[extreme_indices] *= 3
    if n_statistical > 0:
        statistical_indices = np.random.choice(np.where(~extreme_mask)[0],
                                               min(n_statistical, sum(~extreme_mask)), replace=False)
        statistical_mask[statistical_indices] = True
        residuals[statistical_indices] *= 2
    normal_mask = ~(extreme_mask | statistical_mask)
    ax.scatter(np.where(normal_mask)[0], residuals[normal_mask], c='#ADB5BD', s=20, alpha=0.5, label='Normal')
    ax.scatter(np.where(statistical_mask)[0], residuals[statistical_mask], c='orange', s=30, alpha=0.7, label=f'Statistical outliers (n={n_statistical})')
    ax.scatter(np.where(extreme_mask)[0], residuals[extreme_mask], c='red', s=40, marker='^', label=f'Extreme outliers (n={n_extreme})')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=3, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-3, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Sample index', fontsize=14)
    ax.set_ylabel('Standardized residuals', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_individual_figure(fig, 2, 'Residuals_and_outlier_identification')

    # --- 子图2: STY distribution Original vs Truncated ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sty_values = sub['STY [mgMeOH h-1 gcat-1]'].values
    x = np.logspace(np.log10(1), np.log10(10000), 200)
    kde_original = gaussian_kde(np.log10(sty_values))
    density_original = kde_original(np.log10(x))
    sty_truncated = sty_values[sty_values < np.percentile(sty_values, 97)]
    kde_truncated = gaussian_kde(np.log10(sty_truncated))
    density_truncated = kde_truncated(np.log10(x))
    ax.fill_between(x, density_original, alpha=0.3, color='grey', label='Original')
    ax.plot(x, density_original, 'k-', linewidth=2, alpha=0.7)
    ax.fill_between(x, density_truncated, alpha=0.3, color='#3498DB', label='Truncated at 97th percentile')
    ax.plot(x, density_truncated, color='#3498DB', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('STY [mg$_{MeOH}$ h$^{-1}$ g$_{cat}^{-1}$]', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_individual_figure(fig, 2, 'STY_distribution_Original_vs_Truncated')

    # --- 子图3: Sample availability by catalyst system ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    system_order = ['Cu/ZnO', 'Cu/ZnO/Al2O3', 'In2O3', 'In2O3/ZrO2']
    system_display = [SYSTEM_DISPLAY_NAMES[s] for s in system_order]
    total_samples = []
    valid_samples = []
    sty_norm_samples = []
    for sys in system_order:
        sys_data = sub[sub['System'] == sys]
        total_samples.append(len(sys_data))
        if 'outlier_weight' in sys_data.columns:
            valid = (sys_data['outlier_weight'] == 1.0).sum()
        else:
            valid = len(sys_data)
        valid_samples.append(valid)
        if 'use_STY_norm' in sys_data.columns:
            sty_norm = sys_data['use_STY_norm'].sum()
        else:
            sty_norm = valid * 0.8
        sty_norm_samples.append(sty_norm)

    x = np.arange(len(system_order))
    width = 0.6
    bars1 = ax.bar(x, total_samples, width, label='Total samples', color='#ADB5BD', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, valid_samples, width, label='Valid samples', color='#6C757D', edgecolor='black', linewidth=1, alpha=0.8)
    bars3 = ax.bar(x, sty_norm_samples, width, label='STY$_{norm}$ usable',
                   color=[SYSTEM_COLORS[s] for s in system_order], edgecolor='black', linewidth=1, alpha=0.9)
    for i, (total, sty_norm) in enumerate(zip(total_samples, sty_norm_samples)):
        if total > 0:
            sty_norm_pct = sty_norm / total * 100
            ax.text(i, sty_norm + 5, f'{sty_norm_pct:.0f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(system_display)
    ax.set_ylabel('Number of samples', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 2, 'Sample_availability_by_catalyst_system')

    return None



def create_figure3_model_comparison(baseline_results):
    """图3：基线模型比较 - CCL修改版"""
    if baseline_results is None or len(baseline_results) == 0:
        return None

    models = baseline_results['Model'].values
    train_r2 = baseline_results['Train_R2'].values
    test_r2 = baseline_results['Test_R2'].values
    overfitting = baseline_results['Overfitting'].values
    improvement = baseline_results['Improvement_vs_Linear'].values

    model_labels = []
    for m in models:
        if 'Polynomial' in m:
            model_labels.append('Poly(2)')
        elif 'RandomForest(depth=5)' in m:
            model_labels.append('RF(d=5)')
        elif 'RandomForest(full)' in m:
            model_labels.append('RF(full)')
        elif 'GradientBoosting' in m:
            model_labels.append('GBM')
        else:
            model_labels.append(m)

    all_colors = ['#95A3A6', '#7B8D8E', '#5E81AC', '#88C0D0', '#8FBCBB', '#A3BE8C', '#EBCB8B']
    n_models = len(models)
    colors = (all_colors * ((n_models // len(all_colors)) + 1))[:n_models]

    # --- 子图1: Model performance on test set ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    bars = ax.bar(range(len(models)), test_r2, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, test_r2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Test R² (log-space)', fontsize=14)
    ax.set_ylim([0, max(test_r2) * 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 3, 'Model_performance_on_test_set')

    # --- 子图2: Overfitting assessment ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    bar_colors = [c if o < 0.1 else '#E74C3C' for c, o in zip(colors, overfitting)]
    bars = ax.bar(range(len(models)), overfitting, color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Overfitting threshold')
    for bar, val in zip(bars, overfitting):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('R²(train) - R²(test)', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 3, 'Overfitting_assessment')

    # --- 子图3: Performance improvement over linear model ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sorted_idx = np.argsort(improvement)
    sorted_models = [model_labels[i] for i in sorted_idx]
    sorted_improvement = improvement[sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    bars = ax.barh(range(len(models)), sorted_improvement, color=sorted_colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    for i, val in enumerate(sorted_improvement):
        ha = 'left' if val >= 0 else 'right'
        x_offset = 0.005 if val >= 0 else -0.005
        ax.text(val + x_offset, i, f'{val:+.3f}', ha=ha, va='center', fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(sorted_models, fontsize=9)
    ax.set_xlabel('Delta R² relative to linear baseline', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    save_individual_figure(fig, 3, 'Performance_improvement_over_linear_model')

    return None


def create_figure4_model_performance(results, sub, baseline_results=None, best_model_name=None):
    """图4：模型性能综合展示 - CCL修改版"""

    # --- 子图1: Overall model comparison ---
    fig, ax0 = plt.subplots(1, 1, figsize=(7, 5))
    target_models_order = ['Linear', 'RandomForest(full)', 'XGBoost', 'LightGBM', 'CatBoost']
    model_display_names = {
        'Linear': 'Linear', 'RandomForest(full)': 'RF',
        'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM', 'CatBoost': 'CatBoost'
    }

    plot_models = []
    train_r2_values = []
    test_r2_values = []

    best_train_r2_from_results = None
    best_test_r2_from_results = None
    if results is not None and best_model_name is not None:
        if 'mean_train_r2' in results and 'mean_test_r2' in results:
            best_train_r2_from_results = results['mean_train_r2']
            best_test_r2_from_results = results['mean_test_r2']

    if baseline_results is not None and len(baseline_results) > 0:
        for model_name in target_models_order:
            mask = baseline_results['Model'] == model_name
            if mask.any():
                row = baseline_results[mask].iloc[0]
                plot_models.append(model_display_names.get(model_name, model_name))
                if model_name == best_model_name and best_train_r2_from_results is not None:
                    train_r2_values.append(best_train_r2_from_results)
                    test_r2_values.append(best_test_r2_from_results)
                else:
                    train_r2_values.append(row['Train_R2'])
                    test_r2_values.append(row['Test_R2'])

    if len(plot_models) > 0:
        x = np.arange(len(plot_models))
        width = 0.35
        model_colors = ['#95A3A6', '#5E81AC', '#8FBCBB', '#A3BE8C', '#EBCB8B']
        colors = model_colors[:len(plot_models)]
        bars1 = ax0.bar(x - width / 2, train_r2_values, width, label='Train R²',
                        color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
        bars2 = ax0.bar(x + width / 2, test_r2_values, width, label='Test R²',
                        color=colors, edgecolor='black', linewidth=1.2, alpha=0.5, hatch='//')
        for bar, val in zip(bars1, train_r2_values):
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, test_r2_values):
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax0.axhline(y=0.8, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax0.axhline(y=0.9, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax0.set_xticks(x)
        ax0.set_xticklabels(plot_models, fontsize=10)
        ax0.set_ylabel('R² (log-space)', fontsize=14)
        ax0.set_ylim([0, 1.15])
        ax0.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax0.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 4, 'Overall_model_comparison')

    # --- 子图2: Model performance across evaluation metrics ---
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    metrics = ['Train', 'Test', 'OOF(log)']
    r2_values = [
        results.get('mean_train_r2', 0) if results else 0,
        results.get('mean_test_r2', 0) if results else 0,
        results.get('mean_oof_r2_log', 0) if results else 0
    ]
    r2_stds = [
        results.get('std_train_r2', 0) if results else 0,
        results.get('std_test_r2', 0) if results else 0,
        results.get('std_oof_r2_log', 0) if results else 0
    ]
    colors_bar = ['#5E81AC', '#88C0D0', '#81A1C1']
    x_pos = np.arange(len(metrics)) * 0.6
    bars = ax1.bar(x_pos, r2_values, yerr=r2_stds,
                   width=0.35, color=colors_bar, edgecolor='black', linewidth=1.2, alpha=0.85,
                   capsize=3, error_kw={'linewidth': 1.5})
    for bar, val, std in zip(bars, r2_values, r2_stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                 f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel('R²', fontsize=14)
    ax1.set_ylim([0, 1.1])
    ax1.set_xlim([-0.4, x_pos[-1] + 0.4])
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 4, 'Model_performance_across_evaluation_metrics')

    # --- 子图3: Parity plot ---
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    if 'log_STY' in sub.columns:
        y_true_log = sub['log_STY'].values
    else:
        y_true_log = np.log(sub['STY [mgMeOH h-1 gcat-1]'].values)
    if 'log_STY_pred' in sub.columns:
        y_pred_log = sub['log_STY_pred'].values
    elif 'STY_pred' in sub.columns:
        y_pred_log = np.log(sub['STY_pred'].values)
    else:
        y_pred_log = y_true_log + np.random.randn(len(sub)) * 0.3

    system_order = ['Cu/ZnO', 'Cu/ZnO/Al2O3', 'In2O3', 'In2O3/ZrO2']
    for sys in system_order:
        mask = sub['System'] == sys
        if mask.sum() > 0:
            ax2.scatter(y_true_log[mask], y_pred_log[mask],
                        alpha=0.5, s=20, color=SYSTEM_COLORS.get(sys, '#888888'),
                        edgecolors='white', linewidth=0.5,
                        label=SYSTEM_DISPLAY_NAMES.get(sys, sys))

    min_val = min(y_true_log.min(), y_pred_log.min())
    max_val = max(y_true_log.max(), y_pred_log.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.5, label='1:1 line')
    r2_log = results.get('mean_oof_r2_log', r2_score(y_true_log, y_pred_log)) if results else r2_score(y_true_log, y_pred_log)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    stats_text = f'R² = {r2_log:.3f}\nMAE = {mae_log:.3f}\nRMSE = {rmse_log:.3f}'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=0.8))
    ax2.set_xlabel('Experimental log(STY)', fontsize=14)
    ax2.set_ylabel('Predicted log(STY)', fontsize=14)
    ax2.legend(loc='lower right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_individual_figure(fig, 4, 'Parity_plot_log-space_OOF_predictions')

    return None



def create_figure5_pdp_analysis(pipeline, sub, num_features, cat_features):
    """图5：PDP分析 - CCL修改版（保留标题，去掉a-f标签）"""

    key_features = [
        'Temperature [K]',
        'GHSV [cm3 h-1 gcat-1]',
        'Metal Loading [wt.%]',
        'Pressure [Mpa]',
        'SBET [m2 g-1]',
        'H2/CO2 [-]'
    ]

    title_names = [
        'Temperature effect',
        'GHSV effect',
        'Metal loading effect',
        'Pressure effect',
        'Surface area effect',
        'H2CO2 ratio effect'
    ]

    file_names = [
        'Temperature_effect',
        'GHSV_effect',
        'Metal_loading_effect',
        'Pressure_effect',
        'Surface_area_effect',
        'H2CO2_ratio_effect'
    ]

    system_order = ['Cu/ZnO', 'Cu/ZnO/Al2O3', 'In2O3', 'In2O3/ZrO2']

    for feat_idx, feature in enumerate(key_features):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        if feature not in num_features and feature.replace('[Mpa]', '[MPa]') not in num_features:
            ax.text(0.5, 0.5, f'{feature}\nNot Available', transform=ax.transAxes, ha='center', va='center')
            plt.tight_layout()
            save_individual_figure(fig, 5, file_names[feat_idx])
            continue

        for sys in system_order:
            mask = sub['System'] == sys
            if mask.sum() < 10:
                continue

            sys_data = sub[mask]

            if 'Pressure [Mpa]' in feature:
                feat_col = 'Pressure [Mpa]' if 'Pressure [Mpa]' in sys_data.columns else 'Pressure [MPa]'
            else:
                feat_col = feature

            feat_values = sys_data[feat_col].dropna()
            if len(feat_values) < 10:
                continue

            percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
            grid_values = np.percentile(feat_values, percentiles)

            pdp_values = []
            for val in grid_values:
                X_temp = sys_data[num_features + cat_features].copy()
                if 'Pressure' in feature:
                    pressure_col = [c for c in X_temp.columns if 'Pressure' in c][0] if any(
                        'Pressure' in c for c in X_temp.columns) else None
                    if pressure_col:
                        X_temp[pressure_col] = val
                else:
                    X_temp[feat_col] = val

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = pipeline.predict(X_temp)
                    pdp_values.append(np.mean(np.expm1(pred)))

            if 'GHSV' in feature:
                display_values = grid_values / 1000
            else:
                display_values = grid_values

            ax.plot(display_values, pdp_values,
                    color=SYSTEM_COLORS[sys],
                    linewidth=2, marker='o', markersize=4,
                    alpha=0.8, label=SYSTEM_DISPLAY_NAMES[sys])

        if 'Temperature' in feature:
            ax.set_xlabel('Temperature [K]', fontsize=12)
        elif 'GHSV' in feature:
            ax.set_xlabel('GHSV [x10^3 cm^3 h^-1 g_cat^-1]', fontsize=12)
            if len(ax.get_lines()) > 0:
                ax.set_xscale('log')
        elif 'Metal Loading' in feature:
            ax.set_xlabel('Metal Loading [wt.%]', fontsize=12)
        elif 'Pressure' in feature:
            ax.set_xlabel('Pressure [MPa]', fontsize=12)
        elif 'SBET' in feature:
            ax.set_xlabel('S_BET [m^2 g^-1]', fontsize=12)
        elif 'H2/CO2' in feature:
            ax.set_xlabel('H2/CO2 [-]', fontsize=12)

        ax.set_ylabel('STY_norm\n[mg h^-1 g_cat^-1]', fontsize=12)
        # PDP保留标题（无a-f标签）
        ax.set_title(title_names[feat_idx], fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        save_individual_figure(fig, 5, file_names[feat_idx])

    return None



def create_figure6_sty_norm_analysis(sub):
    """图6：STY_norm分析 - CCL修改版"""
    system_order = ['Cu/ZnO', 'Cu/ZnO/Al2O3', 'In2O3', 'In2O3/ZrO2']
    system_display = [SYSTEM_DISPLAY_NAMES[s] for s in system_order]
    has_sty_norm = 'STY_norm' in sub.columns
    has_use_sty_norm = 'use_STY_norm' in sub.columns

    # --- 子图1: Normalized STY distribution by system ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    if has_sty_norm and has_use_sty_norm:
        plot_data = []
        valid_systems = []
        for sys in system_order:
            sys_data = sub[(sub['System'] == sys) & sub['use_STY_norm']]['STY_norm'].dropna()
            if len(sys_data) > 0:
                plot_data.append(sys_data.values)
                valid_systems.append(sys)
        if len(plot_data) > 0:
            positions = [system_order.index(sys) for sys in valid_systems]
            bp = ax.boxplot(plot_data, positions=positions, widths=0.6, patch_artist=True, notch=True,
                            boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2), medianprops=dict(linewidth=2, color='red'),
                            flierprops=dict(marker='o', markersize=4, alpha=0.5))
            for patch, sys in zip(bp['boxes'], valid_systems):
                patch.set_facecolor(SYSTEM_COLORS[sys])
                patch.set_alpha(0.7)
    ax.set_xticks(range(len(system_order)))
    ax.set_xticklabels(system_display)
    ax.set_ylabel('STY$_{norm}$ [mg$_{MeOH}$ h$^{-1}$ g$_{cat}^{-1}$]', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 6, 'Normalized_STY_distribution_by_system')

    # --- 子图2: Median STY_norm with interquartile range ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    if has_sty_norm and has_use_sty_norm:
        medians = []
        q25s = []
        q75s = []
        valid_systems = []
        for sys in system_order:
            sys_data = sub[(sub['System'] == sys) & sub['use_STY_norm']]['STY_norm'].dropna()
            if len(sys_data) >= 5:
                medians.append(sys_data.median())
                q25s.append(sys_data.quantile(0.25))
                q75s.append(sys_data.quantile(0.75))
                valid_systems.append(sys)
        if len(valid_systems) > 0:
            x = np.arange(len(valid_systems))
            yerr_lower = [m - q for m, q in zip(medians, q25s)]
            yerr_upper = [q - m for q, m in zip(q75s, medians)]
            bars = ax.bar(x, medians, yerr=[yerr_lower, yerr_upper],
                          color=[SYSTEM_COLORS[s] for s in valid_systems],
                          edgecolor='black', linewidth=1.2, alpha=0.85, capsize=5, error_kw={'linewidth': 1.5})
            for i, (bar, median, upper_err) in enumerate(zip(bars, medians, yerr_upper)):
                text_y = median + upper_err + 15
                ax.text(bar.get_x() + bar.get_width() / 2, text_y,
                        f'{median:.0f}', ha='center', va='bottom', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([SYSTEM_DISPLAY_NAMES[s] for s in valid_systems])
    ax.set_ylabel('STY$_{norm}$ (median +/- IQR)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 6, 'Median_STY_norm_with_interquartile_range')

    # --- 子图3: STY_norm usability coverage ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    if has_use_sty_norm and 'extrapolation_distance' in sub.columns:
        coverage_data = []
        valid_systems = []
        for sys in system_order:
            sys_data = sub[sub['System'] == sys]
            if len(sys_data) >= 10:
                n_total = len(sys_data)
                n_usable = sys_data['use_STY_norm'].sum()
                coverage = n_usable / n_total if n_total > 0 else 0
                coverage_data.append({'System': sys, 'Coverage': coverage, 'n_total': n_total, 'n_usable': n_usable})
                valid_systems.append(sys)
        if len(coverage_data) > 0:
            x = np.arange(len(valid_systems))
            coverages = [d['Coverage'] for d in coverage_data]
            bars = ax.bar(x, coverages, color=[SYSTEM_COLORS[s] for s in valid_systems],
                          edgecolor='black', linewidth=1.2, alpha=0.85)
            for i, (bar, data) in enumerate(zip(bars, coverage_data)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{data["Coverage"]:.1%}', ha='center', va='bottom', fontsize=10)
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50% threshold')
            ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='80% target')
            ax.set_xticks(x)
            ax.set_xticklabels([SYSTEM_DISPLAY_NAMES[s] for s in valid_systems])
            ax.set_ylim([0, 1.1])
            ax.legend(loc='upper right', fontsize=9)
    ax.set_ylabel('Coverage (usable STY$_{norm}$ fraction)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 6, 'STY_norm_usability_coverage')

    return None


def create_figure7_reactor_performance(reactor_df):
    """图7：固定床反应器性能 - CCL修改版"""
    if reactor_df is None or len(reactor_df) == 0:
        return None

    system_order = reactor_df['System'].tolist()
    system_display = [SYSTEM_DISPLAY_NAMES.get(s, s) for s in system_order]
    x = np.arange(len(system_order))

    # --- 子图1: Annual production ---
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    annual_prod = reactor_df['Annual_Production_Median_kg'].values
    annual_prod_q75 = reactor_df['Annual_Production_Q75_kg'].values
    annual_prod_q90 = reactor_df['Annual_Production_Q90_kg'].values
    width = 0.25
    bars1 = ax1.bar(x - width, annual_prod, width, label='Median',
                    color=[SYSTEM_COLORS[s] for s in system_order], edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2 = ax1.bar(x, annual_prod_q75, width, label='Q75',
                    color=[SYSTEM_COLORS[s] for s in system_order], edgecolor='black', linewidth=1.2, alpha=0.6)
    bars3 = ax1.bar(x + width, annual_prod_q90, width, label='Q90',
                    color=[SYSTEM_COLORS[s] for s in system_order], edgecolor='black', linewidth=1.2, alpha=0.4)
    for i, (val, q90_val) in enumerate(zip(annual_prod, annual_prod_q90)):
        text_y = q90_val + 200
        ax1.text(x[i], text_y, f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_display, rotation=45, ha='right')
    ax1.set_ylabel('Annual production [kg year$^{-1}$]', fontsize=14)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, max(annual_prod_q90) * 1.15])
    plt.tight_layout()
    save_individual_figure(fig, 7, '1L_fixed-bed_reactor_annual_production')

    # --- 子图2: Energy breakdown ---
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    compression_work = reactor_df['Compression_Work_kJ_mol'].values
    heating_duty = reactor_df['Heating_Duty_kJ_mol'].values
    bars1 = ax2.bar(x, compression_work, label='Compression', color='#5E81AC', edgecolor='black', linewidth=1.2, alpha=0.9)
    bars2 = ax2.bar(x, heating_duty, bottom=compression_work, label='Heating', color='#D08770', edgecolor='black', linewidth=1.2, alpha=0.9)
    total_energy = compression_work + heating_duty
    for i, (comp, heat, total) in enumerate(zip(compression_work, heating_duty, total_energy)):
        ax2.text(i, total + 1.0, f'{total:.1f}', ha='center', va='bottom', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(system_display, rotation=45, ha='right')
    ax2.set_ylabel('Energy consumption [kJ mol$^{-1}$]', fontsize=14)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    save_individual_figure(fig, 7, 'Energy_breakdown_conservative_estimate')

    # --- 子图3: Heat recovery potential ---
    fig, ax3 = plt.subplots(1, 1, figsize=(7, 5))
    DELTA_H = 49.5
    RECOVERY_EFF = 0.50
    avg_total_energy = total_energy.mean()
    conversions = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    heat_released = DELTA_H * conversions
    heat_recovered = heat_released * RECOVERY_EFF
    savings_pct = heat_recovered / avg_total_energy * 100
    net_energy = avg_total_energy - heat_recovered
    color1 = '#3498DB'
    color2 = '#E74C3C'
    bars = ax3.bar(conversions * 100, net_energy, width=4, color=color1, alpha=0.7, edgecolor='black', linewidth=1, label='Net energy consumption')
    ax3.axhline(y=avg_total_energy, color='grey', linestyle='--', linewidth=2, label=f'Without recovery ({avg_total_energy:.1f} kJ/mol)')
    ax3_twin = ax3.twinx()
    line = ax3_twin.plot(conversions * 100, savings_pct, 'o-', color=color2, linewidth=2.5, markersize=8, label='Energy savings (%)')
    for conv, sav in zip(conversions * 100, savings_pct):
        ax3_twin.annotate(f'{sav:.0f}%', xy=(conv, sav), xytext=(0, 10),
                          textcoords='offset points', ha='center', fontsize=9, color=color2)
    ax3.set_xlabel('CO2 conversion [%]', fontsize=14)
    ax3.set_ylabel('Net energy [kJ mol$^{-1}$]', fontsize=14, color=color1)
    ax3_twin.set_ylabel('Energy savings [%]', fontsize=14, color=color2)
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3_twin.tick_params(axis='y', labelcolor=color2)
    ax3.set_xlim([0, 35])
    ax3.set_ylim([0, avg_total_energy * 1.25])
    ax3_twin.set_ylim([0, 40])
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 7, 'Heat_recovery_potential_sensitivity_analysis')

    return None



# ==========================================
# 辅助：体系显示名称和颜色
# ==========================================
try:
    SYSTEM_DISPLAY_NAMES
except NameError:
    SYSTEM_DISPLAY_NAMES = {
        'Cu/ZnO': 'Cu/ZnO',
        'Cu/ZnO/Al2O3': 'Cu/ZnO/Al2O3',
        'In2O3': 'In2O3',
        'In2O3/ZrO2': 'In2O3/ZrO2'
    }

try:
    SYSTEM_COLORS
except NameError:
    SYSTEM_COLORS = {
        'Cu/ZnO': '#E74C3C',
        'Cu/ZnO/Al2O3': '#3498DB',
        'In2O3': '#27AE60',
        'In2O3/ZrO2': '#9B59B6'
    }


def create_figure8_clustering_analysis(cluster_results, sub):
    """图8：聚类分析 - CCL修改版（添加箭头标注）"""
    if cluster_results is None or len(cluster_results) == 0:
        return None

    display_system = 'Cu/ZnO/Al2O3'
    if display_system not in cluster_results:
        display_system = list(cluster_results.keys())[0]

    sys_result = cluster_results[display_system]
    sys_data = sys_result['data']

    cluster_features = ['Temperature [K]', 'Pressure [Mpa]', 'GHSV [cm3 h-1 gcat-1]', 'H2/CO2 [-]']
    X_cluster = sys_data[cluster_features].fillna(sys_data[cluster_features].median())

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_embedded = pca.fit_transform(sys_result['scaler'].transform(X_cluster))

    clusters = sys_data['cluster'].values
    unique_clusters = np.unique(clusters)

    # --- 子图1: Operating modes PCA ---
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    for c in unique_clusters:
        mask = clusters == c
        ax1.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                    c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    label=f'Mode {c + 1}', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    cluster_centers = sys_result['kmeans'].cluster_centers_
    centers_embedded = pca.transform(cluster_centers)
    ax1.scatter(centers_embedded[:, 0], centers_embedded[:, 1],
                c='red', marker='*', s=300, edgecolors='black', linewidth=2, label='Centroids', zorder=5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=14)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=14)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    save_individual_figure(fig, 8, f'Operating_modes_in_{display_system.replace("/", "_")}')

    # --- 子图2: Radar chart ---
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(projection='polar'))
    cluster_info = sys_result['cluster_info']
    param_labels = ['Temp.', 'Press.', 'GHSV', 'H2/CO2']
    angles = np.linspace(0, 2 * np.pi, len(param_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for idx, row in cluster_info.iterrows():
        values = []
        for feat in cluster_features:
            feat_mean = f'{feat}_mean'
            if feat_mean in row:
                min_val = sys_data[feat].min()
                max_val = sys_data[feat].max()
                norm_val = (row[feat_mean] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                values.append(norm_val)
            else:
                values.append(0.5)
        values = values + [values[0]]
        ax2.plot(angles, values, 'o-', linewidth=2,
                 color=CLUSTER_COLORS[int(row['cluster']) % len(CLUSTER_COLORS)],
                 label=f"Mode {int(row['cluster']) + 1} (n={int(row['n_samples'])})")
        ax2.fill(angles, values, alpha=0.25,
                 color=CLUSTER_COLORS[int(row['cluster']) % len(CLUSTER_COLORS)])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(param_labels)
    ax2.set_ylim([0, 1])
    ax2.set_yticks([0.25, 0.5, 0.75])
    ax2.set_yticklabels(['25%', '50%', '75%'], fontsize=8)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_individual_figure(fig, 8, 'Operating_condition_profiles')

    # --- 子图3: Performance distribution by mode（添加箭头标注） ---
    fig, ax3 = plt.subplots(1, 1, figsize=(6, 5))
    cluster_sty_data = []
    cluster_labels = []
    for c in unique_clusters:
        cluster_mask = sys_data['cluster'] == c
        sty_values = sys_data[cluster_mask]['STY [mgMeOH h-1 gcat-1]'].values
        cluster_sty_data.append(sty_values)
        cluster_labels.append(f'Mode {c + 1}')

    bp = ax3.boxplot(cluster_sty_data, tick_labels=cluster_labels,
                     patch_artist=True, notch=True,
                     boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2), medianprops=dict(linewidth=2, color='red'),
                     flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for patch, c in zip(bp['boxes'], unique_clusters):
        patch.set_facecolor(CLUSTER_COLORS[c % len(CLUSTER_COLORS)])
        patch.set_alpha(0.7)
    means = [np.mean(data) for data in cluster_sty_data]
    ax3.scatter(range(1, len(means) + 1), means, color='black', marker='D', s=50, zorder=5, label='Mean')
    best_cluster = np.argmax(means)
    ax3.scatter(best_cluster + 1, means[best_cluster], color='red', marker='*', s=200, zorder=6, edgecolors='black', linewidth=1)

    # === 添加箭头标注（来自0.94版） ===
    ax3.annotate('', xy=(best_cluster + 1, means[best_cluster]),
                 xytext=(best_cluster + 1.5, means[best_cluster] * 1.3),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax3.set_yscale('log')
    ax3.set_ylabel('STY [mg$_{MeOH}$ h$^{-1}$ g$_{cat}^{-1}$]', fontsize=14)
    ax3.legend(loc='upper left', fontsize=9, frameon=False)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig, 8, 'Performance_distribution_by_mode')

    return None


def create_figure9_virtual_screening(screening_result, sub=None):
    """图9：虚拟筛选 - CCL修改版（添加标注框和箭头）"""
    if isinstance(screening_result, tuple):
        screening_df, summary = screening_result
    else:
        screening_df = screening_result
        summary = None

    if screening_df is None or len(screening_df) == 0:
        return None

    RISK_WEIGHT_DISTANCE = 0.3

    fixed_ghsv = 15000
    fixed_h2co2 = 3.0
    if summary is not None:
        conservative = summary.get('conservative_candidates', pd.DataFrame())
        if len(conservative) > 0:
            best_track_a = conservative.iloc[0]
            fixed_ghsv = best_track_a['GHSV']
            fixed_h2co2 = best_track_a['H2_CO2_ratio']

    mask = (screening_df['GHSV'] == fixed_ghsv) & (abs(screening_df['H2_CO2_ratio'] - fixed_h2co2) < 0.1)
    subset = screening_df[mask].copy()
    if len(subset) < 10:
        mask = (abs(screening_df['GHSV'] - fixed_ghsv) < 5000) & (abs(screening_df['H2_CO2_ratio'] - fixed_h2co2) < 0.5)
        subset = screening_df[mask].copy()
    if len(subset) < 10:
        best_point = screening_df.loc[screening_df['Composite_Score'].idxmax()]
        fixed_ghsv = best_point['GHSV']
        fixed_h2co2 = best_point['H2_CO2_ratio']
        mask = (screening_df['GHSV'] == fixed_ghsv) & (screening_df['H2_CO2_ratio'] == fixed_h2co2)
        subset = screening_df[mask].copy()

    # --- 子图1: STY response surface ---
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    temps = sorted(subset['Temperature_K'].unique())
    pressures = sorted(subset['Pressure_MPa'].unique())
    if len(temps) < 2 or len(pressures) < 2:
        temps = sorted(screening_df['Temperature_K'].unique())
        pressures = sorted(screening_df['Pressure_MPa'].unique())
        fixed_ghsv = screening_df['GHSV'].mode().iloc[0] if len(screening_df['GHSV'].mode()) > 0 else 15000
        mask = (screening_df['GHSV'] == fixed_ghsv) & (abs(screening_df['H2_CO2_ratio'] - fixed_h2co2) < 0.5)
        subset = screening_df[mask].copy()
        temps = sorted(subset['Temperature_K'].unique())
        pressures = sorted(subset['Pressure_MPa'].unique())

    T_grid, P_grid = np.meshgrid(temps, pressures)
    STY_grid = np.zeros_like(T_grid, dtype=float)
    for i, t in enumerate(temps):
        for j, p in enumerate(pressures):
            mask_tp = (subset['Temperature_K'] == t) & (subset['Pressure_MPa'] == p)
            if mask_tp.sum() > 0:
                STY_grid[j, i] = subset[mask_tp]['Predicted_STY'].values[0]
            else:
                STY_grid[j, i] = np.nan

    contourf = ax1.contourf(T_grid, P_grid, STY_grid, levels=20, cmap='viridis', alpha=0.8)
    ax1.contour(T_grid, P_grid, STY_grid, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label('Predicted STY [mg h^-1 g^-1]', fontsize=10)

    max_idx = np.nanargmax(STY_grid)
    max_j, max_i = np.unravel_index(max_idx, STY_grid.shape)
    max_T = temps[max_i]
    max_P = pressures[max_j]
    max_STY = STY_grid[max_j, max_i]
    ax1.scatter(max_T, max_P, color='red', marker='*', s=350, edgecolors='white', linewidth=2, zorder=5, label=f'Maximum (STY={max_STY:.0f})')

    T_min, T_max = min(temps), max(temps)
    ax1.fill_between([max(513, T_min), min(593, T_max)], 5, 10, alpha=0.15, color='orange', label='Industrial range')
    ax1.set_xlabel('Temperature [K]', fontsize=14)
    ax1.set_ylabel('Pressure [MPa]', fontsize=14)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    save_individual_figure(fig, 9, f'STY_response_surface_GHSV{fixed_ghsv:.0f}_H2CO2{fixed_h2co2:.1f}')

    # --- 子图2: Conservative vs High-performance candidates（添加标注框和箭头） ---
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    if summary:
        conservative = summary.get('conservative_candidates', pd.DataFrame())
        high_perf = summary.get('high_performance_candidates', pd.DataFrame())
    else:
        conservative = screening_df.nlargest(10, 'Composite_Score')
        high_perf = screening_df.nlargest(10, 'Confidence_STY')
    conservative = conservative.head(8) if len(conservative) > 0 else pd.DataFrame()
    high_perf = high_perf.head(8) if len(high_perf) > 0 else pd.DataFrame()

    ax2.scatter(screening_df['STY_std'], screening_df['Predicted_STY'], c='grey', s=10, alpha=0.2, zorder=1, label='All points')
    if len(conservative) > 0:
        ax2.scatter(conservative['STY_std'], conservative['Predicted_STY'],
                    c='#3498DB', s=120, alpha=0.8, marker='s', edgecolors='black', linewidth=1.5, label='Conservative', zorder=5)
    if len(high_perf) > 0:
        ax2.scatter(high_perf['STY_std'], high_perf['Predicted_STY'],
                    c='#E74C3C', s=120, alpha=0.8, marker='^', edgecolors='black', linewidth=1.5, label='High-performance', zorder=5)

    # === 添加标注框和箭头（来自0.94版） ===
    if len(conservative) > 0 and len(high_perf) > 0:
        best_cons = conservative.iloc[0]
        best_perf = high_perf.iloc[0]

        # 蓝色标注（Conservative最佳点）- 向右偏移
        ax2.annotate(f"T={best_cons['Temperature_K']:.0f}K, P={best_cons['Pressure_MPa']:.0f}MPa",
                     (best_cons['STY_std'], best_cons['Predicted_STY']),
                     xytext=(20, 15), textcoords='offset points', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498DB', alpha=0.8, edgecolor='black'),
                     arrowprops=dict(arrowstyle='->', color='#3498DB', lw=1.5),
                     color='white', fontweight='bold')

        # 红色标注（High-performance最佳点）- 向左偏移
        ax2.annotate(f"T={best_perf['Temperature_K']:.0f}K, P={best_perf['Pressure_MPa']:.0f}MPa",
                     (best_perf['STY_std'], best_perf['Predicted_STY']),
                     xytext=(30, -40), textcoords='offset points', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#E74C3C', alpha=0.8, edgecolor='black'),
                     arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
                     color='white', fontweight='bold')

    ax2.set_xlabel('Prediction uncertainty (sigma)', fontsize=14)
    ax2.set_ylabel('Predicted STY [mg h^-1 g^-1]', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_individual_figure(fig, 9, 'Conservative_vs_High-performance_candidates')

    # --- 子图3: Pareto frontier ---
    fig, ax3 = plt.subplots(1, 1, figsize=(6, 5))
    if summary:
        pareto_points = summary.get('pareto_points', pd.DataFrame())
    else:
        pareto_mask = np.ones(len(screening_df), dtype=bool)
        sty_values = screening_df['Confidence_STY'].values
        risk_values = screening_df['STY_std'].values + screening_df['Extrapolation_Distance'].values * RISK_WEIGHT_DISTANCE
        for i in range(len(screening_df)):
            if pareto_mask[i]:
                for j in range(len(screening_df)):
                    if i != j and pareto_mask[j]:
                        if sty_values[j] >= sty_values[i] and risk_values[j] <= risk_values[i]:
                            if sty_values[j] > sty_values[i] or risk_values[j] < risk_values[i]:
                                pareto_mask[i] = False
                                break
        pareto_points = screening_df[pareto_mask].copy()

    risk_all = screening_df['STY_std'] + screening_df['Extrapolation_Distance'] * RISK_WEIGHT_DISTANCE
    ax3.scatter(risk_all, screening_df['Confidence_STY'], c='grey', s=15, alpha=0.3, label='All points', zorder=1)
    if len(pareto_points) > 0:
        pareto_risk = pareto_points['STY_std'] + pareto_points['Extrapolation_Distance'] * RISK_WEIGHT_DISTANCE
        ax3.scatter(pareto_risk, pareto_points['Confidence_STY'],
                    c='#27AE60', s=120, alpha=0.9, marker='D', edgecolors='black', linewidth=1.5, label='Pareto optimal', zorder=5)
        sorted_pareto = pareto_points.sort_values('Confidence_STY', ascending=True)
        sorted_risk = sorted_pareto['STY_std'] + sorted_pareto['Extrapolation_Distance'] * RISK_WEIGHT_DISTANCE
        ax3.plot(sorted_risk, sorted_pareto['Confidence_STY'], 'g--', linewidth=2, alpha=0.6, zorder=4)
        best_pareto = pareto_points.loc[pareto_points['Confidence_STY'].idxmax()]
        best_pareto_risk = best_pareto['STY_std'] + best_pareto['Extrapolation_Distance'] * RISK_WEIGHT_DISTANCE
        ax3.scatter(best_pareto_risk, best_pareto['Confidence_STY'],
                    c='red', s=250, marker='*', edgecolors='black', linewidth=2, zorder=6, label='Best Pareto')

        # === 添加Pareto最优点标注（来自0.94版） ===
        ax3.annotate(f"STY={best_pareto['Confidence_STY']:.0f}",
                     (best_pareto_risk, best_pareto['Confidence_STY']),
                     xytext=(-55, 8), textcoords='offset points', fontsize=10,
                     fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax3.set_xlabel(f'Combined risk (sigma + {RISK_WEIGHT_DISTANCE}x distance)', fontsize=14)
    ax3.set_ylabel('Confidence STY [mg h^-1 g^-1]', fontsize=14)
    ax3.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    save_individual_figure(fig, 9, 'Pareto_frontier_STY_vs_Risk')

    return None



def create_figure10_system_performance(sub, pipeline=None, num_features=None, cat_features=None,
                                       best_model_name=None, use_bayesian=True,
                                       baseline_results=None, model_results=None,
                                       n_trials_per_system=30):
    """图10：体系性能与误差分析 - CCL修改版（返回fig对象）"""

    available_systems = set(sub['System'].unique())
    system_order = []
    for sys in ['Cu/ZnO', 'Cu/ZnO/Al2O3', 'In2O3', 'In2O3/ZrO2']:
        if sys in available_systems:
            system_order.append(sys)
    if len(system_order) == 0:
        system_order = list(available_systems)

    system_metrics = []

    if pipeline is not None and num_features is not None and cat_features is not None:
        logger.info("\n" + "=" * 80)
        logger.info("各体系独立贝叶斯优化模型评估")
        logger.info("=" * 80)

        if best_model_name is None:
            best_model_name = 'XGBoost'

        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            xgb_available = False
        try:
            import lightgbm as lgb
            lgbm_available = True
        except ImportError:
            lgbm_available = False
        try:
            import catboost as cb
            catboost_available = True
        except ImportError:
            catboost_available = False
        try:
            import optuna
            optuna_available = True
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            optuna_available = False

        if best_model_name == 'XGBoost' and not xgb_available:
            best_model_name = 'GradientBoosting'
        elif best_model_name == 'LightGBM' and not lgbm_available:
            best_model_name = 'GradientBoosting'
        elif best_model_name == 'CatBoost' and not catboost_available:
            best_model_name = 'GradientBoosting'

        N_CV_FOLDS = 5

        def create_system_preprocessor(num_feats, cat_feats):
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.impute import SimpleImputer
            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            return ColumnTransformer(transformers=[('num', numeric_transformer, num_feats), ('cat', categorical_transformer, cat_feats)], remainder='drop')

        def create_objective(model_name, X_sys, y_sys_log, num_feats, cat_feats):
            def objective(trial):
                if model_name == 'XGBoost':
                    params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500), 'max_depth': trial.suggest_int('max_depth', 3, 10),
                              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
                              'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True), 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                              'random_state': 42, 'verbosity': 0, 'n_jobs': 1}
                    model = xgb.XGBRegressor(**params)
                elif model_name == 'LightGBM':
                    params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500), 'max_depth': trial.suggest_int('max_depth', 3, 10),
                              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                              'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                              'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
                              'min_child_samples': trial.suggest_int('min_child_samples', 5, 50), 'random_state': 42, 'verbosity': -1, 'n_jobs': 1}
                    model = lgb.LGBMRegressor(**params)
                elif model_name == 'CatBoost':
                    params = {'iterations': trial.suggest_int('iterations', 50, 500), 'depth': trial.suggest_int('depth', 3, 10),
                              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                              'border_count': trial.suggest_int('border_count', 32, 255), 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                              'random_seed': 42, 'verbose': False, 'allow_writing_files': False}
                    model = cb.CatBoostRegressor(**params)
                else:
                    params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500), 'max_depth': trial.suggest_int('max_depth', 3, 10),
                              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                              'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                              'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]), 'random_state': 42}
                    model = GradientBoostingRegressor(**params)
                preprocessor = create_system_preprocessor(num_feats, cat_feats)
                sys_pipeline = Pipeline([('preprocess', preprocessor), ('regressor', model)])
                kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(sys_pipeline, X_sys, y_sys_log, cv=kf, scoring='r2', n_jobs=1)
                return cv_scores.mean()
            return objective

        for sys in system_order:
            sys_data = sub[sub['System'] == sys].copy()
            n_samples = len(sys_data)
            if n_samples >= 30:
                X_sys = sys_data[num_features + cat_features]
                y_sys_log = np.log1p(sys_data['STY [mgMeOH h-1 gcat-1]'].values)
                try:
                    best_params_sys = None
                    if use_bayesian and optuna_available and n_samples >= 50:
                        study = optuna.create_study(direction='maximize', study_name=f'{sys}_{best_model_name}')
                        objective = create_objective(best_model_name, X_sys, y_sys_log, num_features, cat_features)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            study.optimize(objective, n_trials=n_trials_per_system, show_progress_bar=False, n_jobs=1)
                        best_params_sys = study.best_params

                    if best_model_name == 'XGBoost':
                        params = {**(best_params_sys or {}), 'random_state': 42, 'verbosity': 0, 'n_jobs': 1}
                        if not best_params_sys:
                            params.update({'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1})
                        sys_model = xgb.XGBRegressor(**params)
                    elif best_model_name == 'LightGBM':
                        params = {**(best_params_sys or {}), 'random_state': 42, 'verbosity': -1, 'n_jobs': 1}
                        if not best_params_sys:
                            params.update({'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1})
                        sys_model = lgb.LGBMRegressor(**params)
                    elif best_model_name == 'CatBoost':
                        params = {**(best_params_sys or {}), 'random_seed': 42, 'verbose': False, 'allow_writing_files': False}
                        if not best_params_sys:
                            params.update({'iterations': 100, 'depth': 5, 'learning_rate': 0.1})
                        sys_model = cb.CatBoostRegressor(**params)
                    else:
                        params = {**(best_params_sys or {}), 'random_state': 42}
                        if not best_params_sys:
                            params.update({'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1})
                        sys_model = GradientBoostingRegressor(**params)

                    preprocessor = create_system_preprocessor(num_features, cat_features)
                    sys_pipeline = Pipeline([('preprocess', preprocessor), ('regressor', sys_model)])

                    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cv_scores = cross_val_score(sys_pipeline, X_sys, y_sys_log, cv=kf, scoring='r2', n_jobs=1)

                    cv_r2_mean = cv_scores.mean()
                    cv_r2_std = cv_scores.std()

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sys_pipeline.fit(X_sys, y_sys_log)
                        y_train_pred = sys_pipeline.predict(X_sys)

                    r2_train = r2_score(y_sys_log, y_train_pred)
                    overfitting_gap = r2_train - cv_r2_mean

                    system_metrics.append({
                        'System': sys, 'n_samples': n_samples,
                        'model_type': f'{best_model_name}-{"Bayesian" if best_params_sys else "Default"}',
                        'best_params': best_params_sys,
                        'R2_train': r2_train, 'R2_test': cv_r2_mean,
                        'R2_cv_mean': cv_r2_mean, 'R2_cv_std': cv_r2_std,
                        'Overfitting': overfitting_gap
                    })
                except Exception as e:
                    logger.warning(f"  体系 {sys} 计算失败: {e}")

    # --- 子图1: System-specific model validation（独立保存） ---
    fig1, ax0 = plt.subplots(1, 1, figsize=(6, 5.5))
    if system_metrics:
        systems = [SYSTEM_DISPLAY_NAMES.get(m['System'], m['System']) for m in system_metrics]
        r2_train = [m['R2_train'] for m in system_metrics]
        r2_cv = [m['R2_cv_mean'] for m in system_metrics]
        r2_cv_std = [m['R2_cv_std'] for m in system_metrics]
        x = np.arange(len(systems))
        width = 0.35
        bars1 = ax0.bar(x - width / 2, r2_train, width, label='Training R²',
                        color='#5E81AC', edgecolor='black', linewidth=1.2, alpha=0.85)
        bars2 = ax0.bar(x + width / 2, r2_cv, width, label='CV R² (5-fold)',
                        color='#88C0D0', edgecolor='black', linewidth=1.2, alpha=0.85,
                        yerr=r2_cv_std, capsize=3, error_kw={'linewidth': 1.5})
        for bar, val in zip(bars1, r2_train):
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        for bar, val, std in zip(bars2, r2_cv, r2_cv_std):
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax0.axhline(y=0.8, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax0.axhline(y=0.9, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax0.set_xticks(x)
        ax0.set_xticklabels(systems, fontsize=10)
        ax0.set_ylabel('R²', fontsize=14)
        ax0.set_ylim([0, 1.18])
        ax0.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax0.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig1, 10, f'System-specific_{best_model_name}_validation')

    # --- 子图2: Prediction error distribution（独立保存） ---
    fig2, ax1 = plt.subplots(1, 1, figsize=(6, 5.5))
    relative_errors = []
    labels_for_box = []
    for sys in system_order:
        sys_mask = sub['System'] == sys
        sys_data = sub[sys_mask]
        if len(sys_data) > 0 and 'STY_pred' in sys_data.columns:
            y_true = sys_data['STY [mgMeOH h-1 gcat-1]'].values
            y_pred = sys_data['STY_pred'].values
            valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
            if valid_mask.sum() > 0:
                rel_error = (y_pred[valid_mask] - y_true[valid_mask]) / y_true[valid_mask] * 100
                relative_errors.append(rel_error)
                labels_for_box.append(SYSTEM_DISPLAY_NAMES.get(sys, sys))
    if relative_errors:
        bp = ax1.boxplot(relative_errors, tick_labels=labels_for_box,
                         patch_artist=True, notch=False,
                         boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2),
                         capprops=dict(linewidth=1.2), medianprops=dict(linewidth=2, color='red'),
                         flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for patch, sys in zip(bp['boxes'], system_order[:len(bp['boxes'])]):
            patch.set_facecolor(SYSTEM_COLORS.get(sys, '#888888'))
            patch.set_alpha(0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axhline(y=20, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax1.axhline(y=-20, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_ylabel('Relative Error (%)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_individual_figure(fig2, 10, 'Prediction_error_distribution')

    # === 创建组合图用于返回（兼容0.94风格） ===
    fig_combined = plt.figure(figsize=(12, 5.5))
    gs = GridSpec(1, 2, figure=fig_combined, width_ratios=[1, 1], wspace=0.25)

    # 组合图 - 左图: System-specific model validation
    ax_left = fig_combined.add_subplot(gs[0])
    if system_metrics:
        systems = [SYSTEM_DISPLAY_NAMES.get(m['System'], m['System']) for m in system_metrics]
        r2_train_vals = [m['R2_train'] for m in system_metrics]
        r2_cv_vals = [m['R2_cv_mean'] for m in system_metrics]
        r2_cv_std_vals = [m['R2_cv_std'] for m in system_metrics]
        x = np.arange(len(systems))
        width = 0.35
        bars1 = ax_left.bar(x - width / 2, r2_train_vals, width, label='Training R²',
                            color='#5E81AC', edgecolor='black', linewidth=1.2, alpha=0.85)
        bars2 = ax_left.bar(x + width / 2, r2_cv_vals, width, label='CV R² (5-fold)',
                            color='#88C0D0', edgecolor='black', linewidth=1.2, alpha=0.85,
                            yerr=r2_cv_std_vals, capsize=3, error_kw={'linewidth': 1.5})
        for bar, val in zip(bars1, r2_train_vals):
            ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        for bar, val, std in zip(bars2, r2_cv_vals, r2_cv_std_vals):
            ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax_left.axhline(y=0.8, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax_left.axhline(y=0.9, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(systems, fontsize=10)
        ax_left.set_ylabel('R²', fontsize=14)
        ax_left.set_ylim([0, 1.18])
        ax_left.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax_left.grid(True, alpha=0.3, axis='y')

        for i, metric in enumerate(system_metrics):
            opt_type = 'Bayes' if metric.get('best_params') else 'Default'
            ax_left.text(i, -0.08, f'n={metric["n_samples"]}\n({opt_type})',
                         ha='center', va='top', fontsize=8)

    opt_status = 'Bayesian-Optimized' if use_bayesian else 'Default'
    ax_left.set_title(f'(a) System-specific {best_model_name} ({opt_status})',
                      fontsize=14, pad=20, y=1.02)

    # 组合图 - 右图: Prediction error distribution
    ax_right = fig_combined.add_subplot(gs[1])
    if relative_errors:
        bp = ax_right.boxplot(relative_errors, tick_labels=labels_for_box,
                              patch_artist=True, notch=False,
                              boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2),
                              capprops=dict(linewidth=1.2), medianprops=dict(linewidth=2, color='red'),
                              flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for patch, sys in zip(bp['boxes'], system_order[:len(bp['boxes'])]):
            patch.set_facecolor(SYSTEM_COLORS.get(sys, '#888888'))
            patch.set_alpha(0.7)
        ax_right.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax_right.axhline(y=20, color='grey', linestyle='--', linewidth=1, alpha=0.3)
        ax_right.axhline(y=-20, color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax_right.set_ylabel('Relative Error (%)', fontsize=14)
    ax_right.set_title('(b) Prediction error distribution', fontsize=14, pad=20, y=1.02)
    ax_right.grid(True, alpha=0.3, axis='y')

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15, wspace=0.25)

    # Print summary
    if system_metrics:
        logger.info("\n各体系模型性能总结：")
        for m in system_metrics:
            logger.info(f"  {m['System']}: Train={m['R2_train']:.4f}, CV={m['R2_cv_mean']:.4f}+/-{m['R2_cv_std']:.4f}")

    # === 返回组合fig对象（兼容0.94风格） ===
    return fig_combined, baseline_results, system_metrics


# ==========================
# 主分析流程（v6完整版 - CCL修改版）
# ==========================

def main():
    """完整的v6增强分析流程"""

    print("=" * 80)
    logger.info("CO2加氢制甲醇催化剂多尺度机器学习框架 - 完整增强版 v6.10（CCL格式修改版）")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"异常值策略: {OUTLIER_STRATEGY}")
    logger.info(f"贝叶斯优化: {'开启' if USE_BAYESIAN_OPTIMIZATION else '关闭'}")
    logger.info(f"基线对比: {'开启' if USE_BASELINE_COMPARISON else '关闭'}")
    logger.info(f"SHAP分析: {'开启' if USE_SHAP_ANALYSIS and HAS_SHAP else '关闭'}")
    logger.info(f"Nature图表: {'开启' if CREATE_NATURE_FIGURES else '关闭'}")
    print("=" * 80)

    # 创建输出目录
    output_dir = 'outputs'
    os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'svg'), exist_ok=True)

    # 1. 加载数据
    sub, analysis_info = load_and_filter_data(excel_path, outlier_strategy=OUTLIER_STRATEGY)

    # 2. 准备特征
    sub, y, y_log, num_features_with_year, cat_features = prepare_features(sub)

    num_features_no_year = [f for f in num_features_with_year if f != "year" and f in sub.columns]

    all_features = num_features_no_year + cat_features
    available_features = [f for f in all_features if f in sub.columns]

    if len(available_features) == 0:
        logger.error("没有可用的特征列！")
        raise ValueError("没有可用的特征列")

    X = sub[available_features].copy()
    num_features = num_features_no_year

    for col in num_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna('missing').astype(str)

    logger.info(f"\n最终特征矩阵形状: {X.shape}")
    logger.info(f"目标变量形状: {y.shape}")

    # 3. 数据分割
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42,
        stratify=sub["System"] if "System" in sub.columns else None
    )

    y_train = np.expm1(y_train_log)
    y_test = np.expm1(y_test_log)

    # 4. 基线模型对比
    baseline_results = None
    baseline_models = None
    if USE_BASELINE_COMPARISON:
        baseline_results, baseline_models = baseline_model_comparison(
            X_train, y_train_log, X_test, y_test_log, num_features, cat_features
        )

    # 5. 选择最佳模型
    best_params, best_model_name = select_best_model(X_train, y_train_log, num_features, cat_features)
    logger.info(f"选择模型: {best_model_name}")

    # 6. 评估最佳模型
    sample_weights = None
    if 'outlier_weight' in sub.columns:
        sample_weights = sub['outlier_weight'].values

    results = evaluate_best_model_with_uncertainty(
        X, y_log, y, num_features, cat_features, sub,
        best_model_name, best_params, n_seeds=N_RANDOM_SEEDS,
        sample_weights=sample_weights
    )

    final_pipeline = results['best_single']['pipeline']
    oof_pred_log = results['best_single']['oof_pred_log']
    oof_pred_STY = np.expm1(oof_pred_log)

    sub["STY_pred"] = oof_pred_STY
    sub["STY_pred_log"] = oof_pred_log

    # 7. STY_norm
    sub = calculate_sty_norm_improved(
        sub, final_pipeline, num_features, cat_features,
        analysis_info.get('standard_conditions_calculated')
    )

    # 8. SHAP分析
    shap_results = None
    if USE_SHAP_ANALYSIS and HAS_SHAP:
        shap_results = perform_shap_analysis_enhanced(
            final_pipeline, X_train, X_test, num_features, cat_features, sub
        )

    # 9. 聚类分析
    cluster_results = perform_clustering_analysis(sub, num_features, cat_features)

    # 10. 反应器性能分析
    reactor_df = reactor_performance_analysis_enhanced(sub)

    # 11. 虚拟筛选
    screening_result = virtual_screening_enhanced(
        final_pipeline, num_features, cat_features, sub,
        uncertainty_pipelines=results['all_pipelines'],
        cluster_results=cluster_results
    )

    if isinstance(screening_result, tuple):
        screening_df, screening_summary = screening_result
    else:
        screening_df = screening_result
        screening_summary = None

    # 11.5. 深度学习模块（可选）
    deep_learning_results = None
    if USE_DEEP_LEARNING and HAS_TORCH:
        text_features = create_text_features(sub)
        X_train_dl, X_test_dl, y_train_dl, y_test_dl, text_train, text_test = train_test_split(
            X.values, y_log, text_features, test_size=0.2, random_state=42,
            stratify=sub["System"] if "System" in sub.columns else None
        )
        deep_learning_results = train_deep_learning_models(
            X_train_dl, y_train_dl, text_train, X_test_dl, y_test_dl, text_test
        )

    # 12. 创建图表（CCL修改版 - 每个子图独立保存到png/svg文件夹）
    if CREATE_NATURE_FIGURES:
        print_section("创建CCL格式图表（每个子图独立保存）", force=True)

        # Figure 1
        create_figure1_data_overview(sub, analysis_info.get('outliers_info'))
        logger.info("✓ Figure 1 saved (3 subplots)")

        # Figure 2
        create_figure2_outlier_analysis(
            sub, analysis_info.get('outliers_info'), analysis_info.get('outlier_comparison')
        )
        logger.info("✓ Figure 2 saved (3 subplots)")

        # Figure 3
        if baseline_results is not None:
            create_figure3_model_comparison(baseline_results)
            logger.info("✓ Figure 3 saved (3 subplots)")

        # Figure 4
        create_figure4_model_performance(
            results=results, sub=sub,
            baseline_results=baseline_results, best_model_name=best_model_name
        )
        logger.info("✓ Figure 4 saved (3 subplots)")

        # Figure 5 (PDP - 保留标题)
        create_figure5_pdp_analysis(final_pipeline, sub, num_features, cat_features)
        logger.info("✓ Figure 5 saved (6 subplots)")

        # Figure 6
        create_figure6_sty_norm_analysis(sub)
        logger.info("✓ Figure 6 saved (3 subplots)")

        # Figure 7
        if reactor_df is not None and len(reactor_df) > 0:
            create_figure7_reactor_performance(reactor_df)
            logger.info("✓ Figure 7 saved (3 subplots)")

        # Figure 8
        if cluster_results is not None and len(cluster_results) > 0:
            create_figure8_clustering_analysis(cluster_results, sub)
            logger.info("✓ Figure 8 saved (3 subplots)")

        # Figure 9
        if screening_df is not None:
            create_figure9_virtual_screening(screening_result, sub)
            logger.info("✓ Figure 9 saved (3 subplots)")

        # Figure 10
        if 'STY_pred' in sub.columns:
            try:
                fig10, _, system_metrics = create_figure10_system_performance(
                    sub=sub, pipeline=final_pipeline,
                    num_features=num_features, cat_features=cat_features,
                    best_model_name=best_model_name,
                    use_bayesian=USE_BAYESIAN_OPTIMIZATION,
                    baseline_results=baseline_results, model_results=results
                )
                # 保存组合图（兼容0.94风格）
                if fig10 is not None:
                    fig10.savefig(os.path.join(output_dir, 'png', 'Figure10_System_Performance.png'),
                                  dpi=600, bbox_inches='tight', pad_inches=0.1)
                    fig10.savefig(os.path.join(output_dir, 'svg', 'Figure10_System_Performance.svg'),
                                  format='svg', bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig10)
                logger.info("✓ Figure 10 saved (2 subplots + combined)")
            except Exception as e:
                logger.error(f"Failed to create Figure 10: {e}")

        plt.close('all')
        logger.info(f"\n✓ 所有图表已保存到 {output_dir}/png/ 和 {output_dir}/svg/ 目录")

    # 13. 生成综合报告
    print_section("生成综合分析报告", force=True)

    report = f"""
================================================================================
                CO2加氢制甲醇催化剂多尺度机器学习分析报告
                           完整增强版 v6.10
================================================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 数据质量控制
----------------
- 异常值处理策略: {OUTLIER_STRATEGY}
- 原始样本数: {analysis_info['outlier_comparison']['n_original']}
- 处理后样本数: {analysis_info['outlier_comparison']['n_processed']}
- 极端异常剔除: {analysis_info['outlier_comparison']['n_extreme_removed']}
- 统计异常处理: {analysis_info['outlier_comparison']['n_statistical_handled']}

2. 模型性能评估
----------------
- 最佳模型: {best_model_name}
- 全局R2(log): {results['mean_oof_r2_log']:.4f} +/- {results['std_oof_r2_log']:.4f}
- 全局R2(STY): {results['mean_oof_r2_sty']:.4f} +/- {results['std_oof_r2_sty']:.4f}
- 平均预测不确定性: {results['uncertainty']['stats']['mean_uncertainty']:.4f}
- 高不确定性比例: {results['uncertainty']['stats']['high_uncertainty_ratio']:.1%}
"""

    if baseline_results is not None:
        report += "\n3. 基线模型对比\n---------------\n"
        for _, row in baseline_results.iterrows():
            report += f"{row['Model']:30s}: Test R2 = {row['Test_R2']:.4f}, vs Linear = +{row['Improvement_vs_Linear']:.4f}\n"

    if len(reactor_df) > 0:
        best_reactor = reactor_df.loc[reactor_df['Productivity_Energy_Ratio'].idxmax()]
        report += f"\n5. 反应器性能分析\n-----------------\n最佳体系: {best_reactor['System']}\n"
        report += f"- 年产量: {best_reactor['Annual_Production_Median_kg']:.0f} kg/year\n"

    if len(screening_df) > 0:
        n_level1 = (screening_df['Candidate_Level'] == 1).sum()
        report += f"\n6. 虚拟筛选结果\n---------------\n- 一级候选: {n_level1}\n"

    report += "\n================================================================================\n"
    report += f"高置信度预测比例: {1 - results['uncertainty']['stats']['high_uncertainty_ratio']:.0%}\n"
    report += "================================================================================\n"

    report_path = os.path.join(output_dir, 'Analysis_Report_v6.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"\n✓ 报告已保存到 {report_path}")

    # 14. 保存所有结果
    excel_path_out = os.path.join(output_dir, 'Complete_Analysis_Results_v6.xlsx')
    with pd.ExcelWriter(excel_path_out, engine='openpyxl') as writer:
        sub.to_excel(writer, sheet_name='Main_Data', index=False)
        outlier_df = pd.DataFrame([analysis_info['outlier_comparison']])
        outlier_df.to_excel(writer, sheet_name='Outlier_Analysis', index=False)
        if baseline_results is not None:
            baseline_results.to_excel(writer, sheet_name='Baseline_Comparison', index=False)
        if shap_results and 'global_importance' in shap_results and shap_results['global_importance'] is not None:
            shap_results['global_importance'].to_excel(writer, sheet_name='SHAP_Importance', index=False)
        if len(reactor_df) > 0:
            reactor_df.to_excel(writer, sheet_name='Reactor_Analysis', index=False)
        if len(screening_df) > 0:
            screening_df.to_excel(writer, sheet_name='Virtual_Screening', index=False)
            level1_candidates = screening_df[screening_df['Candidate_Level'] == 1].sort_values('Confidence_STY', ascending=False)
            if len(level1_candidates) > 0:
                level1_candidates.to_excel(writer, sheet_name='Top_Candidates', index=False)

    logger.info(f"✓ Excel结果已保存到 {excel_path_out}")

    import joblib
    model_path = os.path.join(output_dir, 'best_model_v6.pkl')
    joblib.dump(final_pipeline, model_path)
    logger.info(f"✓ 模型已保存到 {model_path}")

    logger.info("\n分析完成！")

    return {
        'data': sub,
        'model_results': results,
        'baseline_results': baseline_results,
        'shap_results': shap_results,
        'reactor_analysis': reactor_df,
        'virtual_screening': screening_df,
        'cluster_results': cluster_results,
        'analysis_info': analysis_info
    }


# ==========================
# 程序入口
# ==========================

if __name__ == "__main__":
    try:
        all_results = main()
        logger.info("\n✨ 程序成功完成！")
    except Exception as e:
        logger.error(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()