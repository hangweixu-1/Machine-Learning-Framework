# Machine Learning Framework for CO₂ Hydrogenation Catalyst Benchmarking

A data-driven framework for standardized benchmarking and industrial projection of thermocatalytic CO₂-to-methanol catalysts.

## Overview

Comparing catalyst performance across studies is difficult due to vast heterogeneity in experimental conditions. This framework addresses this challenge by integrating LLM-assisted literature mining, machine learning prediction, and a novel normalized space-time yield (STY_norm) methodology to enable fair cross-system comparison under family-specific representative operating windows.

## Key Features

- **LLM-Assisted Data Extraction** — Automated extraction of 26 variables from 180 publications using DeepSeek-R1 (F1 = 0.907)
- **XGBoost Prediction Model** — Accurate STY prediction (test R² = 0.901) with SHAP-based interpretability
- **Normalized STY (STY_norm)** — Standardized benchmarking under family-specific reference conditions for Cu/ZnO, Cu/ZnO/Al₂O₃, In₂O₃, and In₂O₃/ZrO₂
- **Clustering Analysis** — K-means identification of typical operating modes and underexplored high-performance windows
- **Dual-Track Virtual Screening** — Conservative (Track A) and aggressive (Track B) optimization pathways with explicit risk–performance tradeoffs
- **Reactor-Level Modeling** — Fixed-bed reactor simulation translating normalized performance to industrial productivity and energy metrics

## License

This project is licensed under the MIT License.
