"""路线三：WLS 机理估计全流程编排（报告 §4）。

流程概览
--------
    1. 从 data_dir/N（正常）、F03/F04/F06（异常）加载 CSV
    2. 按控制压差分组（0.8, 1.4, 1.65 MPa 典型工况）
    3. 对三个机理模型分别构造因变量:
       - M1 (泵压差模型): y = 泵压差
       - M2 (板换+供回水压差模型): y = 板换压差 + 供回水压差
       - M3 (系统阻抗模型): y = 板换压差 + 过滤器压差
    4. 对每个模型×控制压差组合执行 WLS 五步拟合
    5. 基于正常样本分位数计算多级阈值
    6. 标准化残差评分

三类机理模型（报告 §4）:
    M1: 泵压差 ~ (speed/100)²
        泵压差与转速平方成正比（离心泵相似定律）
    M2: (板换+供回水压差) ~ (speed/100)²
        板换压差和供回水压差之和与转速平方成正比
    M3: (板换+过滤器压差) ~ (speed/100)²
        系统总阻抗（板换+过滤器）与转速平方成正比
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data.io import read_csv
from data.labels import parse_label
from evaluation.metrics import compute_sample_metrics
from utils.config_loader import ensure_dir
from utils.logging_config import setup_logging
from utils.helpers import normalize_speed
from models.wls.wls_fit import fit_wls


class WLSPipeline:
    """二次侧 WLS 机理异常检测全流程编排器（报告 §4）。

    流程：加载 CSV → 构造因变量 → 按控制压差分组
          → WLS 拟合 → 标准化残差评分 → 阈值判定 → 指标计算

    三类模型（报告 §4）:
        M1: 泵压差 ~ (speed/100)²
        M2: (板换+供回水压差) ~ (speed/100)²
        M3: (板换+过滤器压差) ~ (speed/100)²

    属性:
        config (dict): 全量配置字典。
        MODELS (dict): 模型定义，含 name 和 target_cols。
    """

    # 模型定义（报告 §4 三类机理模型）
    MODELS = {
        "M1": {
            "name": "泵压差模型",
            "target_cols": ["二次侧泵压差"],
        },
        "M2": {
            "name": "板换+供回水压差模型",
            "target_cols": ["二次侧板换压差", "二次侧供回水压差"],
        },
        "M3": {
            "name": "系统阻抗模型（板换+过滤器压差）",
            "target_cols": ["二次侧板换压差", "二次侧过滤器压差"],
        },
    }

    def __init__(self, config: dict):
        """初始化 WLS Pipeline。

        参数:
            config (dict): 配置字典，关键参数（报告 §4）:
                - control_pressures (list[float]): 控制压差列表（默认 [0.8, 1.4, 1.65]），
                  对应泵系统典型工况点。
                - wls_min_speed (float): 转速过滤下限（默认 50，即 50%转速）。
                - wls_max_speed (float): 转速过滤上限（默认 95，即 95%转速）。
                - wls_bins (int): WLS 分箱数（默认 10）。
                - threshold_quantiles (list[float]): 阈值分位数列表，
                  默认 [0.95, 0.975, 0.99, 0.9975, 0.9999]。
                - n_repeats (int): 重复实验次数。
        """
        self.config = config

    def run(self, data_dir: str, output_root: str = None) -> Dict:
        """执行 WLS 机理异常检测（报告 §4 完整流程）。

        处理流程:
            1. 加载数据: N/正常 + F03/F04/F06异常 → 转速过滤 + 控制压差分组
            2. 对三个机理模型 (M1/M2/M3) 分别训练
            3. 对每个控制压差水平独立拟合 WLS
            4. 基于正常样本标准化残差分位数计算多级阈值

        参数:
            data_dir (str): 数据根目录（需含 N/、F03/、F04/、F06/ 子目录）。
                N/: 正常数据 (label=0)
                F03/F04/: 二次侧异常数据 (label=1)
                F06/: 工况异常数据 (label=1)
            output_root (str, optional): 输出目录，默认 "./output/wls"。

        返回:
            dict: {results, output_dir, config}
                results 按 model_id → control_pressure 组织。
        """
        if output_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root = os.path.join("./output/wls", f"report_{timestamp}")
        output_root = ensure_dir(output_root)
        logger = setup_logging(output_root)

        # 读取配置参数
        control_pressures = self.config.get("control_pressures", [0.8, 1.4, 1.65])
        wls_min_speed = self.config.get("wls_min_speed", 50)
        wls_max_speed = self.config.get("wls_max_speed", 95)
        wls_bins = self.config.get("wls_bins", 10)
        min_sigma = float(self.config.get("min_sigma", 1e-6))
        threshold_qs = self.config.get("threshold_quantiles", [0.95, 0.975, 0.99, 0.9975, 0.9999])

        logger.info(
            f"开始 WLS 估计: control_pressures={control_pressures}, "
            f"speed=[{wls_min_speed},{wls_max_speed}), bins={wls_bins}"
        )

        # Step 1: 加载并预处理数据
        data = self._load_data(data_dir, wls_min_speed, wls_max_speed)
        if data is None or data.empty:
            return {"error": "数据加载失败"}

        # Step 2: 对每个模型 × 控制压差组合训练
        all_results = {}
        for model_id, model_def in self.MODELS.items():
            logger.info(f"训练 {model_def['name']}...")
            model_results = []

            for cp in control_pressures:
                # 筛选当前控制压差的数据
                df_cp = data[data["control_pressure"] == cp]
                if len(df_cp) < 20:
                    continue

                # Step 2a: 构造因变量 y（报告 §4）
                target_cols = model_def["target_cols"]
                y = self._build_target(df_cp, target_cols)
                if y is None:
                    continue

                # Step 2b: 构造自变量 x = (speed/100)²（报告 §4）
                x = df_cp["speed_sq"].to_numpy(dtype=float)

                # Step 2c: WLS 五步拟合（报告 §4）
                result = fit_wls(x, y, n_bins=wls_bins, min_sigma=min_sigma)
                scores = np.array(result["scores"])

                # Step 2d: 仅用正常样本计算阈值（报告 §4）
                normal_mask = df_cp.get("_label", 0) == 0
                normal_scores = scores[normal_mask]

                thresholds = {}
                for q in threshold_qs:
                    thresholds[f"q{q}"] = float(np.nanquantile(normal_scores, q))

                model_results.append({
                    "control_pressure": cp,
                    "beta": result["beta"],          # [β₀, β₁]
                    "sigma_per_bin": result["sigma_per_bin"],
                    "bin_edges": result["bin_edges"],
                    "thresholds": thresholds,         # {q0.95: v1, q0.975: v2, ...}
                    "normal_score_mean": float(np.nanmean(normal_scores)),
                })

            all_results[model_id] = model_results

        logger.info(f"WLS 估计完成: {len(all_results)} 个模型")

        # Step 3: 对每个模型 × 控制压差组合打分并生成预测标签
        all_metrics = {}
        for model_id, model_def in self.MODELS.items():
            model_metrics = []
            for result_entry in all_results.get(model_id, []):
                cp = result_entry["control_pressure"]
                beta = result_entry["beta"]
                thresholds = result_entry["thresholds"]

                df_cp = data[data["control_pressure"] == cp].copy()
                if df_cp.empty:
                    continue

                # 构造自变量 x，预测 y_hat
                x = df_cp["speed_sq"].to_numpy(dtype=float)
                y = self._build_target(df_cp, model_def["target_cols"])
                if y is None:
                    continue
                y_hat = beta[0] + beta[1] * x
                residuals = y - y_hat

                # 与阈值口径保持一致：阈值来自标准化残差，因此评价时也必须使用标准化残差，
                # 不能用原始 |residual| 去和标准化阈值比较。
                sigma_per_bin = np.asarray(result_entry["sigma_per_bin"], dtype=float)
                bin_edges = np.asarray(result_entry["bin_edges"], dtype=float)
                bin_idx = np.digitize(x, bin_edges) - 1
                bin_idx = np.clip(bin_idx, 0, len(sigma_per_bin) - 1)
                local_sigma = sigma_per_bin[bin_idx]
                scores = np.abs(residuals) / np.maximum(local_sigma, min_sigma)

                # 用默认主阈值生成预测标签（优先使用 q0.9975）
                main_q = threshold_qs[-2] if len(threshold_qs) >= 2 else threshold_qs[-1]
                thr = thresholds.get(f"q{main_q}", list(thresholds.values())[0] if thresholds else None)
                if thr is not None:
                    y_pred = (scores >= thr).astype(int)
                    y_true = df_cp.get("_label", pd.Series(0, index=df_cp.index)).to_numpy(dtype=int)
                    metrics = compute_sample_metrics(y_true, y_pred)
                else:
                    metrics = {}

                model_metrics.append({
                    "control_pressure": cp,
                    "beta_0": beta[0],
                    "beta_1": beta[1],
                    "threshold": thr,
                    **metrics,
                })

            all_metrics[model_id] = model_metrics

        return {
            "results": all_results,
            "metrics": all_metrics,
            "output_dir": output_root,
            "config": {
                "control_pressures": control_pressures,
                "wls_bins": wls_bins,
                "threshold_quantiles": threshold_qs,
            },
        }

    def _load_data(
        self, data_dir: str, min_speed: float, max_speed: float
    ) -> pd.DataFrame:
        """加载并预处理二次侧数据（报告 §4 数据准备）。

        处理步骤:
            1. 从 N/F03/F04/F06 文件夹加载 CSV
            2. 列名解析（schema.resolve_column 匹配泵转速、控制压差）
            3. 转速归一化: 若 95%分位 > 2 → 量纲为 0-100，除以 100
            4. 构造 speed_sq = (speed/100)²
            5. 转速区间过滤 [min_speed, max_speed)
            6. 控制压差四舍五入到最近有效值，差值 < 0.2 保留

        参数:
            data_dir (str): 数据根目录。
            min_speed (float): 转速过滤下限（%）。
            max_speed (float): 转速过滤上限（%）。

        返回:
            pd.DataFrame: 带 speed_sq、control_pressure、_label 列的合并数据。
        """
        root = Path(data_dir)
        # 文件夹 → 标签映射
        folders = {"N": 0, "F03": 1, "F04": 1, "F06": 1}
        frames = []

        for folder_name, label in folders.items():
            folder_path = root / folder_name
            if not folder_path.exists():
                continue
            for fp in folder_path.glob("*.csv"):
                df = read_csv(str(fp))
                if df is None:
                    continue

                # 列名解析
                from data.schema import resolve_column

                col_speed = resolve_column(df.columns, ["二次侧泵转速", "二次侧泵转速1"])
                col_control = resolve_column(df.columns, ["控制压差目标值", "二次侧控制压差目标值"])

                if not col_speed or not col_control:
                    continue

                speed_raw = pd.to_numeric(df[col_speed], errors="coerce")
                control = pd.to_numeric(df[col_control], errors="coerce")

                # 转速归一化（智能量纲检测: 95分位 > 2 说明量纲为 0-100）
                speed_95 = speed_raw.quantile(0.95)
                if speed_95 > 2:
                    speed_norm = speed_raw / 100.0  # 0-100 → 0-1
                else:
                    speed_norm = speed_raw           # 已是 0-1

                # 转速区间过滤（报告 §4: 排除低转速不稳定区和高转速非线性区）
                speed_mask = (speed_norm * 100 >= min_speed) & (speed_norm * 100 < max_speed)

                df["speed_norm"] = speed_norm
                df["speed_sq"] = speed_norm**2     # 自变量 x = (speed/100)²（报告 §4）
                df["control_pressure"] = control
                df["_label"] = label
                df["_folder"] = folder_name

                df = df[speed_mask]
                if len(df) > 0:
                    frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # 控制压差四舍五入到最近有效值（处理传感器测量误差）
        valid_cps = self.config.get("control_pressures", [0.8, 1.4, 1.65])
        combined["control_pressure_rounded"] = combined["control_pressure"].apply(
            lambda x: min(valid_cps, key=lambda v: abs(v - x)) if pd.notna(x) else np.nan
        )
        # 只保留与有效值偏差 < 0.2 的数据（排除传感器故障导致的异常读数）
        combined["cp_diff"] = np.abs(combined["control_pressure"] - combined["control_pressure_rounded"])
        combined = combined[combined["cp_diff"] < 0.2].copy()
        # 后续建模使用离散控制压差组，避免 0.8000000119 之类浮点误差导致分组为空。
        combined["control_pressure"] = combined["control_pressure_rounded"]

        return combined

    def _build_target(self, df: pd.DataFrame, target_cols: list) -> np.ndarray:
        """构造 WLS 模型的因变量 y（报告 §4）。

        若 target_cols 为单列则直接取值，若多列则求和。
        例如 M2: y = 板换压差 + 供回水压差。

        参数:
            df (pd.DataFrame): 输入 DataFrame。
            target_cols (list[str]): 目标列名列表。

        返回:
            ndarray or None: 因变量数组，若缺少必需列则返回 None。
        """
        from data.schema import resolve_column

        values = []
        for tc in target_cols:
            col = resolve_column(df.columns, [tc])
            if col:
                values.append(pd.to_numeric(df[col], errors="coerce"))
            else:
                return None

        if len(values) == 1:
            return values[0].to_numpy(dtype=float)
        else:
            return np.column_stack(values).sum(axis=1)
