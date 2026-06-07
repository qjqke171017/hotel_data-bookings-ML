"""路线三：WLS 机理估计全流程编排（报告 §4）。

本版本包含两类关键修复：
1. WLS 评分口径修复：阈值和测试都使用 score = |残差| / 局部 sigma；
2. 内存修复：加载阶段只保留 WLS 必需列，并在单文件内完成转速和控制压差过滤，避免
   25M×37 全量宽表在 ``combined.copy()`` 时触发数 GB 级内存申请。
"""

import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.metrics import compute_sample_metrics
from utils.config_loader import ensure_dir
from utils.logging_config import setup_logging
from models.wls.wls_fit import fit_wls, score_wls


class WLSPipeline:
    """二次侧 WLS 机理异常检测全流程编排器。"""

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

    # 只读取 WLS 真正需要的列，避免把 30+ 列宽表整体拼接进内存。
    _ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    _SPEED_CANDIDATES = ["二次侧泵转速", "二次侧泵转速1", "二次侧泵转速2"]
    _CONTROL_CANDIDATES = ["控制压差目标值", "二次侧控制压差目标值"]
    _TARGET_CANDIDATES = {
        "二次侧泵压差": ["二次侧泵压差"],
        "二次侧板换压差": ["二次侧板换压差", "二次侧板换压差1", "二次侧板换压差2"],
        "二次侧供回水压差": ["二次侧供回水压差"],
        "二次侧过滤器压差": ["二次侧过滤器压差"],
    }

    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def _resolve_column(columns, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        return None

    def _read_required_csv(self, fp: Path) -> Optional[pd.DataFrame]:
        """只读取 WLS 所需列。失败返回 None。"""
        last_err = None
        for enc in self._ENCODINGS:
            try:
                header = list(pd.read_csv(fp, encoding=enc, nrows=0).columns)
                col_speed = self._resolve_column(header, self._SPEED_CANDIDATES)
                col_control = self._resolve_column(header, self._CONTROL_CANDIDATES)
                if not col_speed or not col_control:
                    return None

                logical_to_actual = {
                    "__speed_raw__": col_speed,
                    "__control_raw__": col_control,
                }
                for logical, candidates in self._TARGET_CANDIDATES.items():
                    actual = self._resolve_column(header, candidates)
                    if actual:
                        logical_to_actual[logical] = actual

                usecols = list(dict.fromkeys(logical_to_actual.values()))
                raw = pd.read_csv(fp, encoding=enc, usecols=usecols)

                out = pd.DataFrame(index=raw.index)
                out["__speed_raw__"] = pd.to_numeric(raw[logical_to_actual["__speed_raw__"]], errors="coerce")
                out["__control_raw__"] = pd.to_numeric(raw[logical_to_actual["__control_raw__"]], errors="coerce")
                for logical in self._TARGET_CANDIDATES:
                    actual = logical_to_actual.get(logical)
                    if actual:
                        out[logical] = pd.to_numeric(raw[actual], errors="coerce")
                return out
            except Exception as e:
                last_err = e
                continue
        print(f"[WARN] CSV读取失败：{fp}；最后错误：{last_err}")
        return None

    def run(self, data_dir: str, output_root: str = None) -> Dict:
        """执行 WLS 机理异常检测。"""
        if output_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root = os.path.join("./output/wls", f"report_{timestamp}")
        output_root = ensure_dir(output_root)
        logger = setup_logging(output_root)

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

        data = self._load_data(data_dir, wls_min_speed, wls_max_speed)
        if data is None or data.empty:
            return {"error": "数据加载失败"}

        logger.info(f"WLS 纳入记录数: {len(data):,}, 列数: {len(data.columns)}")

        all_results = {}
        for model_id, model_def in self.MODELS.items():
            logger.info(f"训练 {model_def['name']}...")
            model_results = []

            for cp in control_pressures:
                cp_mask = np.isclose(data["control_pressure"].to_numpy(dtype=float), float(cp), atol=1e-9)
                if int(cp_mask.sum()) < 20:
                    logger.warning(f"{model_id}, control_pressure={cp}: 样本数过少，跳过")
                    continue
                df_cp = data.loc[cp_mask]

                y = self._build_target(df_cp, model_def["target_cols"])
                if y is None:
                    logger.warning(f"{model_id}, control_pressure={cp}: 缺少因变量列，跳过")
                    continue

                x = df_cp["speed_sq"].to_numpy(dtype=float)
                try:
                    result = fit_wls(x, y, n_bins=wls_bins, min_sigma=min_sigma, return_full=True)
                except Exception as e:
                    logger.warning(f"{model_id}, control_pressure={cp}: WLS拟合失败：{e}")
                    continue

                scores = np.asarray(result["scores"], dtype=float)
                normal_mask = df_cp["_label"].to_numpy(dtype=int) == 0
                normal_scores = scores[normal_mask & np.isfinite(scores)]
                if len(normal_scores) == 0:
                    logger.warning(f"{model_id}, control_pressure={cp}: 正常样本有效score为空，跳过")
                    continue

                thresholds = {f"q{q}": float(np.nanquantile(normal_scores, q)) for q in threshold_qs}

                model_results.append({
                    "control_pressure": float(cp),
                    "beta": result["beta"],
                    "sigma_per_bin": result["sigma_per_bin"],
                    "bin_edges": result["bin_edges"],
                    "thresholds": thresholds,
                    "normal_score_mean": float(np.nanmean(normal_scores)),
                    "n_records": int(len(df_cp)),
                    "n_normal_scores": int(len(normal_scores)),
                })

                # 释放当前控制压差的大数组引用，降低峰值内存。
                del result, scores, normal_scores, x, y

            all_results[model_id] = model_results

        logger.info(f"WLS 估计完成: {len(all_results)} 个模型")

        all_metrics = {}
        for model_id, model_def in self.MODELS.items():
            model_metrics = []
            for result_entry in all_results.get(model_id, []):
                cp = result_entry["control_pressure"]
                cp_mask = np.isclose(data["control_pressure"].to_numpy(dtype=float), float(cp), atol=1e-9)
                if int(cp_mask.sum()) == 0:
                    continue
                df_cp = data.loc[cp_mask]

                x = df_cp["speed_sq"].to_numpy(dtype=float)
                y = self._build_target(df_cp, model_def["target_cols"])
                if y is None:
                    continue

                scores = score_wls(
                    x=x,
                    y=y,
                    beta=np.asarray(result_entry["beta"], dtype=float),
                    sigma_per_bin=np.asarray(result_entry["sigma_per_bin"], dtype=float),
                    bin_edges=np.asarray(result_entry["bin_edges"], dtype=float),
                    min_sigma=min_sigma,
                )

                main_q = 0.9975 if 0.9975 in threshold_qs else (threshold_qs[-2] if len(threshold_qs) >= 2 else threshold_qs[-1])
                thresholds = result_entry["thresholds"]
                thr = thresholds.get(f"q{main_q}", list(thresholds.values())[0] if thresholds else None)
                if thr is not None:
                    valid_eval = np.isfinite(scores)
                    y_pred = (scores[valid_eval] >= thr).astype(int)
                    y_true = df_cp.loc[valid_eval, "_label"].to_numpy(dtype=int)
                    metrics = compute_sample_metrics(y_true, y_pred)
                else:
                    metrics = {}

                beta = result_entry["beta"]
                model_metrics.append({
                    "control_pressure": cp,
                    "beta_0": float(beta[0]),
                    "beta_1": float(beta[1]),
                    "threshold": float(thr) if thr is not None else np.nan,
                    **metrics,
                })

                del x, y, scores

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

    def _load_data(self, data_dir: str, min_speed: float, max_speed: float) -> pd.DataFrame:
        """加载并预处理二次侧数据。

        与旧版不同：旧版先把所有原始列拼接成 25M×37 的大宽表，再做 copy；
        本函数在每个 CSV 内部先筛转速和控制压差，并只保留 WLS 所需列。
        """
        root = Path(data_dir)
        folders = {"N": 0, "F03": 1, "F04": 1, "F06": 1}
        valid_cps = np.asarray(self.config.get("control_pressures", [0.8, 1.4, 1.65]), dtype=np.float64)
        frames = []
        file_count = 0
        kept_count = 0
        max_rows_per_file = self.config.get("max_rows_per_file", None)

        for folder_name, label in folders.items():
            folder_path = root / folder_name
            if not folder_path.exists():
                continue
            for fp in sorted(folder_path.glob("*.csv")):
                file_count += 1
                df = self._read_required_csv(fp)
                if df is None or df.empty:
                    continue

                speed_raw = df["__speed_raw__"]
                control = df["__control_raw__"]

                speed_95 = speed_raw.dropna().quantile(0.95) if speed_raw.notna().any() else np.nan
                if pd.notna(speed_95) and speed_95 > 2:
                    speed_norm = speed_raw / 100.0
                else:
                    speed_norm = speed_raw

                speed_mask = (speed_norm * 100 >= min_speed) & (speed_norm * 100 < max_speed)

                ctrl = control.to_numpy(dtype=np.float64)
                rounded = np.full(len(ctrl), np.nan, dtype=np.float64)
                ok_ctrl = np.isfinite(ctrl)
                if ok_ctrl.any():
                    nearest_idx = np.argmin(np.abs(ctrl[ok_ctrl, None] - valid_cps[None, :]), axis=1)
                    rounded[ok_ctrl] = valid_cps[nearest_idx]
                cp_diff = np.abs(ctrl - rounded)
                cp_mask = np.isfinite(cp_diff) & (cp_diff < 0.2)

                keep = speed_mask.to_numpy(dtype=bool) & cp_mask
                keep_idx = np.flatnonzero(keep)
                if len(keep_idx) == 0:
                    continue

                # 与原独立脚本保持一致：若配置 max_rows_per_file，则每个 CSV 在过滤后均匀抽样，
                # 避免全量 2500 万记录一次性进入内存。
                if max_rows_per_file is not None:
                    try:
                        cap = int(max_rows_per_file)
                    except Exception:
                        cap = 0
                    if cap > 0 and len(keep_idx) > cap:
                        seed = int(hashlib.md5(str(fp).encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
                        rng = np.random.default_rng(seed)
                        keep_idx = np.sort(rng.choice(keep_idx, size=cap, replace=False))

                out = pd.DataFrame(index=np.arange(len(keep_idx)))
                out["speed_norm"] = speed_norm.to_numpy(dtype=np.float32)[keep_idx]
                out["speed_sq"] = (out["speed_norm"].to_numpy(dtype=np.float32) ** 2).astype(np.float32)
                # control_pressure 使用 float64，保证与配置中的 0.8/1.4/1.65 精确分组稳定。
                out["control_pressure"] = rounded[keep_idx]
                out["_label"] = np.int8(label)
                out["_folder"] = folder_name

                for logical in self._TARGET_CANDIDATES:
                    if logical in df.columns:
                        out[logical] = df[logical].to_numpy(dtype=np.float32)[keep_idx]
                    else:
                        out[logical] = np.nan

                frames.append(out)
                kept_count += len(out)

                # 尽快释放原始文件级 DataFrame。
                del df, out

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, copy=False)
        combined["_folder"] = combined["_folder"].astype("category")
        combined["_label"] = combined["_label"].astype(np.int8)

        print(f"[INFO] WLS读取完成：CSV文件 {file_count} 个，过滤后记录 {kept_count:,} 条，保留列 {len(combined.columns)} 列")
        return combined

    def _build_target(self, df: pd.DataFrame, target_cols: list) -> Optional[np.ndarray]:
        """构造 WLS 模型因变量 y。"""
        values = []
        for tc in target_cols:
            if tc not in df.columns:
                return None
            values.append(pd.to_numeric(df[tc], errors="coerce").to_numpy(dtype=float))

        if len(values) == 1:
            return values[0]
        return np.column_stack(values).sum(axis=1)
