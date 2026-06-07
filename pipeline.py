"""路线三：二次侧 WLS 机理异常检测全流程。

本版 WLS Pipeline 按旧版 ``wls_secondary_multi_repeat_speed50_95.py`` 的实验协议修复：
1. 按正常设备做 ``normal_train / normal_heldout`` 划分，而不是行级混合评价；
2. 每个 repeat、每个模型、每个控制压差组只用 ``normal_train`` 拟合；
3. 训练记录按“设备×控制压差”限额抽样，避免长序列设备支配斜率；
4. 阈值来自对应控制压差组的 ``normal_train`` 标准化残差分位数；
5. 指标只保留 heldout 口径，即 ``normal_heldout + anomaly``，不再输出 all_labeled；
6. ``evaluation_report_metrics.csv`` 仍沿用项目现有扁平指标字段，但对多次重复取均值；
   复杂的 repeat 细节、阈值、拟合参数和数据统计写入最终 JSON。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.config_loader import ensure_dir
from utils.logging_config import setup_logging
from models.wls.wls_fit import fit_wls, predict_wls


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

    SECONDARY_ANOMALY_CODES = {"F03", "F04", "F06"}
    PRIMARY_ANOMALY_CODES = {"F01", "F02"}
    UNLABELED_CODES = {"M", "S", "U"}
    IGNORE_DIR_NAMES = {"正常数据分类分工况"}

    def __init__(self, config: dict):
        self.config = config or {}
        model_cfg = self.config.get("models")
        if isinstance(model_cfg, dict) and model_cfg:
            self.MODELS = {
                str(mid): {
                    "name": cfg.get("name", str(mid)),
                    "target_cols": cfg.get("target_columns") or cfg.get("target_cols") or [],
                }
                for mid, cfg in model_cfg.items()
            }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, data_dir: str, output_root: Optional[str] = None) -> Dict:
        if output_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root = os.path.join("./output/wls", f"report_{timestamp}")
        output_root = ensure_dir(output_root)
        logger = setup_logging(output_root)

        control_pressures = self._float_list(self.config.get("control_pressures", [0.8, 1.4, 1.65]))
        control_round_digits = self._as_int(self.config.get("control_round_digits", 3), 3)
        threshold_qs = self._float_list(self.config.get("threshold_quantiles", [0.95, 0.975, 0.99, 0.9975, 0.9999]))
        main_q = self._as_float(self.config.get("main_threshold_quantile", 0.9975), 0.9975)
        if main_q not in threshold_qs:
            main_q = threshold_qs[-2] if len(threshold_qs) >= 2 else threshold_qs[-1]

        random_state = self._as_int(self.config.get("random_state", 42), 42)
        n_repeats = self._as_int(self.config.get("n_repeats", 20), 20)
        train_frac = self._as_float(self.config.get("train_normal_device_frac", 0.70), 0.70)
        wls_min_speed = self._as_float(self.config.get("wls_min_speed", 50), 50.0)
        wls_max_speed = self._as_float(self.config.get("wls_max_speed", 95), 95.0)
        wls_bins = self._as_int(self.config.get("wls_bins", 10), 10)
        min_sigma = self._as_float(self.config.get("min_sigma", 1e-6), 1e-6)
        min_rows_per_bin = self._as_int(self.config.get("min_rows_per_bin", 30), 30)
        min_train_rows = self._as_int(self.config.get("min_train_rows_per_control_group", 80), 80)
        per_device_cap = self._none_or_int(self.config.get("train_max_rows_per_device_per_control", 2000))
        group_cap = self._none_or_int(self.config.get("train_max_rows_per_control_group", 200000))

        logger.info(
            "开始 WLS 估计: control_pressures=%s, speed=[%s,%s), bins=%s, repeats=%s, main_q=%s",
            control_pressures,
            wls_min_speed,
            wls_max_speed,
            wls_bins,
            n_repeats,
            main_q,
        )

        data, read_logs = self._load_data(
            data_dir=data_dir,
            control_pressures=control_pressures,
            control_round_digits=control_round_digits,
            min_speed=wls_min_speed,
            max_speed=wls_max_speed,
            random_state=random_state,
            logger=logger,
        )
        if data is None or data.empty:
            return {"error": "数据加载失败或无可用记录", "output_dir": output_root}

        data_summary = self._summarize_data(data)
        logger.info("纳入 WLS 分析记录数: %s", f"{len(data):,}")

        repeat_metrics: List[Dict] = []
        fit_rows: List[Dict] = []
        split_summaries: List[Dict] = []
        threshold_rows: List[Dict] = []
        result_rows: List[Dict] = []

        for repeat in range(n_repeats):
            logger.info("WLS repeat %s/%s: 设备级划分", repeat + 1, n_repeats)
            split = self._split_normal_devices(data, repeat, random_state, train_frac)
            split_summaries.extend(self._summarize_split(data, split, repeat))

            for model_id, model_def in self.MODELS.items():
                model_name = model_def.get("name", model_id)
                target_cols = model_def.get("target_cols", [])
                target = self._build_target(data, target_cols)
                target_col = f"target__{model_id}"
                data[target_col] = target

                models_by_cp: Dict[str, Dict] = {}
                for cp in sorted(data["control_group"].dropna().astype(str).unique(), key=self._cp_sort_key):
                    train_mask = (
                        split.eq("normal_train")
                        & data["control_group"].astype(str).eq(str(cp))
                        & (~data["excluded_speed_range_for_wls"].astype(bool))
                        & data["speed_sq"].notna()
                        & data[target_col].notna()
                    )
                    train_df = data.loc[train_mask, ["device_key", "speed_sq", target_col]].copy()
                    train_sampled = self._sample_train_rows(
                        train=train_df,
                        repeat=repeat,
                        model_id=model_id,
                        control_group=str(cp),
                        random_state=random_state,
                        per_device_cap=per_device_cap,
                        group_cap=group_cap,
                    )

                    fit_record = {
                        "repeat": repeat,
                        "model_id": model_id,
                        "model_name": model_name,
                        "control_pressure": self._maybe_float(cp),
                        "fit_status": "ok",
                        "train_rows_before_sample": int(len(train_df)),
                        "train_rows_used": int(len(train_sampled)),
                        "beta_0": np.nan,
                        "beta_1": np.nan,
                    }
                    try:
                        fit_result = fit_wls(
                            train_sampled["speed_sq"].to_numpy(dtype=float),
                            train_sampled[target_col].to_numpy(dtype=float),
                            n_bins=wls_bins,
                            min_sigma=min_sigma,
                            min_rows_per_bin=min_rows_per_bin,
                            min_train_rows=min_train_rows,
                        )
                        fit_result["control_pressure"] = self._maybe_float(cp)
                        fit_result["model_id"] = model_id
                        fit_result["model_name"] = model_name
                        fit_result["target_cols"] = list(target_cols)
                        models_by_cp[str(cp)] = fit_result
                        fit_record.update({
                            "beta_0": float(fit_result["beta0"]),
                            "beta_1": float(fit_result["beta1"]),
                            "n_train": int(fit_result["n_train"]),
                            "global_sigma": float(fit_result["global_sigma"]),
                        })
                    except Exception as exc:
                        fit_record["fit_status"] = f"failed: {exc}"
                        logger.warning("%s control=%s repeat=%s 拟合失败: %s", model_id, cp, repeat, exc)
                    fit_rows.append(fit_record)

                if not models_by_cp:
                    continue

                score_df = self._score_model(data, target_col, models_by_cp, min_sigma)
                thresholds = self._thresholds_by_control(data, split, score_df["score"], threshold_qs)

                for cp, thr_map in thresholds.items():
                    for q, thr in thr_map.items():
                        threshold_rows.append({
                            "repeat": repeat,
                            "model_id": model_id,
                            "model_name": model_name,
                            "control_pressure": self._maybe_float(cp),
                            "threshold_quantile": float(q),
                            "threshold": thr,
                        })

                rows = self._build_heldout_metrics(
                    data=data,
                    split=split,
                    score=score_df["score"],
                    model_id=model_id,
                    model_name=model_name,
                    repeat=repeat,
                    thresholds=thresholds,
                    quantiles=threshold_qs,
                    fit_rows_for_repeat=[r for r in fit_rows if r.get("repeat") == repeat and r.get("model_id") == model_id],
                )
                repeat_metrics.extend(rows)

                # JSON 中保留每次 repeat 的模型和阈值细节；不写额外 CSV，保持项目输出路径不变。
                for cp, m in models_by_cp.items():
                    result_rows.append({
                        "repeat": repeat,
                        "model_id": model_id,
                        "model_name": model_name,
                        "control_pressure": self._maybe_float(cp),
                        "beta": m.get("beta"),
                        "beta_0": m.get("beta0"),
                        "beta_1": m.get("beta1"),
                        "n_train": m.get("n_train"),
                        "bin_edges": m.get("bin_edges"),
                        "sigma_by_bin": m.get("sigma_by_bin"),
                        "global_sigma": m.get("global_sigma"),
                        "thresholds": {f"q{q}": thresholds.get(cp, {}).get(q, np.nan) for q in threshold_qs},
                    })

        # CSV 使用项目现有 metrics 扁平格式，但多次重复取均值，只保留 heldout + 主阈值 + 分控制压差。
        avg_metrics = self._average_metrics_for_csv(repeat_metrics, main_q)
        avg_results = self._average_results(result_rows)

        return {
            "results": avg_results,
            "metrics": avg_metrics,
            "output_dir": output_root,
            "config": {
                "control_pressures": control_pressures,
                "control_round_digits": control_round_digits,
                "wls_min_speed": wls_min_speed,
                "wls_max_speed": wls_max_speed,
                "wls_bins": wls_bins,
                "min_sigma": min_sigma,
                "min_rows_per_bin": min_rows_per_bin,
                "min_train_rows_per_control_group": min_train_rows,
                "threshold_quantiles": threshold_qs,
                "main_threshold_quantile": main_q,
                "n_repeats": n_repeats,
                "train_normal_device_frac": train_frac,
                "random_state": random_state,
                "f01_f02_mode": self.config.get("f01_f02_mode", "exclude"),
                "sampling_note": (
                    "每次重复先按设备划分正常样本；WLS 拟合只使用 normal_train；"
                    "每个训练设备、每个控制压差组最多抽取 train_max_rows_per_device_per_control 条记录；"
                    "阈值来自对应控制压差组 normal_train score 分位数；评价只使用 heldout。"
                ),
            },
            "details": {
                "data_summary": data_summary,
                "read_logs": read_logs,
                "split_summaries": split_summaries,
                "fit_models_by_repeat": fit_rows,
                "thresholds_by_repeat": threshold_rows,
                "heldout_metrics_by_repeat": repeat_metrics,
            },
        }

    # ------------------------------------------------------------------
    # 数据读取与预处理
    # ------------------------------------------------------------------
    def _load_data(
        self,
        data_dir: str,
        control_pressures: Sequence[float],
        control_round_digits: int,
        min_speed: float,
        max_speed: float,
        random_state: int,
        logger,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"数据目录不存在: {root}")

        max_rows = self._none_or_int(
            self.config.get("data_max_rows_per_file", self.config.get("max_rows_per_file", 5000))
        )
        chunksize = self._as_int(self.config.get("csv_chunksize", 200000), 200000)
        recursive = self._as_bool(self.config.get("csv_recursive", True), True)
        f01_f02_mode = str(self.config.get("f01_f02_mode", "exclude")).strip().lower()
        skip_unlabeled = self._as_bool(self.config.get("skip_m_s_u", self.config.get("skip_unlabeled", True)), True)

        required_cols = self._required_columns()
        scale = 10 ** int(control_round_digits)
        allowed_keys = {int(round(float(v) * scale)) for v in control_pressures}

        frames: List[pd.DataFrame] = []
        read_logs: List[Dict] = []

        for code_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            if code_dir.name.strip() in self.IGNORE_DIR_NAMES:
                continue
            code = code_dir.name.strip().upper()
            label = self._label_from_code(code, f01_f02_mode)

            if label is None and code in self.PRIMARY_ANOMALY_CODES and f01_f02_mode == "exclude":
                logger.info("跳过 %s: f01_f02_mode=exclude", code)
                continue
            if code in self.UNLABELED_CODES and skip_unlabeled:
                logger.info("跳过 %s: skip_m_s_u=True", code)
                continue
            if code.startswith("F") and code not in self.SECONDARY_ANOMALY_CODES and code not in self.PRIMARY_ANOMALY_CODES:
                logger.warning("跳过未定义 F 文件夹 %s，请确认是否属于二次侧异常", code)
                continue
            if not (code == "N" or code in self.SECONDARY_ANOMALY_CODES or code in self.PRIMARY_ANOMALY_CODES or code in self.UNLABELED_CODES):
                continue

            csv_files = sorted(code_dir.rglob("*.csv") if recursive else code_dir.glob("*.csv"))
            logger.info("读取 %s: %s 个 CSV", code, len(csv_files))

            for fp in csv_files:
                try:
                    seed = self._stable_seed(random_state, str(fp))
                    raw = self._read_csv_selected(fp, required_cols, max_rows, chunksize, seed)
                    if raw.empty:
                        continue

                    df = pd.DataFrame(index=raw.index)
                    for col in required_cols:
                        df[col] = self._to_numeric(raw[col])
                    df["row_id"] = raw["row_id"].astype(np.int64)

                    cp_key = self._control_key_from_values(df["控制压差目标值"], control_round_digits)
                    before = len(df)
                    keep = cp_key.isin(allowed_keys)
                    df = df.loc[keep].copy()
                    cp_key = cp_key.loc[keep]
                    if df.empty:
                        read_logs.append({
                            "source_code": code,
                            "device_name": fp.stem,
                            "file_path": str(fp),
                            "read_rows": int(before),
                            "kept_rows": 0,
                            "status": "no_valid_control_pressure",
                        })
                        continue

                    df["control_pressure_key"] = cp_key.astype(np.int64)
                    df["control_group"] = (df["control_pressure_key"] / scale).map(lambda x: f"{float(x):g}")
                    df["control_pressure"] = df["control_pressure_key"] / scale
                    df["speed_raw"] = df["二次侧泵转速"]
                    df["speed_norm"] = self._normalize_speed(df["speed_raw"])
                    df["speed_sq"] = df["speed_norm"] ** 2
                    df["excluded_speed_range_for_wls"] = self._speed_range_exclusion_mask(
                        df["speed_raw"], min_speed, max_speed
                    )
                    df["source_code"] = code
                    df["device_name"] = fp.stem
                    df["file_path"] = str(fp)
                    df["device_key"] = code + "/" + fp.stem
                    df["binary_label"] = np.nan if label is None else int(label)
                    df["label_name"] = df["binary_label"].map({0: "normal", 1: "secondary_anomaly"}).fillna("unlabeled")

                    keep_cols = [
                        "source_code",
                        "device_name",
                        "file_path",
                        "device_key",
                        "row_id",
                        "binary_label",
                        "label_name",
                        "control_group",
                        "control_pressure",
                        "control_pressure_key",
                        "控制压差目标值",
                        "speed_raw",
                        "speed_norm",
                        "speed_sq",
                        "excluded_speed_range_for_wls",
                    ] + required_cols
                    # 去重，避免 required_cols 中已有控制压差/转速列导致重复。
                    keep_cols = list(dict.fromkeys(keep_cols))
                    frames.append(df[keep_cols].reset_index(drop=True))
                    read_logs.append({
                        "source_code": code,
                        "device_name": fp.stem,
                        "file_path": str(fp),
                        "read_rows": int(before),
                        "kept_rows": int(len(df)),
                        "status": "ok",
                    })
                except Exception as exc:
                    logger.error("读取失败: %s; 错误: %s", fp, exc)
                    read_logs.append({
                        "source_code": code,
                        "device_name": fp.stem,
                        "file_path": str(fp),
                        "read_rows": 0,
                        "kept_rows": 0,
                        "status": f"error: {exc}",
                    })

        if not frames:
            return pd.DataFrame(), read_logs

        data = pd.concat(frames, ignore_index=True)
        for c in ["source_code", "device_name", "file_path", "device_key", "label_name", "control_group"]:
            data[c] = data[c].astype("category")
        return data, read_logs

    def _read_csv_selected(
        self,
        path: Path,
        usecols: List[str],
        max_rows: Optional[int],
        chunksize: int,
        random_state: int,
    ) -> pd.DataFrame:
        encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
        last_err: Optional[Exception] = None
        for enc in encodings:
            try:
                header = list(pd.read_csv(path, encoding=enc, nrows=0).columns)
                missing = [c for c in usecols if c not in header]
                if missing:
                    raise RuntimeError(f"缺少必需列 {missing}")

                if max_rows is None:
                    df = pd.read_csv(path, encoding=enc, usecols=usecols)
                    df["row_id"] = np.arange(len(df), dtype=np.int64)
                    return df

                rng = np.random.default_rng(random_state)
                kept: Optional[pd.DataFrame] = None
                offset = 0
                for chunk in pd.read_csv(path, encoding=enc, usecols=usecols, chunksize=chunksize):
                    n = len(chunk)
                    chunk = chunk.copy()
                    chunk["row_id"] = np.arange(offset, offset + n, dtype=np.int64)
                    chunk["__rand_key__"] = rng.random(n)
                    offset += n
                    kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
                    if len(kept) > max_rows:
                        kept = kept.nsmallest(max_rows, "__rand_key__")
                if kept is None:
                    return pd.DataFrame(columns=usecols + ["row_id"])
                kept = kept.sort_values("row_id").drop(columns=["__rand_key__"])
                return kept.reset_index(drop=True)
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(f"无法读取 CSV：{path}；最后错误：{last_err}")

    def _required_columns(self) -> List[str]:
        cols = {"二次侧泵转速", "控制压差目标值"}
        for cfg in self.MODELS.values():
            cols.update(cfg.get("target_cols", []))
        return list(cols)

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        s = series.astype(str).str.strip()
        s = (
            s.str.replace("−", "-", regex=False)
            .str.replace("－", "-", regex=False)
            .str.replace("，", ".", regex=False)
        )
        direct = pd.to_numeric(s, errors="coerce")
        need = direct.isna() & s.notna() & ~s.str.lower().isin(["", "nan", "none", "null"])
        if need.any():
            extracted = s[need].str.extract(r"([-+]?\d+(?:[\.,]\d+)?)", expand=False)
            extracted = extracted.str.replace(",", ".", regex=False)
            direct.loc[need] = pd.to_numeric(extracted, errors="coerce")
        return direct

    @staticmethod
    def _control_key_from_values(values: pd.Series, digits: int) -> pd.Series:
        scale = 10 ** int(digits)
        x = pd.to_numeric(values, errors="coerce").astype("float64")
        out = pd.Series(np.nan, index=values.index, dtype="float64")
        ok = x.notna()
        out.loc[ok] = np.rint(x.loc[ok] * scale)
        return out

    @staticmethod
    def _normalize_speed(speed_raw: pd.Series) -> pd.Series:
        speed = pd.to_numeric(speed_raw, errors="coerce")
        q95 = speed.dropna().quantile(0.95) if speed.notna().any() else np.nan
        if pd.notna(q95) and q95 > 2:
            return speed / 100.0
        return speed

    @staticmethod
    def _speed_range_exclusion_mask(speed_raw: pd.Series, min_speed: float, max_speed: float) -> pd.Series:
        x = pd.to_numeric(speed_raw, errors="coerce")
        q95 = x.dropna().quantile(0.95) if x.notna().any() else np.nan
        if pd.notna(q95) and q95 > 2:
            low, high = float(min_speed), float(max_speed)
        else:
            low, high = float(min_speed) / 100.0, float(max_speed) / 100.0
        return x.isna() | (x < low) | (x >= high)

    def _label_from_code(self, code: str, f01_f02_mode: str) -> Optional[int]:
        code = code.upper()
        if code == "N":
            return 0
        if code in self.SECONDARY_ANOMALY_CODES:
            return 1
        if code in self.PRIMARY_ANOMALY_CODES:
            return 0 if f01_f02_mode == "as_normal" else None
        return None

    # ------------------------------------------------------------------
    # 训练、打分、阈值、指标
    # ------------------------------------------------------------------
    def _split_normal_devices(self, data: pd.DataFrame, repeat: int, random_state: int, train_frac: float) -> pd.Series:
        rng = np.random.default_rng(int(random_state) + int(repeat) * 1009)
        normal_keys = np.array(sorted(data.loc[data["binary_label"].eq(0), "device_key"].dropna().astype(str).unique()))
        if len(normal_keys) < 2:
            raise RuntimeError("正常设备数量少于 2，无法做设备级训练/留出划分。")
        rng.shuffle(normal_keys)
        n_train = max(1, int(round(len(normal_keys) * float(train_frac))))
        n_train = min(n_train, len(normal_keys) - 1)
        train_keys = set(normal_keys[:n_train])

        split = pd.Series("unlabeled", index=data.index, dtype="object")
        device_str = data["device_key"].astype(str)
        split.loc[data["binary_label"].eq(1)] = "anomaly"
        split.loc[data["binary_label"].eq(0) & device_str.isin(train_keys)] = "normal_train"
        split.loc[data["binary_label"].eq(0) & ~device_str.isin(train_keys)] = "normal_heldout"
        return split

    def _sample_train_rows(
        self,
        train: pd.DataFrame,
        repeat: int,
        model_id: str,
        control_group: str,
        random_state: int,
        per_device_cap: Optional[int],
        group_cap: Optional[int],
    ) -> pd.DataFrame:
        if train.empty:
            return train
        parts = []
        for dev, sub in train.groupby("device_key", observed=True):
            if per_device_cap is not None and len(sub) > per_device_cap:
                seed = self._stable_seed(random_state + repeat * 1009, f"{model_id}|{control_group}|{dev}")
                parts.append(sub.sample(n=per_device_cap, random_state=seed))
            else:
                parts.append(sub)
        out = pd.concat(parts, axis=0) if parts else train.iloc[0:0]
        if group_cap is not None and len(out) > group_cap:
            seed = self._stable_seed(random_state + repeat * 1013, f"{model_id}|{control_group}|group_cap")
            out = out.sample(n=group_cap, random_state=seed)
        return out

    def _build_target(self, df: pd.DataFrame, target_cols: Sequence[str]) -> pd.Series:
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"缺少模型因变量列: {missing}")
        values = [pd.to_numeric(df[c], errors="coerce") for c in target_cols]
        if len(values) == 1:
            return values[0]
        return pd.concat(values, axis=1).sum(axis=1, min_count=len(values))

    def _score_model(self, data: pd.DataFrame, target_col: str, models_by_cp: Dict[str, Dict], min_sigma: float) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["score"] = np.nan
        out["residual"] = np.nan
        out["yhat"] = np.nan
        out["sigma"] = np.nan
        out["model_available"] = False

        valid_common = (
            (~data["excluded_speed_range_for_wls"].astype(bool))
            & data["speed_sq"].notna()
            & data[target_col].notna()
        )
        for cp, model in models_by_cp.items():
            idx = data.index[valid_common & data["control_group"].astype(str).eq(str(cp))]
            if len(idx) == 0:
                continue
            x = data.loc[idx, "speed_sq"].to_numpy(dtype=float)
            y = data.loc[idx, target_col].to_numpy(dtype=float)
            yhat, sigma = predict_wls(x, model)
            resid = y - yhat
            score = np.abs(resid) / np.maximum(sigma, float(min_sigma))
            out.loc[idx, "score"] = score
            out.loc[idx, "residual"] = resid
            out.loc[idx, "yhat"] = yhat
            out.loc[idx, "sigma"] = sigma
            out.loc[idx, "model_available"] = True
        return out

    def _thresholds_by_control(
        self,
        data: pd.DataFrame,
        split: pd.Series,
        score: pd.Series,
        quantiles: Sequence[float],
    ) -> Dict[str, Dict[float, float]]:
        thresholds: Dict[str, Dict[float, float]] = {}
        for cp in sorted(data["control_group"].dropna().astype(str).unique(), key=self._cp_sort_key):
            s = score.loc[split.eq("normal_train") & data["control_group"].astype(str).eq(str(cp))].dropna()
            if len(s) == 0:
                thresholds[str(cp)] = {float(q): np.nan for q in quantiles}
            else:
                thresholds[str(cp)] = {float(q): float(np.nanquantile(s.to_numpy(dtype=float), q)) for q in quantiles}
        return thresholds

    def _build_heldout_metrics(
        self,
        data: pd.DataFrame,
        split: pd.Series,
        score: pd.Series,
        model_id: str,
        model_name: str,
        repeat: int,
        thresholds: Dict[str, Dict[float, float]],
        quantiles: Sequence[float],
        fit_rows_for_repeat: List[Dict],
    ) -> List[Dict]:
        rows: List[Dict] = []
        y_true_all = data["binary_label"].to_numpy(dtype=float)
        score_all = score.to_numpy(dtype=float)
        fit_by_cp = {str(r.get("control_pressure")): r for r in fit_rows_for_repeat}
        # 同时兼容 1.4 / "1.4" 这种键。
        fit_by_cp.update({f"{float(r.get('control_pressure')):g}": r for r in fit_rows_for_repeat if pd.notna(r.get("control_pressure"))})

        heldout_mask = split.isin(["normal_heldout", "anomaly"]) & data["binary_label"].notna()

        for q in quantiles:
            row_threshold = np.full(len(data), np.nan, dtype=float)
            for cp, thr_map in thresholds.items():
                thr = thr_map.get(float(q), np.nan)
                idx = data.index[data["control_group"].astype(str).eq(str(cp))]
                row_threshold[idx] = thr
            y_pred_all = (score_all >= row_threshold).astype(float)
            y_pred_all[~np.isfinite(score_all) | ~np.isfinite(row_threshold)] = np.nan

            # JSON 里保留三组合并的 heldout/all；CSV 汇总时会过滤掉 all，保持当前 CSV 结构简洁。
            m_all = self._metric_from_arrays(
                y_true_all[heldout_mask.values], score_all[heldout_mask.values], y_pred_all[heldout_mask.values]
            )
            rows.append({
                "repeat": repeat,
                "model_id": model_id,
                "model_name": model_name,
                "eval_scope": "heldout",
                "control_pressure": "all",
                "threshold_quantile": float(q),
                "threshold": np.nan,
                **m_all,
            })

            for cp in sorted(thresholds, key=self._cp_sort_key):
                mask = heldout_mask & data["control_group"].astype(str).eq(str(cp))
                m = self._metric_from_arrays(y_true_all[mask.values], score_all[mask.values], y_pred_all[mask.values])
                fit_info = fit_by_cp.get(str(cp), {})
                rows.append({
                    "repeat": repeat,
                    "model_id": model_id,
                    "model_name": model_name,
                    "eval_scope": "heldout",
                    "control_pressure": self._maybe_float(cp),
                    "threshold_quantile": float(q),
                    "threshold": thresholds[cp].get(float(q), np.nan),
                    "beta_0": fit_info.get("beta_0", np.nan),
                    "beta_1": fit_info.get("beta_1", np.nan),
                    **m,
                })
        return rows

    @staticmethod
    def _metric_from_arrays(y_true: np.ndarray, score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mask = np.isfinite(y_true) & np.isfinite(score) & np.isfinite(y_pred)
        y_true = y_true[mask].astype(int)
        y_pred = y_pred[mask].astype(int)
        if len(y_true) == 0:
            return {
                "n": 0,
                "normal_count": 0,
                "anomaly_count": 0,
                "TP": 0,
                "FP": 0,
                "TN": 0,
                "FN": 0,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "fpr": np.nan,
                "fnr": np.nan,
                "accuracy": np.nan,
                "mcc": np.nan,
                "balanced_accuracy": np.nan,
            }

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        normal_count = tn + fp
        anomaly_count = tp + fn
        pred_pos = tp + fp
        n = len(y_true)

        precision = tp / pred_pos if pred_pos else np.nan
        recall = tp / anomaly_count if anomaly_count else np.nan
        fpr = fp / normal_count if normal_count else np.nan
        fnr = fn / anomaly_count if anomaly_count else np.nan
        accuracy = (tp + tn) / n if n else np.nan
        f1 = 2 * precision * recall / (precision + recall) if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0 else np.nan
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else np.nan
        tnr = tn / normal_count if normal_count else np.nan
        balanced_accuracy = np.nanmean([recall, tnr]) if (pd.notna(recall) or pd.notna(tnr)) else np.nan

        return {
            "n": int(n),
            "normal_count": int(normal_count),
            "anomaly_count": int(anomaly_count),
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": float(precision) if pd.notna(precision) else np.nan,
            "recall": float(recall) if pd.notna(recall) else np.nan,
            "f1": float(f1) if pd.notna(f1) else np.nan,
            "fpr": float(fpr) if pd.notna(fpr) else np.nan,
            "fnr": float(fnr) if pd.notna(fnr) else np.nan,
            "accuracy": float(accuracy) if pd.notna(accuracy) else np.nan,
            "mcc": float(mcc) if pd.notna(mcc) else np.nan,
            "balanced_accuracy": float(balanced_accuracy) if pd.notna(balanced_accuracy) else np.nan,
        }

    def _average_metrics_for_csv(self, repeat_metrics: List[Dict], main_q: float) -> Dict[str, List[Dict]]:
        if not repeat_metrics:
            return {mid: [] for mid in self.MODELS}
        df = pd.DataFrame(repeat_metrics)
        df = df[
            (df["eval_scope"] == "heldout")
            & (df["control_pressure"].astype(str) != "all")
            & np.isclose(pd.to_numeric(df["threshold_quantile"], errors="coerce"), float(main_q))
        ].copy()
        if df.empty:
            return {mid: [] for mid in self.MODELS}

        current_metric_cols = [
            "beta_0",
            "beta_1",
            "threshold",
            "TP",
            "FP",
            "TN",
            "FN",
            "precision",
            "recall",
            "f1",
            "fpr",
            "fnr",
            "accuracy",
        ]
        rows_by_model: Dict[str, List[Dict]] = {mid: [] for mid in self.MODELS}
        group_cols = ["model_id", "model_name", "control_pressure", "threshold_quantile"]
        for key, sub in df.groupby(group_cols, dropna=False):
            model_id, model_name, cp, q = key
            row = {
                "control_pressure": self._maybe_float(cp),
                "threshold_quantile": float(q),
                "repeat_count": int(sub["repeat"].nunique()),
            }
            for c in current_metric_cols:
                if c in sub.columns:
                    row[c] = self._nanmean_or_nan(sub[c])
            rows_by_model.setdefault(str(model_id), []).append(row)
        for model_id in rows_by_model:
            rows_by_model[model_id] = sorted(rows_by_model[model_id], key=lambda r: self._cp_sort_key(r.get("control_pressure")))
        return rows_by_model

    def _average_results(self, result_rows: List[Dict]) -> Dict[str, List[Dict]]:
        if not result_rows:
            return {mid: [] for mid in self.MODELS}
        df = pd.DataFrame(result_rows)
        out: Dict[str, List[Dict]] = {mid: [] for mid in self.MODELS}
        for (model_id, model_name, cp), sub in df.groupby(["model_id", "model_name", "control_pressure"], dropna=False):
            thresholds = {}
            # thresholds 是 dict 列，需要手工聚合。
            all_qs = sorted({q for d in sub["thresholds"].dropna() for q in d.keys()})
            for q in all_qs:
                vals = [d.get(q, np.nan) for d in sub["thresholds"] if isinstance(d, dict)]
                thresholds[q] = self._nanmean_or_nan(vals)
            out.setdefault(str(model_id), []).append({
                "control_pressure": self._maybe_float(cp),
                "beta": [self._nanmean_or_nan(sub["beta_0"]), self._nanmean_or_nan(sub["beta_1"])],
                "beta_0": self._nanmean_or_nan(sub["beta_0"]),
                "beta_1": self._nanmean_or_nan(sub["beta_1"]),
                "n_train_mean": self._nanmean_or_nan(sub["n_train"]),
                "thresholds": thresholds,
                "repeat_count": int(sub["repeat"].nunique()),
            })
        for model_id in out:
            out[model_id] = sorted(out[model_id], key=lambda r: self._cp_sort_key(r.get("control_pressure")))
        return out

    # ------------------------------------------------------------------
    # 汇总与小工具
    # ------------------------------------------------------------------
    def _summarize_data(self, data: pd.DataFrame) -> List[Dict]:
        if data.empty:
            return []
        summary = (
            data.groupby(["control_group", "source_code", "label_name"], observed=True, dropna=False)
            .size()
            .reset_index(name="records")
        )
        return summary.to_dict("records")

    def _summarize_split(self, data: pd.DataFrame, split: pd.Series, repeat: int) -> List[Dict]:
        tmp = data[["source_code", "label_name"]].copy()
        tmp["split"] = split.values
        summary = tmp.groupby(["source_code", "label_name", "split"], observed=True, dropna=False).size().reset_index(name="records")
        summary["repeat"] = repeat
        return summary.to_dict("records")

    @staticmethod
    def _stable_seed(base: int, text: str) -> int:
        h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        return (int(h[:8], 16) + int(base)) % (2**32 - 1)

    @staticmethod
    def _float_list(value) -> List[float]:
        if isinstance(value, str):
            return sorted({float(x.strip()) for x in value.split(",") if x.strip()})
        if isinstance(value, Iterable):
            return sorted({float(x) for x in value})
        return [float(value)]

    @staticmethod
    def _as_float(value, default: float) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_int(value, default: int) -> int:
        try:
            if value is None:
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _as_bool(value, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _none_or_int(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"none", "null", "", "nan"}:
            return None
        out = int(value)
        return None if out <= 0 else out

    @staticmethod
    def _maybe_float(value):
        try:
            return float(value)
        except Exception:
            return value

    @staticmethod
    def _cp_sort_key(value):
        try:
            return float(value)
        except Exception:
            return str(value)

    @staticmethod
    def _nanmean_or_nan(values) -> float:
        arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) else np.nan
