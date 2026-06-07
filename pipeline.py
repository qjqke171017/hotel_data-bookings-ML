"""路线三：WLS 机理估计全流程编排。

以独立 WLS 脚本的实验口径为标准，本文件修复了以下关键问题：
1. 按正常设备划分 normal_train / normal_heldout，评价使用 heldout 正常样本 + 异常样本；
2. WLS 拟合和阈值计算只使用 normal_train，不混入异常样本；
3. 阈值和测试均使用 score = |残差| / 局部 sigma；
4. 读取阶段只保留 WLS 必需列，并支持每个 CSV 过滤后抽样，降低内存占用。
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
        self.config = config or {}
        # 允许 wls_config.yaml 覆盖模型列定义，但保留 M1/M2/M3 变量名。
        cfg_models = self.config.get("models", {}) if isinstance(self.config.get("models", {}), dict) else {}
        if cfg_models:
            for model_id, model_def in cfg_models.items():
                if model_id in self.MODELS:
                    self.MODELS[model_id]["name"] = model_def.get("name", self.MODELS[model_id]["name"])
                    self.MODELS[model_id]["target_cols"] = model_def.get(
                        "target_columns",
                        model_def.get("target_cols", self.MODELS[model_id]["target_cols"]),
                    )

    @staticmethod
    def _resolve_column(columns: Sequence[str], candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        return None

    @staticmethod
    def _stable_seed(base: int, text: str) -> int:
        h = hashlib.md5(str(text).encode("utf-8", errors="ignore")).hexdigest()
        return (int(h[:8], 16) + int(base)) % (2**32 - 1)

    def _prepare_csv_reader(self, fp: Path):
        """确定编码、usecols 与逻辑列映射。"""
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
                return enc, logical_to_actual, usecols
            except Exception as e:
                last_err = e
                continue
        print(f"[WARN] CSV 表头读取失败：{fp}；最后错误：{last_err}")
        return None

    def _iter_required_csv_chunks(self, fp: Path):
        """分块读取单个 CSV 的 WLS 必需列，并统一为逻辑列名。"""
        prepared = self._prepare_csv_reader(fp)
        if prepared is None:
            return
        enc, logical_to_actual, usecols = prepared
        chunksize = self.config.get("csv_chunksize", 200000)
        try:
            chunks = pd.read_csv(fp, encoding=enc, usecols=usecols, chunksize=chunksize)
        except TypeError:
            chunks = [pd.read_csv(fp, encoding=enc, usecols=usecols)]
        except Exception as e:
            print(f"[WARN] CSV 读取失败：{fp}；错误：{e}")
            return

        try:
            for raw in chunks:
                out = pd.DataFrame(index=raw.index)
                out["__speed_raw__"] = pd.to_numeric(raw[logical_to_actual["__speed_raw__"]], errors="coerce")
                out["__control_raw__"] = pd.to_numeric(raw[logical_to_actual["__control_raw__"]], errors="coerce")
                for logical in self._TARGET_CANDIDATES:
                    actual = logical_to_actual.get(logical)
                    if actual:
                        out[logical] = pd.to_numeric(raw[actual], errors="coerce")
                yield out
        except Exception as e:
            print(f"[WARN] CSV 分块处理失败：{fp}；错误：{e}")
            return

    def run(self, data_dir: str, output_root: str = None) -> Dict:
        """执行 WLS 机理异常检测。"""
        if output_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root = os.path.join("./output/wls", f"report_{timestamp}")
        output_root = ensure_dir(output_root)
        logger = setup_logging(output_root)

        control_pressures = [float(x) for x in self.config.get("control_pressures", [0.8, 1.4, 1.65])]
        wls_min_speed = float(self.config.get("wls_min_speed", 50))
        wls_max_speed = float(self.config.get("wls_max_speed", 95))
        wls_bins = int(self.config.get("wls_bins", 10))
        min_sigma = float(self.config.get("min_sigma", 1e-6))
        min_rows_per_bin = int(self.config.get("min_rows_per_bin", 30))
        threshold_qs = [float(q) for q in self.config.get("threshold_quantiles", [0.95, 0.975, 0.99, 0.9975, 0.9999])]
        main_q = float(self.config.get("main_threshold_quantile", 0.9975))
        if main_q not in threshold_qs:
            threshold_qs = sorted(set(threshold_qs + [main_q]))
        n_repeats = int(self.config.get("n_repeats", 20))
        random_state = int(self.config.get("random_state", 42))
        train_frac = float(self.config.get("train_normal_device_frac", 0.70))

        logger.info(
            f"开始 WLS 估计: control_pressures={control_pressures}, "
            f"speed=[{wls_min_speed},{wls_max_speed}), bins={wls_bins}, repeats={n_repeats}"
        )

        data = self._load_data(data_dir, wls_min_speed, wls_max_speed)
        if data is None or data.empty:
            return {"error": "数据加载失败"}
        logger.info(f"WLS 纳入记录数: {len(data):,}, 列数: {len(data.columns)}")

        repeat_metrics: List[Dict] = []
        repeat_models: List[Dict] = []

        for repeat in range(n_repeats):
            logger.info(f"WLS repeat {repeat + 1}/{n_repeats}: 设备级划分")
            split = self._split_normal_devices(data, repeat, train_frac, random_state)
            repeat_model_summary = {"repeat": repeat, "models": {}}

            for model_id, model_def in self.MODELS.items():
                repeat_model_summary["models"].setdefault(model_id, [])
                for cp in control_pressures:
                    cp_mask = np.isclose(data["control_pressure"].to_numpy(dtype=float), float(cp), atol=1e-9)
                    if not cp_mask.any():
                        continue

                    target = self._build_target(data.loc[cp_mask], model_def["target_cols"])
                    if target is None:
                        logger.warning(f"{model_id}, control_pressure={cp}: 缺少因变量列，跳过")
                        continue

                    # 只用 normal_train 拟合 WLS；异常样本绝不参与拟合。
                    sub_index = data.index[cp_mask]
                    x_all = data.loc[sub_index, "speed_sq"].to_numpy(dtype=float)
                    y_all = target
                    label_all = data.loc[sub_index, "_label"].to_numpy(dtype=float)
                    split_all = split.loc[sub_index].to_numpy(dtype=object)
                    device_all = data.loc[sub_index, "device_key"].astype(str).to_numpy()

                    valid_xy = np.isfinite(x_all) & np.isfinite(y_all)
                    train_mask = valid_xy & (split_all == "normal_train") & (label_all == 0)
                    if int(train_mask.sum()) < int(self.config.get("min_train_rows_per_control_group", 80)):
                        logger.warning(
                            f"repeat={repeat}, {model_id}, cp={cp}: normal_train 有效样本不足，跳过"
                        )
                        continue

                    train_df = pd.DataFrame({
                        "device_key": device_all[train_mask],
                        "speed_sq": x_all[train_mask],
                        "target": y_all[train_mask],
                    })
                    train_sample = self._sample_train_rows(train_df, repeat, model_id, cp, random_state)
                    if len(train_sample) < int(self.config.get("min_train_rows_per_control_group", 80)):
                        logger.warning(
                            f"repeat={repeat}, {model_id}, cp={cp}: 抽样后训练样本不足，跳过"
                        )
                        continue

                    try:
                        model = fit_wls(
                            train_sample["speed_sq"].to_numpy(dtype=float),
                            train_sample["target"].to_numpy(dtype=float),
                            n_bins=wls_bins,
                            min_sigma=min_sigma,
                            min_rows_per_bin=min_rows_per_bin,
                            return_full=False,
                        )
                    except Exception as e:
                        logger.warning(f"repeat={repeat}, {model_id}, cp={cp}: WLS 拟合失败：{e}")
                        continue

                    # 阈值只来自 normal_train 的标准化残差分数。
                    train_scores = score_wls(
                        x_all[train_mask],
                        y_all[train_mask],
                        np.asarray(model["beta"], dtype=float),
                        np.asarray(model["sigma_per_bin"], dtype=float),
                        np.asarray(model["bin_edges"], dtype=float),
                        min_sigma=min_sigma,
                    )
                    train_scores = train_scores[np.isfinite(train_scores)]
                    if len(train_scores) == 0:
                        logger.warning(f"repeat={repeat}, {model_id}, cp={cp}: normal_train score 为空，跳过")
                        continue
                    thresholds = {f"q{q}": float(np.nanquantile(train_scores, q)) for q in threshold_qs}
                    thr = thresholds.get(f"q{main_q}", float(np.nanquantile(train_scores, main_q)))

                    # 主评价：normal_heldout + anomaly。
                    eval_mask = valid_xy & ((split_all == "normal_heldout") | (split_all == "anomaly"))
                    if int(eval_mask.sum()) == 0:
                        logger.warning(f"repeat={repeat}, {model_id}, cp={cp}: 评价样本为空，跳过")
                        continue

                    eval_scores = score_wls(
                        x_all[eval_mask],
                        y_all[eval_mask],
                        np.asarray(model["beta"], dtype=float),
                        np.asarray(model["sigma_per_bin"], dtype=float),
                        np.asarray(model["bin_edges"], dtype=float),
                        min_sigma=min_sigma,
                    )
                    valid_eval_score = np.isfinite(eval_scores)
                    y_true = label_all[eval_mask][valid_eval_score].astype(int)
                    y_pred = (eval_scores[valid_eval_score] >= thr).astype(int)
                    metrics = compute_sample_metrics(y_true, y_pred)

                    beta = model["beta"]
                    row = {
                        "repeat": repeat,
                        "model": model_id,
                        "control_pressure": float(cp),
                        "beta_0": float(beta[0]),
                        "beta_1": float(beta[1]),
                        "threshold": float(thr),
                        **metrics,
                    }
                    repeat_metrics.append(row)

                    repeat_model_summary["models"][model_id].append({
                        "control_pressure": float(cp),
                        "beta": model["beta"],
                        "threshold": float(thr),
                        "thresholds": thresholds,
                        "train_rows_before_sample": int(train_mask.sum()),
                        "train_rows_used": int(len(train_sample)),
                        "eval_rows": int(len(y_true)),
                    })

                    del train_df, train_sample, model, train_scores, eval_scores

            repeat_models.append(repeat_model_summary)

        if not repeat_metrics:
            return {"error": "WLS 没有产生有效指标，请检查训练样本、列名和控制压差配置。"}

        metrics = self._average_metrics_for_report(repeat_metrics, control_pressures)

        # 每次实验明细单独保存 JSON；evaluation_report.json 也会包含同样信息。
        repeat_json_path = os.path.join(output_root, "wls_repeat_metrics.json")
        try:
            with open(repeat_json_path, "w", encoding="utf-8") as f:
                json.dump({"repeat_metrics": repeat_metrics, "repeat_models": repeat_models}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存 WLS repeat JSON 失败：{e}")

        return {
            "results": {"repeat_models": repeat_models},
            "repeat_metrics": repeat_metrics,
            "metrics": metrics,
            "output_dir": output_root,
            "config": {
                "control_pressures": control_pressures,
                "wls_bins": wls_bins,
                "threshold_quantiles": threshold_qs,
                "main_threshold_quantile": main_q,
                "n_repeats": n_repeats,
                "train_normal_device_frac": train_frac,
                "eval_scope": "normal_heldout + anomaly",
            },
        }

    def _load_data(self, data_dir: str, min_speed: float, max_speed: float) -> pd.DataFrame:
        """加载并预处理二次侧数据，只保留 WLS 必需列。"""
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"数据目录不存在: {root}")

        f01_f02_mode = str(self.config.get("f01_f02_mode", "exclude")).lower()
        skip_m_s_u = bool(self.config.get("skip_m_s_u", True))

        folders: Dict[str, Optional[int]] = {"N": 0, "F03": 1, "F04": 1, "F06": 1}
        if f01_f02_mode == "as_normal":
            folders.update({"F01": 0, "F02": 0})
        elif f01_f02_mode != "exclude":
            print(f"[WARN] 未知 f01_f02_mode={f01_f02_mode}，按 exclude 处理")
        if not skip_m_s_u:
            # 当前主指标不使用未标注样本。为避免污染训练/评价，这里仍不纳入。
            print("[WARN] WLS 主指标不使用 M/S/U；skip_m_s_u=False 时仍不会纳入训练和评价。")

        valid_cps = np.asarray(self.config.get("control_pressures", [0.8, 1.4, 1.65]), dtype=np.float64)
        cp_tolerance = float(self.config.get("control_pressure_tolerance", 0.2))
        max_rows_per_file = self.config.get("max_rows_per_file", None)
        cap = None if max_rows_per_file in (None, "", "none", "None") else int(max_rows_per_file)
        csv_recursive = bool(self.config.get("csv_recursive", True))
        random_state = int(self.config.get("random_state", 42))

        frames = []
        file_count = 0
        kept_count = 0

        for folder_name, label in folders.items():
            folder_path = root / folder_name
            if not folder_path.exists():
                continue
            csv_files = sorted(folder_path.rglob("*.csv") if csv_recursive else folder_path.glob("*.csv"))
            for fp in csv_files:
                file_count += 1
                file_kept = None
                rng = np.random.default_rng(self._stable_seed(random_state, str(fp)))

                for chunk in self._iter_required_csv_chunks(fp):
                    if chunk is None or chunk.empty:
                        continue

                    speed_raw = chunk["__speed_raw__"]
                    control = chunk["__control_raw__"]
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
                    cp_mask = np.isfinite(cp_diff) & (cp_diff < cp_tolerance)

                    keep = speed_mask.to_numpy(dtype=bool) & cp_mask
                    keep_idx = np.flatnonzero(keep)
                    if len(keep_idx) == 0:
                        continue

                    out = pd.DataFrame(index=np.arange(len(keep_idx)))
                    out["speed_norm"] = speed_norm.to_numpy(dtype=np.float32)[keep_idx]
                    out["speed_sq"] = (out["speed_norm"].to_numpy(dtype=np.float32) ** 2).astype(np.float32)
                    out["control_pressure"] = rounded[keep_idx]
                    out["_label"] = np.int8(label)
                    out["_folder"] = folder_name
                    out["device_name"] = fp.stem
                    out["device_key"] = f"{folder_name}/{fp.stem}"

                    for logical in self._TARGET_CANDIDATES:
                        if logical in chunk.columns:
                            out[logical] = chunk[logical].to_numpy(dtype=np.float32)[keep_idx]
                        else:
                            out[logical] = np.nan

                    if cap is not None and cap > 0:
                        out["__rand_key__"] = rng.random(len(out))
                        if file_kept is None:
                            file_kept = out
                        else:
                            file_kept = pd.concat([file_kept, out], ignore_index=True, copy=False)
                        if len(file_kept) > cap:
                            file_kept = file_kept.nsmallest(cap, "__rand_key__").reset_index(drop=True)
                    else:
                        frames.append(out)
                        kept_count += len(out)

                if cap is not None and cap > 0 and file_kept is not None and not file_kept.empty:
                    file_kept = file_kept.sort_values("__rand_key__").drop(columns=["__rand_key__"]).reset_index(drop=True)
                    frames.append(file_kept)
                    kept_count += len(file_kept)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, copy=False)
        combined["_folder"] = combined["_folder"].astype("category")
        combined["device_name"] = combined["device_name"].astype("category")
        combined["device_key"] = combined["device_key"].astype("category")
        combined["_label"] = combined["_label"].astype(np.int8)

        print(
            f"[INFO] WLS读取完成：CSV文件 {file_count} 个，过滤后记录 {kept_count:,} 条，"
            f"保留列 {len(combined.columns)} 列"
        )
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

    def _split_normal_devices(
        self,
        data: pd.DataFrame,
        repeat: int,
        train_frac: float,
        random_state: int,
    ) -> pd.Series:
        """按正常设备划分 normal_train / normal_heldout；异常样本只进入评价。"""
        normal_keys = np.array(sorted(data.loc[data["_label"].eq(0), "device_key"].astype(str).unique()))
        if len(normal_keys) < 2:
            raise RuntimeError("正常设备数量少于 2，无法做设备级训练/测试划分。")

        rng = np.random.default_rng(random_state + repeat * 1009)
        rng.shuffle(normal_keys)
        n_train = max(1, int(round(len(normal_keys) * train_frac)))
        n_train = min(n_train, len(normal_keys) - 1)
        train_keys = set(normal_keys[:n_train])

        split = pd.Series("ignore", index=data.index, dtype="object")
        device_str = data["device_key"].astype(str)
        split.loc[data["_label"].eq(1)] = "anomaly"
        split.loc[data["_label"].eq(0) & device_str.isin(train_keys)] = "normal_train"
        split.loc[data["_label"].eq(0) & ~device_str.isin(train_keys)] = "normal_heldout"
        return split

    def _sample_train_rows(
        self,
        train: pd.DataFrame,
        repeat: int,
        model_id: str,
        control_pressure: float,
        random_state: int,
    ) -> pd.DataFrame:
        """每个训练设备、每个控制压差组内限制训练记录数，避免长序列设备支配斜率。"""
        if train.empty:
            return train

        per_dev_cap = self.config.get("train_max_rows_per_device_per_control", 2000)
        group_cap = self.config.get("train_max_rows_per_control_group", 200000)
        per_dev_cap = None if per_dev_cap in (None, "", "none", "None") else int(per_dev_cap)
        group_cap = None if group_cap in (None, "", "none", "None") else int(group_cap)

        parts = []
        for dev, sub in train.groupby("device_key", observed=True):
            if per_dev_cap is not None and per_dev_cap > 0 and len(sub) > per_dev_cap:
                seed = self._stable_seed(random_state + repeat * 1009, f"{model_id}|{control_pressure}|{dev}")
                parts.append(sub.sample(n=per_dev_cap, random_state=seed))
            else:
                parts.append(sub)
        out = pd.concat(parts, axis=0, ignore_index=True) if parts else train.iloc[0:0].copy()

        if group_cap is not None and group_cap > 0 and len(out) > group_cap:
            seed = self._stable_seed(random_state + repeat * 1013, f"{model_id}|{control_pressure}|group_cap")
            out = out.sample(n=group_cap, random_state=seed).reset_index(drop=True)
        return out

    @staticmethod
    def _average_metrics_for_report(repeat_metrics: List[Dict], control_pressures: List[float]) -> Dict[str, List[Dict]]:
        """将多次重复实验指标压缩为 evaluation_report_metrics.csv 需要的平均行。"""
        df = pd.DataFrame(repeat_metrics)
        if df.empty:
            return {}

        # 保持截图中的列名和列顺序；repeat 只保留在 JSON 明细中，不进入 CSV。
        count_cols = ["TP", "FP", "TN", "FN"]
        metric_cols = ["precision", "recall", "f1", "fpr", "fnr", "accuracy"]
        out: Dict[str, List[Dict]] = {}

        for model_id, mdf in df.groupby("model", sort=True):
            rows = []
            for cp, sub in mdf.groupby("control_pressure", sort=True):
                row = {
                    "control_pressure": float(cp),
                    "beta_0": float(sub["beta_0"].mean()),
                    "beta_1": float(sub["beta_1"].mean()),
                    "threshold": float(sub["threshold"].mean()),
                }
                for c in count_cols:
                    # 这是重复实验的平均混淆矩阵计数；四舍五入后便于和原截图保持一致。
                    row[c] = int(round(float(sub[c].mean())))
                for c in metric_cols:
                    row[c] = float(sub[c].mean())
                rows.append(row)

            # 按配置中的控制压差顺序输出。
            cp_order = {float(cp): i for i, cp in enumerate(control_pressures)}
            rows = sorted(rows, key=lambda r: cp_order.get(float(r["control_pressure"]), 999))
            out[model_id] = rows
        return out
