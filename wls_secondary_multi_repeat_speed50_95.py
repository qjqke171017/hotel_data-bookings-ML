# -*- coding: utf-8 -*-
"""
WLS 二次侧机理异常检测：多模型、多次重复、分控制压差评估版

核心功能
1. 按控制压差目标值 0.8 / 1.4 / 1.65 分组建模与评价。
2. 重复随机划分健康设备，默认重复 20 次，降低单次 70% 健康设备训练划分带来的随机性。
3. 同时评估三种 WLS 机理模型：
   M1: 二次侧泵压差 ~ (二次侧泵转速/100)^2
   M2: (二次侧板换压差 + 二次侧供回水压差) ~ (二次侧泵转速/100)^2
   M3: (二次侧板换压差 + 二次侧过滤器压差) ~ (二次侧泵转速/100)^2
4. 每个模型、每次重复、每个阈值、每个控制压差组均输出误报率、召回率等指标；同时输出三组控制压差合并后的总体指标。
5. 默认不输出逐记录预测结果，只输出汇总指标与设备级结果，避免 3000 万级记录占用大量磁盘空间。
6. 丰富可视化：阈值指标图、FPR-Recall 权衡图、分控制压差散点图、残差图、score 分布图、设备异常率图等。

数据结构示例：
root_dir/
  N/*.csv
  F01/*.csv, F02/*.csv, F03/*.csv, F04/*.csv, F06/*.csv
  M/*.csv, S/*.csv, U/*.csv
  正常数据分类分工况/  # 自动忽略

重要说明
- N 为正常类。
- F03/F04/F06 为二次侧异常类。
- F01/F02 为一次侧异常，可通过 F01_F02_MODE_CONFIG 设置为剔除或当作二次侧正常。
- M/S/U 可按配置选择是否读取；当前默认完全跳过。
- 同一设备内用于拟合的样本抽样方式：在每次重复中，先按设备划分 normal_train；然后对每个 normal_train 设备、每个控制压差组，在满足转速区间过滤和变量有效的记录中，均匀随机无放回抽取不超过 TRAIN_MAX_ROWS_PER_DEVICE_PER_CONTROL_CONFIG 条记录用于 WLS 拟合。这样可避免长时间序列设备支配回归斜率。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
import hashlib
import gc
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# =========================
# 0. 用户配置区：PyCharm 直接运行主要改这里
# =========================
DATA_ROOT_DIR = r"D:\data\secondary_raw"
OUTPUT_DIR = r"D:\wls_secondary_multi_repeat_output"

# F01/F02 是一次侧异常。二次侧实验有两种处理方式：
# "exclude"：剔除 F01/F02；"as_normal"：当作二次侧正常样本，相当于并入 N。
F01_F02_MODE_CONFIG = "exclude"  # 可选："exclude" 或 "as_normal"
SKIP_M_S_U_CONFIG = True          # False=读取 M/S/U 并打分但不参与主指标；True=完全跳过
CSV_RECURSIVE_CONFIG = True

# 控制压差只用于分组，不作为自变量；只研究这三组，其他取值跳过。
VALID_CONTROL_PRESSURES_CONFIG = "0.8,1.4,1.65"
CONTROL_ROUND_DIGITS_CONFIG = 3

# 重复实验配置
N_REPEATS_CONFIG = 20
TRAIN_NORMAL_DEVICE_FRAC_CONFIG = 0.70
RANDOM_STATE_CONFIG = 42

# 阈值分位数：阈值来自每个控制压差组 normal_train score 的分位数。
THRESHOLD_QUANTILES_CONFIG = "0.9500,0.9750,0.9900,0.9975,0.9999"
MAIN_PLOT_QUANTILE_CONFIG = 0.9975

# WLS 转速区间过滤：若转速为 0-100，则保留 50 <= speed < 95；若为 0-1，则保留 0.50 <= speed < 0.95。
WLS_MIN_SPEED_CONFIG = 50.0
WLS_MAX_SPEED_CONFIG = 95.0

# 内存与抽样控制
# DATA_MAX_ROWS_PER_FILE_CONFIG=None 表示每个 CSV 全量读入并参与评价；如果内存不足，可改成 20000、50000 等。
# 若设置为整数，程序会对每个 CSV 做均匀随机抽样，而不是只取前几行。
DATA_MAX_ROWS_PER_FILE_CONFIG = 5000
CSV_CHUNKSIZE_CONFIG = 200000

# 用于 WLS 拟合的训练样本上限：每个训练设备、每个控制压差组最多抽取这些记录。
TRAIN_MAX_ROWS_PER_DEVICE_PER_CONTROL_CONFIG = 2000
# 每个模型、每个控制压差组拟合时，训练记录总量上限；超过则再次均匀随机抽样。
TRAIN_MAX_ROWS_PER_CONTROL_GROUP_CONFIG = 200000
MIN_TRAIN_ROWS_PER_CONTROL_GROUP_CONFIG = 80

# WLS 异方差估计配置
WLS_BINS_CONFIG = 10
MIN_ROWS_PER_BIN_CONFIG = 30
MIN_SIGMA_CONFIG = 1e-6

# 可视化抽样：只影响散点图实际绘制点数，不影响指标计算。
PLOT_SAMPLE_SIZE_CONFIG = 20000
MIN_POINTS_FOR_GROUP_PLOT_CONFIG = 300

# 默认不输出逐记录预测，避免磁盘占用过大。
SAVE_RECORD_LEVEL_OUTPUT_CONFIG = False
ENABLE_PLOTLY_HTML_CONFIG = False

# =========================
# 1. 精确列名配置区：不做模糊搜索
# =========================
EXACT_COLUMN_NAMES_CONFIG = {
    "二次侧泵转速": "二次侧泵转速",
    "控制压差目标值": "控制压差目标值",
    "二次侧泵压差": "二次侧泵压差",
    "二次侧板换压差": "二次侧板换压差",
    "二次侧供回水压差": "二次侧供回水压差",
    "二次侧过滤器压差": "二次侧过滤器压差",
    # 下列列不直接进入三种 WLS 模型，但保留在配置中便于后续扩展和核对。
    "二次侧入口压力1": "二次侧入口压力1",
    "二次侧入口压力2": "二次侧入口压力2",
    "二次侧出口压力1": "二次侧出口压力1",
    "二次侧出口压力2": "二次侧出口压力2",
    "二次侧泵入口压力1": "二次侧泵入口压力1",
    "二次侧泵入口压力2": "二次侧泵入口压力2",
    "二次侧泵出口压力": "二次侧泵出口压力",
    "二次侧管阻比": "二次侧管阻比",
    "二次侧阀开度": "二次侧阀开度",
    "二次侧入口温度": "二次侧入口温度",
    "二次侧出口温度1": "二次侧出口温度1",
    "二次侧出口温度2": "二次侧出口温度2",
}

# 三种模型的因变量定义。每个模型因变量为若干列之和。
TARGET_MODELS_CONFIG = {
    "M1_pump_dp": {
        "name": "模型1：二次侧泵压差 ~ 转速平方",
        "target_columns": ["二次侧泵压差"],
    },
    "M2_plate_plus_supply_return": {
        "name": "模型2：二次侧板换压差 + 二次侧供回水压差 ~ 转速平方",
        "target_columns": ["二次侧板换压差", "二次侧供回水压差"],
    },
    "M3_plate_plus_filter": {
        "name": "模型3：二次侧板换压差 + 二次侧过滤器压差 ~ 转速平方",
        "target_columns": ["二次侧板换压差", "二次侧过滤器压差"],
    },
}

# =========================
# 2. 常量与绘图设置
# =========================
IGNORE_DIR_NAMES = {"正常数据分类分工况"}
SECONDARY_ANOMALY_CODES = {"F03", "F04", "F06"}
PRIMARY_ANOMALY_CODES = {"F01", "F02"}
UNLABELED_CODES = {"M", "S", "U"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 3. 基础工具函数
# =========================
def parse_list_float(s: str, digits: Optional[int] = None) -> List[float]:
    vals: List[float] = []
    for x in str(s).split(','):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if digits is not None:
            v = round(v, digits)
        vals.append(v)
    return sorted(set(vals))


def control_key_from_values(values: pd.Series, digits: int) -> pd.Series:
    """把控制压差转为整数键，避免 0.8 与 0.8000000119 精确比较失败。"""
    scale = 10 ** int(digits)
    x = pd.to_numeric(values, errors="coerce").astype("float64")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    ok = x.notna()
    out.loc[ok] = np.rint(x.loc[ok] * scale)
    return out


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = (s.str.replace("−", "-", regex=False)
           .str.replace("－", "-", regex=False)
           .str.replace("，", ".", regex=False))
    direct = pd.to_numeric(s, errors="coerce")
    need = direct.isna() & s.notna() & ~s.str.lower().isin(["", "nan", "none", "null"])
    if need.any():
        extracted = s[need].str.extract(r"([-+]?\d+(?:[\.,]\d+)?)", expand=False)
        extracted = extracted.str.replace(",", ".", regex=False)
        direct.loc[need] = pd.to_numeric(extracted, errors="coerce")
    return direct


def stable_seed(base: int, text: str) -> int:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return (int(h[:8], 16) + int(base)) % (2**32 - 1)


def code_to_label(code: str, f01_f02_mode: str) -> Optional[int]:
    code = code.upper()
    if code == "N":
        return 0
    if code in SECONDARY_ANOMALY_CODES:
        return 1
    if code in PRIMARY_ANOMALY_CODES:
        return 0 if f01_f02_mode == "as_normal" else None
    return None


def source_label_name(binary_label: Optional[int]) -> str:
    if binary_label == 0:
        return "normal"
    if binary_label == 1:
        return "secondary_anomaly"
    return "unlabeled"


def normalize_speed(speed_raw: pd.Series) -> pd.Series:
    q95 = speed_raw.dropna().quantile(0.95) if speed_raw.notna().any() else np.nan
    if pd.notna(q95) and q95 > 2:
        return speed_raw / 100.0
    return speed_raw


def speed_range_exclusion_mask(speed_raw: pd.Series, wls_min_speed: float, wls_max_speed: float) -> pd.Series:
    """返回 WLS 建模与评价中需要剔除的转速记录。

    当转速列为 0--100 量纲时，保留区间为 [wls_min_speed, wls_max_speed)；
    当转速列已归一化至 0--1 时，自动换算为 [wls_min_speed/100, wls_max_speed/100)。
    返回 True 表示该记录位于保留区间之外，应从 WLS 拟合、阈值和主指标中剔除。
    """
    q95 = speed_raw.dropna().quantile(0.95) if speed_raw.notna().any() else np.nan
    if pd.notna(q95) and q95 > 2:
        low = float(wls_min_speed)
        high = float(wls_max_speed)
    else:
        low = float(wls_min_speed) / 100.0
        high = float(wls_max_speed) / 100.0
    x = pd.to_numeric(speed_raw, errors="coerce")
    return x.isna() | (x < low) | (x >= high)


def required_physical_columns() -> List[str]:
    cols = {"二次侧泵转速", "控制压差目标值"}
    for cfg in TARGET_MODELS_CONFIG.values():
        cols.update(cfg["target_columns"])
    return [EXACT_COLUMN_NAMES_CONFIG[c] for c in cols]


def read_header_columns(path: Path, encoding: str) -> List[str]:
    return list(pd.read_csv(path, encoding=encoding, nrows=0).columns)


def read_csv_selected(path: Path, usecols: List[str], max_rows: Optional[int], chunksize: int, random_state: int) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            header = read_header_columns(path, enc)
            real_usecols = [c for c in usecols if c in header]
            missing = [c for c in usecols if c not in header]
            if missing:
                raise RuntimeError(f"缺少必需列 {missing}")
            if max_rows is None:
                df = pd.read_csv(path, encoding=enc, usecols=real_usecols)
                df["row_id"] = np.arange(len(df), dtype=np.int64)
                return df
            # 均匀随机抽样，而不是只取前 max_rows 行。
            rng = np.random.default_rng(random_state)
            kept: Optional[pd.DataFrame] = None
            offset = 0
            for chunk in pd.read_csv(path, encoding=enc, usecols=real_usecols, chunksize=chunksize):
                n = len(chunk)
                chunk = chunk.copy()
                chunk["row_id"] = np.arange(offset, offset + n, dtype=np.int64)
                chunk["__rand_key__"] = rng.random(n)
                offset += n
                if kept is None:
                    kept = chunk
                else:
                    kept = pd.concat([kept, chunk], ignore_index=True)
                if len(kept) > max_rows:
                    kept = kept.nsmallest(max_rows, "__rand_key__")
            if kept is None:
                return pd.DataFrame(columns=real_usecols + ["row_id"])
            kept = kept.sort_values("row_id").drop(columns=["__rand_key__"])
            return kept.reset_index(drop=True)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无法读取 CSV：{path}；最后错误：{last_err}")

# =========================
# 4. 数据读取和预处理
# =========================
def load_dataset(args: argparse.Namespace, tables_dir: Path) -> pd.DataFrame:
    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir 不存在：{root}")

    required_cols = required_physical_columns()
    allowed_vals = parse_list_float(args.valid_control_pressures, args.control_round_digits)
    scale = 10 ** int(args.control_round_digits)
    allowed_keys = {int(round(v * scale)) for v in allowed_vals}

    rows: List[pd.DataFrame] = []
    file_logs: List[Dict[str, object]] = []
    print(f"[INFO] root_dir = {root.resolve()}")
    print(f"[INFO] 只研究控制压差目标值 = {allowed_vals}")

    code_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.strip() not in IGNORE_DIR_NAMES]
    for code_dir in code_dirs:
        code = code_dir.name.strip().upper()
        if not (code == "N" or code in SECONDARY_ANOMALY_CODES or code in PRIMARY_ANOMALY_CODES or code in UNLABELED_CODES or code.startswith("F")):
            continue
        if code in PRIMARY_ANOMALY_CODES and args.f01_f02_mode == "exclude":
            print(f"[INFO] 跳过 {code}：f01_f02_mode=exclude")
            continue
        if code.startswith("F") and code not in SECONDARY_ANOMALY_CODES and code not in PRIMARY_ANOMALY_CODES:
            print(f"[WARN] 跳过未定义 F 文件夹 {code}：请确认是否属于二次侧异常")
            continue
        if code in UNLABELED_CODES and args.skip_unlabeled:
            print(f"[INFO] 跳过 {code}：skip_unlabeled=True")
            continue

        if args.csv_recursive:
            csv_files = sorted([p for p in code_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"])
        else:
            csv_files = sorted([p for p in code_dir.glob("*") if p.is_file() and p.suffix.lower() == ".csv"])
        print(f"[INFO] 读取 {code}: {len(csv_files)} 个 CSV")

        for fp in csv_files:
            try:
                seed = stable_seed(args.random_state, str(fp))
                raw = read_csv_selected(fp, required_cols, args.data_max_rows_per_file, args.csv_chunksize, seed)
                if raw.empty:
                    continue

                # 精确列名转为统一逻辑名。
                df = pd.DataFrame(index=raw.index)
                for logical, actual in EXACT_COLUMN_NAMES_CONFIG.items():
                    if actual in raw.columns:
                        df[logical] = _to_numeric(raw[actual])
                df["row_id"] = raw["row_id"].astype(np.int64)

                # 只保留指定控制压差，使用整数键避免浮点误差。
                cp_key = control_key_from_values(df["控制压差目标值"], args.control_round_digits)
                keep = cp_key.isin(allowed_keys)
                before = len(df)
                df = df.loc[keep].copy()
                cp_key = cp_key.loc[keep]
                if df.empty:
                    file_logs.append({"source_code": code, "device_name": fp.stem, "file_path": str(fp), "read_rows": before, "kept_rows": 0, "status": "no_valid_control_pressure"})
                    continue

                df["control_pressure_key"] = cp_key.astype(np.int64)
                df["control_group"] = (df["control_pressure_key"] / scale).map(lambda x: f"{float(x):g}")

                # 转速、转速平方、转速区间过滤。
                df["speed_raw"] = df["二次侧泵转速"]
                df["speed_norm"] = normalize_speed(df["speed_raw"])
                df["speed_sq"] = df["speed_norm"] ** 2
                df["excluded_speed_range_for_wls"] = speed_range_exclusion_mask(df["speed_raw"], args.wls_min_speed, args.wls_max_speed)

                # 三种模型因变量。
                for model_id, cfg in TARGET_MODELS_CONFIG.items():
                    cols = cfg["target_columns"]
                    target = df[cols].sum(axis=1, min_count=len(cols))
                    df[f"target__{model_id}"] = target

                label = code_to_label(code, args.f01_f02_mode)
                df["source_code"] = code
                df["device_name"] = fp.stem
                df["file_path"] = str(fp)
                df["device_key"] = code + "/" + fp.stem
                df["binary_label"] = np.nan if label is None else label
                df["label_name"] = source_label_name(label)

                # 只保留后续需要的列，降低内存。
                keep_cols = [
                    "source_code", "device_name", "file_path", "device_key", "row_id", "binary_label", "label_name",
                    "control_group", "control_pressure_key", "控制压差目标值", "speed_raw", "speed_norm", "speed_sq",
                    "excluded_speed_range_for_wls",
                ] + [f"target__{mid}" for mid in TARGET_MODELS_CONFIG]
                df = df[keep_cols]
                rows.append(df.reset_index(drop=True))
                file_logs.append({"source_code": code, "device_name": fp.stem, "file_path": str(fp), "read_rows": before, "kept_rows": len(df), "status": "ok"})
            except Exception as e:
                print(f"[ERROR] 读取失败：{fp}；错误：{e}")
                file_logs.append({"source_code": code, "device_name": fp.stem, "file_path": str(fp), "read_rows": 0, "kept_rows": 0, "status": f"error: {e}"})

    if not rows:
        raise RuntimeError("没有读取到任何可用记录。请检查路径、列名、控制压差目标值。")

    data = pd.concat(rows, ignore_index=True)
    pd.DataFrame(file_logs).to_csv(tables_dir / "read_file_log.csv", index=False, encoding="utf-8-sig")

    # 内存优化。
    for c in ["source_code", "device_name", "file_path", "device_key", "label_name", "control_group"]:
        data[c] = data[c].astype("category")
    for c in data.columns:
        if pd.api.types.is_numeric_dtype(data[c]) and c not in {"row_id", "control_pressure_key"}:
            data[c] = pd.to_numeric(data[c], errors="coerce").astype("float32")

    summary = data.groupby(["control_group", "source_code", "label_name"], dropna=False).size().reset_index(name="records")
    summary.to_csv(tables_dir / "data_summary_by_control_code.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] 实际纳入分析记录数：{len(data):,}")
    print(summary.to_string(index=False, max_rows=40))
    return data

# =========================
# 5. WLS 拟合和打分
# =========================
def robust_sigma(x: np.ndarray, min_sigma: float = 1e-6) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return min_sigma
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig < min_sigma:
        sig = np.std(x)
    if not np.isfinite(sig) or sig < min_sigma:
        sig = min_sigma
    return float(sig)


def wls_solve(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    if w is None:
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    sw = np.sqrt(np.asarray(w, dtype=float))
    Xw = X * sw[:, None]
    yw = y * sw
    return np.linalg.pinv(Xw.T @ Xw) @ Xw.T @ yw


def make_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], x, side="right")


def fit_one_wls(train: pd.DataFrame, target_col: str, args: argparse.Namespace) -> Dict[str, object]:
    x = train["speed_sq"].to_numpy(dtype=float)
    y = train[target_col].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < args.min_train_rows_per_control_group or len(np.unique(x)) < 2:
        raise RuntimeError(f"训练样本不足或转速平方唯一值不足：n={len(x)}, unique_x={len(np.unique(x))}")

    X = np.column_stack([np.ones_like(x), x])
    beta_ols = wls_solve(X, y)
    resid0 = y - X @ beta_ols
    global_sigma0 = robust_sigma(resid0, args.min_sigma)

    edges = make_bins(x, args.wls_bins)
    bid = assign_bins(x, edges)
    sigma0 = []
    for b in range(len(edges) - 1):
        rb = resid0[bid == b]
        sigma0.append(robust_sigma(rb, args.min_sigma) if len(rb) >= args.min_rows_per_bin else global_sigma0)
    sigma0 = np.maximum(np.asarray(sigma0, dtype=float), args.min_sigma)
    w = 1.0 / sigma0[np.clip(bid, 0, len(sigma0)-1)] ** 2

    beta = wls_solve(X, y, w)
    resid = y - X @ beta
    global_sigma = robust_sigma(resid, args.min_sigma)
    bid = assign_bins(x, edges)
    sigma_final = []
    for b in range(len(edges) - 1):
        rb = resid[bid == b]
        sigma_final.append(robust_sigma(rb, args.min_sigma) if len(rb) >= args.min_rows_per_bin else global_sigma)
    sigma_final = np.maximum(np.asarray(sigma_final, dtype=float), args.min_sigma)

    return {
        "n_train": int(len(x)),
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "bin_edges": edges.tolist(),
        "sigma_by_bin": sigma_final.tolist(),
        "global_sigma": float(global_sigma),
        "x_min": float(np.nanmin(x)),
        "x_max": float(np.nanmax(x)),
    }


def predict_model(x: np.ndarray, model: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    beta0 = float(model["beta0"])
    beta1 = float(model["beta1"])
    yhat = beta0 + beta1 * x
    edges = np.asarray(model["bin_edges"], dtype=float)
    sigmas = np.asarray(model["sigma_by_bin"], dtype=float)
    bid = assign_bins(x, edges)
    bid = np.clip(bid, 0, len(sigmas) - 1)
    return yhat, sigmas[bid]


def split_normal_devices(data: pd.DataFrame, repeat: int, args: argparse.Namespace) -> pd.Series:
    rng = np.random.default_rng(args.random_state + repeat * 1009)
    normal_keys = np.array(sorted(data.loc[data["binary_label"].eq(0), "device_key"].dropna().astype(str).unique()))
    if len(normal_keys) < 2:
        raise RuntimeError("正常设备数量少于 2，无法做设备级训练/留出划分。")
    rng.shuffle(normal_keys)
    n_train = max(1, int(round(len(normal_keys) * args.train_normal_device_frac)))
    n_train = min(n_train, len(normal_keys) - 1)
    train_keys = set(normal_keys[:n_train])

    split = pd.Series("unlabeled", index=data.index, dtype="object")
    device_str = data["device_key"].astype(str)
    split.loc[data["binary_label"].eq(1)] = "anomaly"
    split.loc[data["binary_label"].eq(0) & device_str.isin(train_keys)] = "normal_train"
    split.loc[data["binary_label"].eq(0) & ~device_str.isin(train_keys)] = "normal_heldout"
    return split


def sample_train_rows(train: pd.DataFrame, repeat: int, model_id: str, control_group: str, args: argparse.Namespace) -> pd.DataFrame:
    if train.empty:
        return train
    parts = []
    cap = args.train_max_rows_per_device_per_control
    for dev, sub in train.groupby("device_key", observed=True):
        if cap is not None and len(sub) > cap:
            seed = stable_seed(args.random_state + repeat * 1009, f"{model_id}|{control_group}|{dev}")
            parts.append(sub.sample(n=cap, random_state=seed))
        else:
            parts.append(sub)
    out = pd.concat(parts, axis=0) if parts else train.iloc[0:0]
    if args.train_max_rows_per_control_group and len(out) > args.train_max_rows_per_control_group:
        seed = stable_seed(args.random_state + repeat * 1013, f"{model_id}|{control_group}|group_cap")
        out = out.sample(n=args.train_max_rows_per_control_group, random_state=seed)
    return out


def fit_models_for_repeat(data: pd.DataFrame, split: pd.Series, model_id: str, args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    target_col = f"target__{model_id}"
    models: Dict[str, Dict[str, object]] = {}
    fit_rows = []
    for grp in sorted(data["control_group"].dropna().astype(str).unique()):
        mask = (
            split.eq("normal_train") &
            data["control_group"].astype(str).eq(grp) &
            (~data["excluded_speed_range_for_wls"].astype(bool)) &
            data["speed_sq"].notna() &
            data[target_col].notna()
        )
        train = data.loc[mask, ["device_key", "speed_sq", target_col]].copy()
        train_sampled = sample_train_rows(train, 0, model_id, grp, args)
        try:
            model = fit_one_wls(train_sampled, target_col, args)
            model["control_group"] = grp
            model["target_col"] = target_col
            models[grp] = model
            fit_rows.append({"model_id": model_id, "control_group": grp, "fit_status": "ok", "train_rows_before_sample": len(train), "train_rows_used": len(train_sampled), "beta0": model["beta0"], "beta1": model["beta1"]})
        except Exception as e:
            fit_rows.append({"model_id": model_id, "control_group": grp, "fit_status": f"failed: {e}", "train_rows_before_sample": len(train), "train_rows_used": len(train_sampled), "beta0": np.nan, "beta1": np.nan})
            print(f"[WARN] {model_id}, control={grp} 拟合失败：{e}")
    return models, pd.DataFrame(fit_rows)


def score_model(data: pd.DataFrame, model_id: str, models: Dict[str, Dict[str, object]], args: argparse.Namespace) -> pd.DataFrame:
    target_col = f"target__{model_id}"
    out = pd.DataFrame(index=data.index)
    out["score"] = np.nan
    out["residual"] = np.nan
    out["yhat"] = np.nan
    out["sigma"] = np.nan
    out["model_available"] = False
    valid_common = (~data["excluded_speed_range_for_wls"].astype(bool)) & data["speed_sq"].notna() & data[target_col].notna()
    for grp, model in models.items():
        idx = data.index[valid_common & data["control_group"].astype(str).eq(str(grp))]
        if len(idx) == 0:
            continue
        x = data.loc[idx, "speed_sq"].to_numpy(dtype=float)
        y = data.loc[idx, target_col].to_numpy(dtype=float)
        yhat, sigma = predict_model(x, model)
        resid = y - yhat
        score = np.abs(resid) / np.maximum(sigma, args.min_sigma)
        out.loc[idx, "score"] = score
        out.loc[idx, "residual"] = resid
        out.loc[idx, "yhat"] = yhat
        out.loc[idx, "sigma"] = sigma
        out.loc[idx, "model_available"] = True
    return out


def thresholds_by_control(data: pd.DataFrame, split: pd.Series, score: pd.Series, quantiles: Sequence[float]) -> Dict[str, Dict[float, float]]:
    thrs: Dict[str, Dict[float, float]] = {}
    for grp in sorted(data["control_group"].dropna().astype(str).unique()):
        s = score.loc[split.eq("normal_train") & data["control_group"].astype(str).eq(grp)].dropna()
        if len(s) == 0:
            thrs[grp] = {q: np.nan for q in quantiles}
        else:
            thrs[grp] = {q: float(np.quantile(s, q)) for q in quantiles}
    return thrs

# =========================
# 6. 指标与设备级汇总
# =========================
def safe_auc(y_true: np.ndarray, score: np.ndarray, kind: str) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        if kind == "roc":
            return float(roc_auc_score(y_true, score))
        return float(average_precision_score(y_true, score))
    except Exception:
        return np.nan


def metric_from_arrays(y_true: np.ndarray, score: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    mask = np.isfinite(y_true) & np.isfinite(score) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(int)
    score = score[mask]
    y_pred = y_pred[mask].astype(int)
    if len(y_true) == 0:
        return {k: np.nan for k in ["n", "normal_count", "anomaly_count", "tp", "fp", "tn", "fn", "fpr", "recall", "precision", "accuracy", "f1", "mcc", "balanced_accuracy", "auc_roc", "auc_pr"]}
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    normal_count = tn + fp
    anomaly_count = tp + fn
    pred_pos = tp + fp
    pred_neg = tn + fn
    n = len(y_true)
    fpr = fp / normal_count if normal_count else np.nan
    recall = tp / anomaly_count if anomaly_count else np.nan
    precision = tp / pred_pos if pred_pos else np.nan
    accuracy = (tp + tn) / n if n else np.nan
    f1 = 2 * precision * recall / (precision + recall) if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0 else np.nan
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = ((tp*tn - fp*fn) / denom) if denom > 0 else np.nan
    tnr = tn / normal_count if normal_count else np.nan
    balanced_accuracy = np.nanmean([recall, tnr]) if (pd.notna(recall) or pd.notna(tnr)) else np.nan
    return {
        "n": int(n), "normal_count": int(normal_count), "anomaly_count": int(anomaly_count),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr": float(fpr) if pd.notna(fpr) else np.nan,
        "recall": float(recall) if pd.notna(recall) else np.nan,
        "precision": float(precision) if pd.notna(precision) else np.nan,
        "accuracy": float(accuracy) if pd.notna(accuracy) else np.nan,
        "f1": float(f1) if pd.notna(f1) else np.nan,
        "mcc": float(mcc) if pd.notna(mcc) else np.nan,
        "balanced_accuracy": float(balanced_accuracy) if pd.notna(balanced_accuracy) else np.nan,
        "auc_roc": safe_auc(y_true, score, "roc"),
        "auc_pr": safe_auc(y_true, score, "pr"),
    }


def max_consecutive_ones(arr: Sequence[int]) -> int:
    best = cur = 0
    for v in arr:
        if int(v) == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def build_metrics_and_device_summary(
    data: pd.DataFrame,
    split: pd.Series,
    score_df: pd.DataFrame,
    model_id: str,
    model_name: str,
    repeat: int,
    thresholds: Dict[str, Dict[float, float]],
    quantiles: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, object]] = []
    code_rows: List[Dict[str, object]] = []
    device_rows: List[Dict[str, object]] = []

    labeled = data["binary_label"].notna()
    scopes = {
        "heldout": split.isin(["normal_heldout", "anomaly"]),
        "all_labeled": labeled,
    }

    y_true_all = data["binary_label"].to_numpy(dtype=float)
    score_all = score_df["score"].to_numpy(dtype=float)

    for q in quantiles:
        # 每条记录使用其控制压差组自己的阈值。
        row_threshold = np.full(len(data), np.nan, dtype=float)
        for grp, thr_map in thresholds.items():
            thr = thr_map.get(q, np.nan)
            idx = data.index[data["control_group"].astype(str).eq(str(grp))]
            row_threshold[idx] = thr
        y_pred_all = (score_all >= row_threshold).astype(float)
        y_pred_all[~np.isfinite(score_all) | ~np.isfinite(row_threshold)] = np.nan

        for scope_name, scope_mask in scopes.items():
            # all: 三个控制压差组一起计算，但每组阈值不同。
            mask = scope_mask & labeled
            m = metric_from_arrays(y_true_all[mask.values], score_all[mask.values], y_pred_all[mask.values])
            row = {"repeat": repeat, "model_id": model_id, "model_name": model_name, "eval_scope": scope_name, "control_group": "all", "threshold_quantile": q, "threshold_note": "group_specific", **m}
            for grp in sorted(thresholds):
                row[f"threshold_cp_{grp}"] = thresholds[grp].get(q, np.nan)
            metric_rows.append(row)

            # 分控制压差组。
            for grp in sorted(thresholds):
                mask_g = mask & data["control_group"].astype(str).eq(str(grp))
                m = metric_from_arrays(y_true_all[mask_g.values], score_all[mask_g.values], y_pred_all[mask_g.values])
                metric_rows.append({
                    "repeat": repeat, "model_id": model_id, "model_name": model_name,
                    "eval_scope": scope_name, "control_group": grp, "threshold_quantile": q,
                    "threshold": thresholds[grp].get(q, np.nan), **m
                })

        # 按文件夹代码 + 控制压差统计。
        tmp = data[["source_code", "control_group", "binary_label", "row_id"]].copy()
        tmp["score"] = score_df["score"].values
        tmp["pred"] = y_pred_all
        for (grp, code), sub_idx in tmp.groupby(["control_group", "source_code"], observed=True).groups.items():
            idx = list(sub_idx)
            sub = tmp.loc[idx]
            valid = sub["score"].notna() & sub["pred"].notna()
            pred_abn = int(sub.loc[valid, "pred"].sum()) if valid.any() else 0
            valid_n = int(valid.sum())
            row = {
                "repeat": repeat, "model_id": model_id, "model_name": model_name,
                "threshold_quantile": q, "control_group": str(grp), "source_code": str(code),
                "total_records": int(len(sub)), "valid_scored_records": valid_n,
                "normal_record_count": int(sub["binary_label"].eq(0).sum()),
                "secondary_anomaly_record_count": int(sub["binary_label"].eq(1).sum()),
                "unlabeled_record_count": int(sub["binary_label"].isna().sum()),
                "pred_abnormal_count": pred_abn,
                "pred_abnormal_rate": pred_abn / valid_n if valid_n else np.nan,
            }
            if sub["binary_label"].notna().any():
                mm = metric_from_arrays(sub["binary_label"].to_numpy(dtype=float), sub["score"].to_numpy(dtype=float), sub["pred"].to_numpy(dtype=float))
                row.update({f"metric_{k}": v for k, v in mm.items()})
            code_rows.append(row)

        # 设备级统计。
        dev_tmp = data[["source_code", "device_name", "device_key", "control_group", "binary_label", "label_name", "row_id", "excluded_speed_range_for_wls"]].copy()
        dev_tmp["score"] = score_df["score"].values
        dev_tmp["pred"] = y_pred_all
        for (code, dev, grp), sub in dev_tmp.groupby(["source_code", "device_name", "control_group"], observed=True):
            s = sub.sort_values("row_id")
            valid = s["score"].notna() & s["pred"].notna()
            pred = s.loc[valid, "pred"].astype(int).values
            valid_n = int(valid.sum())
            abn = int(pred.sum()) if len(pred) else 0
            device_rows.append({
                "repeat": repeat, "model_id": model_id, "model_name": model_name,
                "threshold_quantile": q, "control_group": str(grp),
                "source_code": str(code), "device_name": str(dev), "device_key": f"{code}/{dev}",
                "label_name": str(s["label_name"].iloc[0]) if len(s) else "",
                "total_records": int(len(s)),
                "valid_scored_records": valid_n,
                "excluded_speed_range_count": int(s["excluded_speed_range_for_wls"].astype(bool).sum()),
                "normal_record_count": int(s["binary_label"].eq(0).sum()),
                "secondary_anomaly_record_count": int(s["binary_label"].eq(1).sum()),
                "unlabeled_record_count": int(s["binary_label"].isna().sum()),
                "pred_normal_count": int(valid_n - abn),
                "pred_abnormal_count": abn,
                "pred_abnormal_rate": abn / valid_n if valid_n else np.nan,
                "max_consecutive_abnormal": max_consecutive_ones(pred) if len(pred) else 0,
                "mean_score": float(s["score"].mean(skipna=True)) if valid_n else np.nan,
                "median_score": float(s["score"].median(skipna=True)) if valid_n else np.nan,
                "p95_score": float(s["score"].quantile(0.95)) if valid_n else np.nan,
                "p99_score": float(s["score"].quantile(0.99)) if valid_n else np.nan,
            })

    return pd.DataFrame(metric_rows), pd.DataFrame(code_rows), pd.DataFrame(device_rows)


def summarize_repeats(metrics: pd.DataFrame) -> pd.DataFrame:
    """按模型、阈值、评价口径和控制压差汇总重复实验结果。

    除均值、标准差、最小值和最大值外，额外统计每个组合在全部重复实验中
    同时满足 FPR < 0.05 与 Recall > 0.50 的次数和比例。该字段用于判断
    候选模型是否只是均值达标，还是在重复划分中稳定达标。
    """
    numeric_cols = ["n", "normal_count", "anomaly_count", "tp", "fp", "tn", "fn", "fpr", "recall", "precision", "accuracy", "f1", "mcc", "balanced_accuracy", "auc_roc", "auc_pr"]
    group_cols = ["model_id", "model_name", "eval_scope", "control_group", "threshold_quantile"]
    rows = []
    for key, sub in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        repeat_count = int(sub["repeat"].nunique())
        row["repeat_count"] = repeat_count
        pass_mask = (sub["fpr"] < 0.05) & (sub["recall"] > 0.50)
        row["delivery_pass_count"] = int(pass_mask.sum())
        row["delivery_pass_rate"] = float(pass_mask.mean()) if len(pass_mask) else np.nan
        for c in numeric_cols:
            if c in sub.columns:
                row[f"{c}_mean"] = sub[c].mean(skipna=True)
                row[f"{c}_std"] = sub[c].std(skipna=True)
                row[f"{c}_min"] = sub[c].min(skipna=True)
                row[f"{c}_max"] = sub[c].max(skipna=True)
        row["meets_fpr_lt_0_05_recall_gt_0_50_mean"] = bool((row.get("fpr_mean", np.inf) < 0.05) and (row.get("recall_mean", -np.inf) > 0.50)) if pd.notna(row.get("recall_mean", np.nan)) else False
        rows.append(row)
    return pd.DataFrame(rows)

# =========================
# 7. 可视化
# =========================
def count_box_text(df: pd.DataFrame, score: Optional[pd.Series] = None) -> str:
    total = len(df)
    if score is None:
        valid = total
    else:
        valid = int(score.notna().sum())
    normal = int(df["binary_label"].eq(0).sum())
    anomaly = int(df["binary_label"].eq(1).sum())
    unlabeled = int(df["binary_label"].isna().sum())
    pct = lambda n: 100*n/total if total else 0
    return f"全部 {total:,}；有效打分 {valid:,}\n正常 {normal:,} ({pct(normal):.1f}%)；异常 {anomaly:,} ({pct(anomaly):.1f}%)；未标注 {unlabeled:,} ({pct(unlabeled):.1f}%)"


def add_count_box(ax, text: str) -> None:
    ax.text(0.01, 0.99, text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="gray"))


def sample_plot_df(df: pd.DataFrame, n: int, seed: int, stratify_col: str = "source_code") -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    parts = []
    vc = df[stratify_col].astype(str).value_counts()
    for val, cnt in vc.items():
        take = max(1, int(round(n * cnt / len(df))))
        sub = df[df[stratify_col].astype(str).eq(val)]
        parts.append(sub.sample(n=min(take, len(sub)), random_state=stable_seed(seed, val)))
    out = pd.concat(parts, axis=0)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out


def plot_metric_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    sub = summary[(summary["eval_scope"] == "heldout") & (summary["control_group"].astype(str) == "all")].copy()
    if sub.empty:
        return
    for model_id, mdf in sub.groupby("model_id"):
        mdf = mdf.sort_values("threshold_quantile")
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in ["fpr", "recall", "precision", "accuracy", "f1", "mcc", "balanced_accuracy"]:
            col = f"{metric}_mean"
            if col in mdf.columns:
                ax.plot(mdf["threshold_quantile"], mdf[col], marker="o", label=metric)
        ax.axhline(0.05, linestyle="--", linewidth=1, label="FPR=0.05")
        ax.axhline(0.50, linestyle=":", linewidth=1, label="Recall=0.50")
        ax.set_title(f"{model_id}：20次重复平均指标 - heldout/all")
        ax.set_xlabel("阈值分位数")
        ax.set_ylabel("指标均值")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"metrics_mean_vs_threshold_{model_id}.png", dpi=220)
        plt.close(fig)
        gc.collect()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(mdf["fpr_mean"], mdf["recall_mean"], s=80)
        for _, r in mdf.iterrows():
            ax.annotate(f"{r['threshold_quantile']:.4f}", (r["fpr_mean"], r["recall_mean"]), textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.axvline(0.05, linestyle="--", linewidth=1)
        ax.axhline(0.50, linestyle=":", linewidth=1)
        ax.set_xlabel("FPR 均值")
        ax.set_ylabel("Recall 均值")
        ax.set_title(f"{model_id}：FPR-Recall 权衡（20次均值）")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fpr_recall_tradeoff_mean_{model_id}.png", dpi=220)
        plt.close(fig)
        gc.collect()


def plot_diagnostic_for_repeat0(data: pd.DataFrame, split: pd.Series, model_id: str, model_name: str, models: Dict[str, Dict[str, object]], score_df: pd.DataFrame, thresholds: Dict[str, Dict[float, float]], args: argparse.Namespace, fig_dir: Path, html_dir: Path) -> None:
    target_col = f"target__{model_id}"
    q = args.main_plot_quantile
    if q not in list(next(iter(thresholds.values())).keys()):
        q = sorted(next(iter(thresholds.values())).keys())[-1]
    base = data.copy()
    base["score"] = score_df["score"]
    base["residual"] = score_df["residual"]
    base["yhat"] = score_df["yhat"]
    base["sigma"] = score_df["sigma"]
    base["target_y"] = data[target_col]
    finite_base = (
        base["score"].notna() & np.isfinite(base["score"].astype(float)) &
        base["speed_sq"].notna() & np.isfinite(base["speed_sq"].astype(float)) &
        base["speed_raw"].notna() & np.isfinite(base["speed_raw"].astype(float)) &
        base["target_y"].notna() & np.isfinite(base["target_y"].astype(float)) &
        base["residual"].notna() & np.isfinite(base["residual"].astype(float)) &
        base["yhat"].notna() & np.isfinite(base["yhat"].astype(float)) &
        base["sigma"].notna() & np.isfinite(base["sigma"].astype(float))
    )
    base = base.loc[finite_base].copy()
    if base.empty:
        return
    base["真实类别"] = np.where(base["binary_label"].eq(0), "正常", np.where(base["binary_label"].eq(1), "二次侧异常", "未标注"))
    base["split"] = split.reindex(base.index).astype(str).values

    for grp in sorted(base["control_group"].astype(str).unique()):
        sub_full = base[base["control_group"].astype(str).eq(grp)].copy()
        if len(sub_full) < args.min_points_for_group_plot:
            continue
        thr = thresholds.get(grp, {}).get(q, np.nan)
        sub_full["判定"] = np.where(sub_full["score"] >= thr, "判异常", "判正常")
        sub_plot = sample_plot_df(sub_full, min(args.plot_sample_size, len(sub_full)), args.random_state + 123)
        safe_grp = re.sub(r"[^0-9A-Za-z_\-.]+", "_", str(grp))
        mdir = fig_dir / model_id / f"cp_{safe_grp}"
        mdir.mkdir(parents=True, exist_ok=True)

        # 真实标签散点 + 拟合线/阈值带
        fig, ax = plt.subplots(figsize=(10, 7))
        for label, ss in sub_plot.groupby("真实类别"):
            ax.scatter(ss["speed_sq"], ss["target_y"], s=6, alpha=0.38, label=f"{label} plotted={len(ss):,}")
        model = models.get(grp)
        if model is not None:
            x_min = float(sub_full["speed_sq"].quantile(0.01))
            x_max = float(sub_full["speed_sq"].quantile(0.99))
            grid = np.linspace(x_min, x_max, 250)
            yhat, sigma = predict_model(grid, model)
            ax.plot(grid, yhat, linewidth=2, label="WLS拟合线")
            if np.isfinite(thr):
                ax.plot(grid, yhat + thr * sigma, linestyle="--", linewidth=1, label="阈值带上界")
                ax.plot(grid, yhat - thr * sigma, linestyle="--", linewidth=1, label="阈值带下界")
        ax.set_xlabel("(二次侧泵转速/100)^2")
        ax.set_ylabel(model_name.split("：", 1)[-1].split("~")[0].strip())
        ax.set_title(f"{model_id} | 控制压差={grp} | 真实标签散点 q={q:.4f}")
        add_count_box(ax, count_box_text(sub_full, sub_full["score"]) + f"\n实际绘制 {len(sub_plot):,} 点；阈值={thr:.4g}")
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(mdir / f"scatter_true_label_{model_id}_cp_{safe_grp}_q{int(q*10000):04d}.png", dpi=220)
        plt.close(fig)
        gc.collect()

        # 预测标签散点
        fig, ax = plt.subplots(figsize=(10, 7))
        for label, ss in sub_plot.groupby("判定"):
            ax.scatter(ss["speed_sq"], ss["target_y"], s=6, alpha=0.38, label=f"{label} plotted={len(ss):,}")
        if model is not None:
            x_min = float(sub_full["speed_sq"].quantile(0.01))
            x_max = float(sub_full["speed_sq"].quantile(0.99))
            grid = np.linspace(x_min, x_max, 250)
            yhat, sigma = predict_model(grid, model)
            ax.plot(grid, yhat, linewidth=2, label="WLS拟合线")
            if np.isfinite(thr):
                ax.plot(grid, yhat + thr * sigma, linestyle="--", linewidth=1)
                ax.plot(grid, yhat - thr * sigma, linestyle="--", linewidth=1)
        ax.set_xlabel("(二次侧泵转速/100)^2")
        ax.set_ylabel(model_name.split("：", 1)[-1].split("~")[0].strip())
        ax.set_title(f"{model_id} | 控制压差={grp} | 模型判定散点 q={q:.4f}")
        add_count_box(ax, count_box_text(sub_full, sub_full["score"]) + f"\n实际绘制 {len(sub_plot):,} 点；阈值={thr:.4g}")
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(mdir / f"scatter_prediction_{model_id}_cp_{safe_grp}_q{int(q*10000):04d}.png", dpi=220)
        plt.close(fig)
        gc.collect()

        # score 分布
        fig, ax = plt.subplots(figsize=(10, 6))
        normal = sub_full.loc[sub_full["binary_label"].eq(0), "score"].to_numpy(dtype=float)
        anomaly = sub_full.loc[sub_full["binary_label"].eq(1), "score"].to_numpy(dtype=float)
        normal = normal[np.isfinite(normal)]
        anomaly = anomaly[np.isfinite(anomaly)]
        all_scores = np.concatenate([normal, anomaly]) if (len(normal) + len(anomaly)) else np.array([], dtype=float)
        finite_thresholds = np.array([v for v in thresholds.get(grp, {}).values() if np.isfinite(v)], dtype=float)
        if len(all_scores):
            upper = float(np.nanquantile(all_scores, 0.995))
            if len(finite_thresholds):
                upper = max(upper, float(np.nanmax(finite_thresholds)) * 1.10)
            upper = max(upper, 1.0)
            normal_plot = normal[normal <= upper]
            anomaly_plot = anomaly[anomaly <= upper]
        else:
            upper = 1.0
            normal_plot = normal
            anomaly_plot = anomaly
        if len(normal_plot):
            ax.hist(normal_plot, bins=80, range=(0, upper), alpha=0.55, density=True, label=f"正常 n={len(normal):,}")
        if len(anomaly_plot):
            ax.hist(anomaly_plot, bins=80, range=(0, upper), alpha=0.55, density=True, label=f"异常 n={len(anomaly):,}")
        for qq, tt in thresholds.get(grp, {}).items():
            if np.isfinite(tt) and tt <= upper:
                ax.axvline(tt, linestyle="--", linewidth=1, label=f"q={qq:.4f}")
        ax.set_xlim(0, upper)
        ax.set_xlabel("WLS score = |残差| / 局部σ")
        ax.set_ylabel("密度")
        ax.set_title(f"{model_id} | 控制压差={grp} | score 分布")
        clipped_n = int(((normal > upper).sum() if len(normal) else 0) + ((anomaly > upper).sum() if len(anomaly) else 0))
        clip_note = f"\n绘图上限={upper:.3g}；超出上限 {clipped_n:,} 点"
        add_count_box(ax, count_box_text(sub_full, sub_full["score"]) + clip_note)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(mdir / f"score_distribution_{model_id}_cp_{safe_grp}.png", dpi=220)
        plt.close(fig)
        gc.collect()

        # 残差 vs 转速、score vs 转速
        for ycol, ylabel, fname in [("residual", "WLS 残差", "residual_vs_speed"), ("score", "WLS score", "score_vs_speed")]:
            fig, ax = plt.subplots(figsize=(10, 6))
            for label, ss in sub_plot.groupby("真实类别"):
                ax.scatter(ss["speed_raw"], ss[ycol], s=6, alpha=0.38, label=f"{label} plotted={len(ss):,}")
            if ycol == "residual":
                ax.axhline(0, linestyle="--", linewidth=1)
            else:
                for qq, tt in thresholds.get(grp, {}).items():
                    ax.axhline(tt, linestyle="--", linewidth=1, label=f"q={qq:.4f}")
            if ycol == "score":
                yvals = sub_full["score"].to_numpy(dtype=float)
                yvals = yvals[np.isfinite(yvals)]
                if len(yvals):
                    y_upper = max(float(np.nanquantile(yvals, 0.995)), 1.0)
                    finite_thresholds = np.array([v for v in thresholds.get(grp, {}).values() if np.isfinite(v)], dtype=float)
                    if len(finite_thresholds):
                        y_upper = max(y_upper, float(np.nanmax(finite_thresholds)) * 1.10)
                    ax.set_ylim(0, y_upper)
            elif ycol == "residual":
                yvals = sub_full["residual"].to_numpy(dtype=float)
                yvals = yvals[np.isfinite(yvals)]
                if len(yvals) >= 10:
                    lo = float(np.nanquantile(yvals, 0.005))
                    hi = float(np.nanquantile(yvals, 0.995))
                    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                        pad = 0.05 * (hi - lo)
                        ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel("二次侧泵转速")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model_id} | 控制压差={grp} | {ylabel} vs 转速")
            add_count_box(ax, count_box_text(sub_full, sub_full["score"]) + f"\n实际绘制 {len(sub_plot):,} 点")
            ax.legend(fontsize=8, markerscale=2)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(mdir / f"{fname}_{model_id}_cp_{safe_grp}.png", dpi=220)
            plt.close(fig)
        gc.collect()

        if PLOTLY_AVAILABLE and getattr(args, "enable_plotly_html", False):
            hdir = html_dir / model_id
            hdir.mkdir(parents=True, exist_ok=True)
            fig = px.scatter(
                sub_plot, x="speed_sq", y="target_y", color="真实类别",
                hover_data=["source_code", "device_name", "row_id", "score", "residual", "判定", "split"],
                title=f"{model_id} 控制压差={grp} 交互散点：全量 {len(sub_full):,}，绘制 {len(sub_plot):,}", opacity=0.55,
            )
            fig.write_html(hdir / f"interactive_scatter_true_label_cp_{safe_grp}.html")


def plot_device_top(device_df: pd.DataFrame, fig_dir: Path, html_dir: Path, q: float, enable_html: bool = False) -> None:
    """绘制正常文件夹 N 中误报率最高的设备图，并输出设备级误报汇总。

    异常文件夹中的设备本就应该被模型识别为异常，因此不进入正常设备误报 Top 图。
    本函数只使用 source_code == 'N' 的设备结果，并按模型、阈值、设备和控制压差
    汇总重复实验均值，便于定位正常样本中误报较集中的设备。
    """
    if device_df.empty:
        return

    normal = device_df[device_df["source_code"].astype(str).eq("N")].copy()
    if normal.empty:
        return

    tables_dir = fig_dir.parent / "tables"
    group_cols = ["model_id", "model_name", "threshold_quantile", "control_group", "source_code", "device_name", "device_key"]
    normal_summary = (
        normal.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            repeat_count=("repeat", "nunique"),
            valid_scored_records_mean=("valid_scored_records", "mean"),
            pred_abnormal_count_mean=("pred_abnormal_count", "mean"),
            false_alarm_rate_mean=("pred_abnormal_rate", "mean"),
            false_alarm_rate_std=("pred_abnormal_rate", "std"),
            false_alarm_rate_max=("pred_abnormal_rate", "max"),
            max_consecutive_abnormal_mean=("max_consecutive_abnormal", "mean"),
            max_consecutive_abnormal_max=("max_consecutive_abnormal", "max"),
            mean_score_mean=("mean_score", "mean"),
            p95_score_mean=("p95_score", "mean"),
            p99_score_mean=("p99_score", "mean"),
        )
    )
    normal_summary = normal_summary.sort_values(["model_id", "threshold_quantile", "false_alarm_rate_mean"], ascending=[True, True, False])
    normal_summary.to_csv(tables_dir / "wls_normal_device_false_alarm_by_device.csv", index=False, encoding="utf-8-sig")

    sub = normal_summary[np.isclose(normal_summary["threshold_quantile"], q)].copy()
    if sub.empty:
        return

    for model_id, mdf in sub.groupby("model_id"):
        top = mdf.sort_values("false_alarm_rate_mean", ascending=False).head(30)
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 8))
        labels = (top["device_key"].astype(str) + " | cp=" + top["control_group"].astype(str)).iloc[::-1]
        ax.barh(labels, top["false_alarm_rate_mean"].values[::-1])
        ax.set_xlabel("正常设备平均误报率")
        ax.set_title(f"{model_id}：N 文件夹 Top 30 正常设备误报率（q={q:.4f}）")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"top30_N_device_false_alarm_rate_{model_id}_q{int(q*10000):04d}.png", dpi=220)
        plt.close(fig)
        gc.collect()

        topc = mdf.sort_values("max_consecutive_abnormal_max", ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(12, 8))
        labels = (topc["device_key"].astype(str) + " | cp=" + topc["control_group"].astype(str)).iloc[::-1]
        ax.barh(labels, topc["max_consecutive_abnormal_max"].values[::-1])
        ax.axvline(10, linestyle="--", linewidth=1, label="连续误报 10 次参考线")
        ax.set_xlabel("最大连续误报次数")
        ax.set_title(f"{model_id}：N 文件夹 Top 30 最大连续误报次数（q={q:.4f}）")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"top30_N_device_max_consecutive_false_alarm_{model_id}_q{int(q*10000):04d}.png", dpi=220)
        plt.close(fig)
        gc.collect()

        if PLOTLY_AVAILABLE and enable_html:
            hdir = html_dir / "normal_device_false_alarm"
            hdir.mkdir(parents=True, exist_ok=True)
            top_html = mdf.sort_values("false_alarm_rate_mean", ascending=False).head(300)
            fig = px.bar(
                top_html.sort_values("false_alarm_rate_mean"),
                x="false_alarm_rate_mean", y="device_key", color="control_group", orientation="h",
                hover_data=["valid_scored_records_mean", "pred_abnormal_count_mean", "max_consecutive_abnormal_max", "mean_score_mean", "p95_score_mean"],
                title=f"{model_id}：N 文件夹正常设备误报率 Top 300（q={q:.4f}）",
            )
            fig.write_html(hdir / f"N_device_false_alarm_rate_top300_{model_id}.html")

# =========================
# 8. 主流程
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="WLS 二次侧多模型多次重复异常检测")
    parser.add_argument("--root_dir", default=DATA_ROOT_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--f01_f02_mode", choices=["exclude", "as_normal"], default=F01_F02_MODE_CONFIG)
    parser.add_argument("--skip_unlabeled", action="store_true", default=SKIP_M_S_U_CONFIG)
    parser.add_argument("--csv_recursive", action="store_true", default=CSV_RECURSIVE_CONFIG)
    parser.add_argument("--valid_control_pressures", default=VALID_CONTROL_PRESSURES_CONFIG)
    parser.add_argument("--control_round_digits", type=int, default=CONTROL_ROUND_DIGITS_CONFIG)
    parser.add_argument("--n_repeats", type=int, default=N_REPEATS_CONFIG)
    parser.add_argument("--train_normal_device_frac", type=float, default=TRAIN_NORMAL_DEVICE_FRAC_CONFIG)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE_CONFIG)
    parser.add_argument("--threshold_quantiles", default=THRESHOLD_QUANTILES_CONFIG)
    parser.add_argument("--main_plot_quantile", type=float, default=MAIN_PLOT_QUANTILE_CONFIG)
    parser.add_argument("--wls_min_speed", type=float, default=WLS_MIN_SPEED_CONFIG)
    parser.add_argument("--wls_max_speed", type=float, default=WLS_MAX_SPEED_CONFIG)
    parser.add_argument("--data_max_rows_per_file", type=int, default=DATA_MAX_ROWS_PER_FILE_CONFIG)
    parser.add_argument("--csv_chunksize", type=int, default=CSV_CHUNKSIZE_CONFIG)
    parser.add_argument("--train_max_rows_per_device_per_control", type=int, default=TRAIN_MAX_ROWS_PER_DEVICE_PER_CONTROL_CONFIG)
    parser.add_argument("--train_max_rows_per_control_group", type=int, default=TRAIN_MAX_ROWS_PER_CONTROL_GROUP_CONFIG)
    parser.add_argument("--min_train_rows_per_control_group", type=int, default=MIN_TRAIN_ROWS_PER_CONTROL_GROUP_CONFIG)
    parser.add_argument("--wls_bins", type=int, default=WLS_BINS_CONFIG)
    parser.add_argument("--min_rows_per_bin", type=int, default=MIN_ROWS_PER_BIN_CONFIG)
    parser.add_argument("--min_sigma", type=float, default=MIN_SIGMA_CONFIG)
    parser.add_argument("--plot_sample_size", type=int, default=PLOT_SAMPLE_SIZE_CONFIG)
    parser.add_argument("--min_points_for_group_plot", type=int, default=MIN_POINTS_FOR_GROUP_PLOT_CONFIG)
    parser.add_argument("--enable_plotly_html", action="store_true", default=ENABLE_PLOTLY_HTML_CONFIG)
    args = parser.parse_args()

    if (not args.root_dir) or ("请把这里改成" in str(args.root_dir)):
        raise ValueError("DATA_ROOT_DIR 未配置或路径无效。")
    if (not args.output_dir) or ("请把这里改成" in str(args.output_dir)):
        raise ValueError("OUTPUT_DIR 未配置或路径无效。")

    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    html_dir = out_dir / "html"
    for d in [tables_dir, fig_dir, html_dir]:
        d.mkdir(parents=True, exist_ok=True)

    quantiles = parse_list_float(args.threshold_quantiles)

    print("[STEP] 读取数据与预处理")
    data = load_dataset(args, tables_dir)

    all_metric_rows: List[pd.DataFrame] = []
    all_code_rows: List[pd.DataFrame] = []
    all_device_rows: List[pd.DataFrame] = []
    all_fit_rows: List[pd.DataFrame] = []

    repeat0_plot_payload = []

    for repeat in range(args.n_repeats):
        print(f"[STEP] Repeat {repeat+1}/{args.n_repeats}: 设备级划分")
        split = split_normal_devices(data, repeat, args)
        split_summary = data.assign(split=split).groupby(["source_code", "label_name", "split"], observed=True).size().reset_index(name="records")
        split_summary["repeat"] = repeat
        split_summary.to_csv(tables_dir / f"split_summary_repeat_{repeat:02d}.csv", index=False, encoding="utf-8-sig")

        for model_id, cfg in TARGET_MODELS_CONFIG.items():
            model_name = cfg["name"]
            print(f"[STEP] Repeat {repeat+1}/{args.n_repeats}, {model_id}: 拟合与打分")
            models, fit_df = fit_models_for_repeat(data, split, model_id, args)
            fit_df["repeat"] = repeat
            fit_df["model_name"] = model_name
            all_fit_rows.append(fit_df)
            if not models:
                print(f"[WARN] Repeat {repeat}, {model_id}: 所有控制压差组均拟合失败，跳过")
                continue
            score_df = score_model(data, model_id, models, args)
            thresholds = thresholds_by_control(data, split, score_df["score"], quantiles)
            metric_df, code_df, device_df = build_metrics_and_device_summary(
                data, split, score_df, model_id, model_name, repeat, thresholds, quantiles
            )
            all_metric_rows.append(metric_df)
            all_code_rows.append(code_df)
            all_device_rows.append(device_df)

            # 保存 repeat=0 诊断图需要的内容；只保存引用，避免重复占内存。
            if repeat == 0:
                repeat0_plot_payload.append((model_id, model_name, models, score_df, thresholds, split.copy()))

    print("[STEP] 汇总并写出表格")
    metrics_all = pd.concat(all_metric_rows, ignore_index=True) if all_metric_rows else pd.DataFrame()
    code_all = pd.concat(all_code_rows, ignore_index=True) if all_code_rows else pd.DataFrame()
    device_all = pd.concat(all_device_rows, ignore_index=True) if all_device_rows else pd.DataFrame()
    fit_all = pd.concat(all_fit_rows, ignore_index=True) if all_fit_rows else pd.DataFrame()

    metrics_all.to_csv(tables_dir / "wls_repeat_metrics_by_threshold_control.csv", index=False, encoding="utf-8-sig")
    code_all.to_csv(tables_dir / "wls_repeat_metrics_by_code_threshold_control.csv", index=False, encoding="utf-8-sig")
    device_all.to_csv(tables_dir / "wls_repeat_device_summary_by_threshold_control.csv", index=False, encoding="utf-8-sig")
    if not device_all.empty:
        device_all[device_all["source_code"].astype(str).eq("N")].to_csv(tables_dir / "wls_repeat_normal_device_false_alarm_detail.csv", index=False, encoding="utf-8-sig")
    fit_all.to_csv(tables_dir / "wls_fit_models_by_repeat_model_control.csv", index=False, encoding="utf-8-sig")

    avg_metrics = summarize_repeats(metrics_all) if not metrics_all.empty else pd.DataFrame()
    avg_metrics.to_csv(tables_dir / "wls_average_metrics_over_repeats.csv", index=False, encoding="utf-8-sig")

    # 便于快速查看：筛选 heldout + all，并按是否达标排序。
    if not avg_metrics.empty:
        quick = avg_metrics[(avg_metrics["eval_scope"] == "heldout") & (avg_metrics["control_group"].astype(str) == "all")].copy()
        if not quick.empty:
            quick = quick.sort_values(["delivery_pass_count", "meets_fpr_lt_0_05_recall_gt_0_50_mean", "fpr_mean", "recall_mean"], ascending=[False, False, True, False])
            quick.to_csv(tables_dir / "wls_quick_select_heldout_all_average.csv", index=False, encoding="utf-8-sig")

    print("[STEP] 生成可视化")
    if not avg_metrics.empty:
        plot_metric_summary(avg_metrics, fig_dir)
    for model_id, model_name, models, score_df, thresholds, split0 in repeat0_plot_payload:
        try:
            plot_diagnostic_for_repeat0(data, split0, model_id, model_name, models, score_df, thresholds, args, fig_dir, html_dir)
        except Exception as e:
            print(f"[WARN] {model_id} 诊断图生成失败：{e}")
    if not device_all.empty:
        try:
            plot_device_top(device_all, fig_dir, html_dir, args.main_plot_quantile, args.enable_plotly_html)
        except Exception as e:
            print(f"[WARN] 正常设备误报 Top 图生成失败：{e}")

    config = {
        "model": "WLS multi-repeat multi-target",
        "root_dir": str(Path(args.root_dir)),
        "output_dir": str(out_dir),
        "f01_f02_mode": args.f01_f02_mode,
        "n_repeats": args.n_repeats,
        "train_normal_device_frac": args.train_normal_device_frac,
        "valid_control_pressures": parse_list_float(args.valid_control_pressures, args.control_round_digits),
        "threshold_quantiles": quantiles,
        "wls_min_speed": args.wls_min_speed,
        "wls_max_speed": args.wls_max_speed,
        "data_max_rows_per_file": args.data_max_rows_per_file,
        "train_max_rows_per_device_per_control": args.train_max_rows_per_device_per_control,
        "train_max_rows_per_control_group": args.train_max_rows_per_control_group,
        "enable_plotly_html": args.enable_plotly_html,
        "target_models": TARGET_MODELS_CONFIG,
        "sampling_note": "训练抽样：每次重复先按设备划分70%正常设备为normal_train；随后在每个训练设备、每个控制压差组中，对有效记录均匀随机无放回抽取不超过train_max_rows_per_device_per_control条用于WLS拟合。评价指标基于读入数据计算；若data_max_rows_per_file为None则为全量读入，否则为每个CSV均匀随机抽样后的评价。",
        "speed_filter_note": "WLS仅保留 wls_min_speed <= speed < wls_max_speed 的记录；若转速已归一化为0-1，则阈值自动除以100。",
        "outputs": {
            "repeat_metrics": "tables/wls_repeat_metrics_by_threshold_control.csv",
            "average_metrics": "tables/wls_average_metrics_over_repeats.csv",
            "quick_select": "tables/wls_quick_select_heldout_all_average.csv",
            "device_summary": "tables/wls_repeat_device_summary_by_threshold_control.csv",
        }
    }
    with open(tables_dir / "wls_multi_repeat_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("[DONE] 输出完成")
    print(f"  表格目录：{tables_dir}")
    print(f"  图片目录：{fig_dir}")
    print(f"  HTML目录：{html_dir}")
    print("  关键表：")
    print("    wls_repeat_metrics_by_threshold_control.csv  # 每次重复/每个模型/每个阈值/每个控制压差组的指标")
    print("    wls_average_metrics_over_repeats.csv          # 20次重复均值总表")
    print("    wls_quick_select_heldout_all_average.csv      # 快速筛选总体达标阈值")


if __name__ == "__main__":
    main()
