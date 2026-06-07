"""WLS 拟合模块：二次侧机理模型的加权最小二乘核心。

本文件尽量保持旧版 ``wls_secondary_multi_repeat_speed50_95.py`` 的统计逻辑：
先用 OLS 初拟合残差估计不同转速平方区间的局部尺度，再以
``1 / sigma^2`` 为权重重新拟合 WLS，最终用 WLS 残差重新估计局部
``sigma``，异常分数定义为 ``abs(residual) / local_sigma``。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _robust_sigma(x: np.ndarray, min_sigma: float = 1e-6) -> float:
    """稳健尺度估计：优先使用 MAD，退化时使用标准差，最后使用下界。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(min_sigma)

    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig < min_sigma:
        sig = np.std(x)
    if not np.isfinite(sig) or sig < min_sigma:
        sig = min_sigma
    return float(sig)


def _wls_solve(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """求解 OLS/WLS 正规方程，使用 pinv 兜底以适应工业数据退化情况。"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if w is None:
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        Xv = X[mask]
        yv = y[mask]
        if len(yv) == 0:
            raise ValueError("有效样本数为 0，无法拟合 OLS/WLS 模型")
        return np.linalg.pinv(Xv.T @ Xv) @ Xv.T @ yv

    w = np.asarray(w, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.all(np.isfinite(X), axis=1)
    Xv = X[mask]
    yv = y[mask]
    wv = w[mask]
    if len(yv) == 0:
        raise ValueError("有效样本数为 0，无法拟合 OLS/WLS 模型")

    sw = np.sqrt(wv)
    Xw = Xv * sw[:, None]
    yw = yv * sw
    return np.linalg.pinv(Xw.T @ Xw) @ Xw.T @ yw


def _make_bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """按照旧代码逻辑用 x 分位数构造分箱边界。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    n_bins = max(1, int(n_bins))
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """根据边界分配箱号，箱号范围为 [0, len(edges)-2]。"""
    x = np.asarray(x, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if len(edges) <= 2:
        return np.zeros(len(x), dtype=int)
    return np.searchsorted(edges[1:-1], x, side="right")


def fit_wls(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    min_sigma: float = 1e-6,
    min_rows_per_bin: int = 30,
    min_train_rows: int = 80,
) -> Dict:
    """执行旧代码式 WLS 拟合。

    参数:
        x: 自变量，通常为 ``(二次侧泵转速/100)^2``。
        y: 因变量，三种二次侧压差组合之一。
        n_bins: 转速平方分箱数。
        min_sigma: 局部尺度下界。
        min_rows_per_bin: 每个分箱内估计局部尺度所需的最小记录数。
        min_train_rows: 当前控制压差组最小训练记录数。

    返回:
        包含 beta、bin_edges、sigma_by_bin、global_sigma、scores 等字段的字典。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x_v = x[valid]
    y_v = y[valid]

    if len(x_v) < int(min_train_rows) or len(np.unique(x_v)) < 2:
        raise ValueError(
            f"训练样本不足或转速平方唯一值不足：n={len(x_v)}, unique_x={len(np.unique(x_v))}"
        )

    X_v = np.column_stack([np.ones_like(x_v), x_v])

    # Step 1: OLS 初拟合
    beta_ols = _wls_solve(X_v, y_v)
    resid0 = y_v - X_v @ beta_ols
    global_sigma0 = _robust_sigma(resid0, min_sigma)

    # Step 2: 按 x 分箱估计 OLS 残差局部尺度
    edges = _make_bin_edges(x_v, n_bins)
    bins = _assign_bins(x_v, edges)
    n_actual_bins = len(edges) - 1

    sigma0 = []
    for b in range(n_actual_bins):
        rb = resid0[bins == b]
        sigma0.append(_robust_sigma(rb, min_sigma) if len(rb) >= int(min_rows_per_bin) else global_sigma0)
    sigma0 = np.maximum(np.asarray(sigma0, dtype=float), float(min_sigma))

    # Step 3/4: 构造权重并 WLS 拟合
    w = 1.0 / sigma0[np.clip(bins, 0, len(sigma0) - 1)] ** 2
    beta = _wls_solve(X_v, y_v, w)

    # Step 5: 用 WLS 残差重新估计局部 sigma
    resid = y_v - X_v @ beta
    global_sigma = _robust_sigma(resid, min_sigma)
    sigma_final = []
    for b in range(n_actual_bins):
        rb = resid[bins == b]
        sigma_final.append(_robust_sigma(rb, min_sigma) if len(rb) >= int(min_rows_per_bin) else global_sigma)
    sigma_final = np.maximum(np.asarray(sigma_final, dtype=float), float(min_sigma))

    # 对输入 x/y 生成训练输入上的预测、残差、sigma 和 score，便于诊断写入 JSON。
    pred_all = np.full(len(x), np.nan, dtype=float)
    resid_all = np.full(len(x), np.nan, dtype=float)
    sigma_all = np.full(len(x), np.nan, dtype=float)
    scores_all = np.full(len(x), np.nan, dtype=float)

    valid_positions = np.flatnonzero(valid)
    pred_all[valid_positions] = X_v @ beta
    resid_all[valid_positions] = resid
    sigma_v = sigma_final[np.clip(bins, 0, len(sigma_final) - 1)]
    sigma_all[valid_positions] = sigma_v
    scores_all[valid_positions] = np.abs(resid) / np.maximum(sigma_v, float(min_sigma))

    return {
        "beta": beta.tolist(),
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "n_train": int(len(x_v)),
        "bin_edges": edges.tolist(),
        "sigma_by_bin": sigma_final.tolist(),
        "sigma_per_bin": sigma_final.tolist(),  # 兼容旧项目字段名
        "global_sigma": float(global_sigma),
        "x_min": float(np.nanmin(x_v)),
        "x_max": float(np.nanmax(x_v)),
        "bin_assignments": bins.tolist(),
        "predictions": pred_all.tolist(),
        "residuals": resid_all.tolist(),
        "sigma": sigma_all.tolist(),
        "scores": scores_all.tolist(),
    }


def predict_wls(x: np.ndarray, model: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """用 ``fit_wls`` 保存的模型参数预测 yhat 和局部 sigma。"""
    x = np.asarray(x, dtype=float).reshape(-1)
    beta = np.asarray(model.get("beta", [model.get("beta0"), model.get("beta1")]), dtype=float)
    yhat = beta[0] + beta[1] * x

    edges = np.asarray(model.get("bin_edges", [-np.inf, np.inf]), dtype=float)
    sigmas = np.asarray(model.get("sigma_by_bin", model.get("sigma_per_bin", [model.get("global_sigma", 1.0)])), dtype=float)
    if len(sigmas) == 0:
        sigmas = np.asarray([float(model.get("global_sigma", 1.0))], dtype=float)
    bins = _assign_bins(x, edges)
    bins = np.clip(bins, 0, len(sigmas) - 1)
    return yhat, sigmas[bins]
