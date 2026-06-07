"""WLS 拟合模块：加权最小二乘回归（报告 §4）。

关键修复：
1. 局部 sigma 赋值避免链式索引；
2. 默认不把百万级数组转换为 Python list，避免内存暴涨；
3. 提供 score_wls()，保证训练阈值和测试评分使用同一套标准化残差口径。
"""

from typing import Dict, Optional

import numpy as np


def _wls_solve(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """求解 WLS/OLS 正规方程。"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if w is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).reshape(-1)

    mask = (
        np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
        & np.all(np.isfinite(X), axis=1)
    )
    X = X[mask]
    y = y[mask]
    w = w[mask]

    if len(y) == 0:
        raise ValueError("有效样本数为 0，无法拟合 WLS/OLS 模型")

    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtWX) @ XtWy
    return beta


def _robust_sigma(residuals: np.ndarray, min_sigma: float = 1e-6) -> float:
    """MAD-based 稳健尺度估计。"""
    residuals = np.asarray(residuals, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return float(min_sigma)
    med = np.nanmedian(residuals)
    mad = np.nanmedian(np.abs(residuals - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < min_sigma:
        sigma = np.nanstd(residuals)
    if not np.isfinite(sigma) or sigma < min_sigma:
        sigma = min_sigma
    return float(sigma)


def _make_bin_edges(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """等频分箱边界。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(np.float64)


def _assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """按训练阶段保存的边界分配箱号。"""
    edges = np.asarray(edges, dtype=np.float64)
    n_bins = max(1, len(edges) - 1)
    bins = np.searchsorted(edges[1:-1], x, side="right")
    return np.clip(bins, 0, n_bins - 1)


def score_wls(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    sigma_per_bin: np.ndarray,
    bin_edges: np.ndarray,
    min_sigma: float = 1e-6,
) -> np.ndarray:
    """使用训练好的 WLS 参数计算标准化残差 score = |y-y_hat|/sigma_local。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    sigma_per_bin = np.asarray(sigma_per_bin, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)

    scores = np.full(len(x), np.nan, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return scores

    x_v = x[valid]
    y_v = y[valid]
    y_hat = beta[0] + beta[1] * x_v
    resid = y_v - y_hat

    bins = _assign_bins(x_v, bin_edges)
    bins = np.clip(bins, 0, len(sigma_per_bin) - 1)
    local_sigma = sigma_per_bin[bins]
    scores[valid] = np.abs(resid) / np.maximum(local_sigma, min_sigma)
    return scores


def fit_wls(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    min_sigma: float = 1e-6,
    return_full: bool = True,
) -> Dict:
    """执行完整 WLS 五步拟合。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x_v = x[valid]
    y_v = y[valid]
    if len(x_v) < 2 or len(np.unique(x_v)) < 2:
        raise ValueError(f"有效样本不足或 x 唯一值不足：n={len(x_v)}, unique_x={len(np.unique(x_v))}")

    X_v = np.column_stack([np.ones(len(x_v), dtype=np.float64), x_v])

    # Step 1: OLS 初拟合
    beta_ols = _wls_solve(X_v, y_v)
    resid_ols = y_v - X_v @ beta_ols

    # Step 2: 分箱估计初始 sigma
    bin_edges = _make_bin_edges(x_v, n_bins)
    n_actual_bins = len(bin_edges) - 1
    bins = _assign_bins(x_v, bin_edges)

    global_sigma0 = _robust_sigma(resid_ols, min_sigma)
    sigma_per_bin = np.empty(n_actual_bins, dtype=np.float64)
    for b in range(n_actual_bins):
        mask = bins == b
        sigma_per_bin[b] = _robust_sigma(resid_ols[mask], min_sigma) if mask.sum() >= 5 else global_sigma0
    sigma_per_bin = np.maximum(sigma_per_bin, min_sigma)

    # Step 3: 权重
    w = 1.0 / sigma_per_bin[bins] ** 2

    # Step 4: WLS 拟合
    beta = _wls_solve(X_v, y_v, w)

    # Step 5: 用 WLS 残差重新估计 sigma
    resid_wls = y_v - X_v @ beta
    global_sigma = _robust_sigma(resid_wls, min_sigma)
    sigma_final = np.empty(n_actual_bins, dtype=np.float64)
    for b in range(n_actual_bins):
        mask = bins == b
        sigma_final[b] = _robust_sigma(resid_wls[mask], min_sigma) if mask.sum() >= 5 else global_sigma
    sigma_final = np.maximum(sigma_final, min_sigma)

    result = {
        "beta": beta.tolist(),
        "sigma_per_bin": sigma_final.tolist(),
        "bin_edges": bin_edges.tolist(),
        "n_valid": int(len(x_v)),
    }

    if return_full:
        # 注意：这里返回 numpy 数组，而不是 Python list，避免百万级数据内存暴涨。
        pred_all = np.full(len(x), np.nan, dtype=np.float64)
        pred_all[valid] = X_v @ beta
        resid_all = np.full(len(x), np.nan, dtype=np.float64)
        resid_all[valid] = resid_wls
        scores = score_wls(x, y, beta, sigma_final, bin_edges, min_sigma=min_sigma)
        result.update({
            "bin_assignments": bins,
            "predictions": pred_all,
            "residuals": resid_all,
            "scores": scores,
        })

    return result
