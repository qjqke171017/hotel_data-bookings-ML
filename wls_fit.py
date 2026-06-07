"""WLS 拟合模块：加权最小二乘回归。

核心约定：
    score = |y - y_hat| / sigma_local

本模块只保存 WLS 预测所需的轻量参数（beta、bin_edges、sigma_per_bin），
避免把百万级预测数组写入 JSON 或长期占用内存。
"""

from typing import Dict, Optional

import numpy as np


def _wls_solve(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """求解 OLS/WLS 正规方程。"""
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

    # 不构造 W = diag(w)，避免 N×N 大矩阵。
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtWX) @ XtWy
    return beta


def _robust_sigma(residuals: np.ndarray, min_sigma: float = 1e-6) -> float:
    """MAD 稳健尺度估计；退化时回退到标准差。"""
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
    """按 x 的分位数构造等频分箱边界。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)

    n_bins = max(1, int(n_bins))
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.array([-np.inf, np.inf], dtype=np.float64)

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(np.float64)


def _assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """使用训练阶段保存的边界为样本分箱。"""
    x = np.asarray(x, dtype=np.float64)
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
    """按训练好的 WLS 参数计算标准化残差分数。"""
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
    residual = y_v - y_hat

    bins = _assign_bins(x_v, bin_edges)
    bins = np.clip(bins, 0, len(sigma_per_bin) - 1)
    sigma = sigma_per_bin[bins]
    scores[valid] = np.abs(residual) / np.maximum(sigma, min_sigma)
    return scores


def fit_wls(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    min_sigma: float = 1e-6,
    min_rows_per_bin: int = 30,
    return_full: bool = False,
) -> Dict:
    """执行 WLS 五步拟合，并返回轻量模型参数。

    参数:
        x: 自变量，即 (speed/100)^2。
        y: 因变量。
        n_bins: 分箱数量。
        min_sigma: 局部 sigma 下界。
        min_rows_per_bin: 单个分箱估计局部 sigma 的最小样本数。
        return_full: 兼容参数；若为 True，额外返回训练数组的 score/residual。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x_v = x[valid]
    y_v = y[valid]
    if len(x_v) < 2 or len(np.unique(x_v)) < 2:
        raise ValueError(f"有效样本不足或 x 唯一值不足：n={len(x_v)}, unique_x={len(np.unique(x_v))}")

    X_v = np.column_stack([np.ones(len(x_v), dtype=np.float64), x_v])

    # Step 1: OLS 初拟合。
    beta_ols = _wls_solve(X_v, y_v)
    resid_ols = y_v - X_v @ beta_ols

    # Step 2: 分箱估计初始局部 sigma。
    bin_edges = _make_bin_edges(x_v, n_bins)
    n_actual_bins = len(bin_edges) - 1
    bins = _assign_bins(x_v, bin_edges)

    global_sigma0 = _robust_sigma(resid_ols, min_sigma)
    sigma_init = np.empty(n_actual_bins, dtype=np.float64)
    for b in range(n_actual_bins):
        m = bins == b
        sigma_init[b] = _robust_sigma(resid_ols[m], min_sigma) if int(m.sum()) >= min_rows_per_bin else global_sigma0
    sigma_init = np.maximum(sigma_init, min_sigma)

    # Step 3/4: WLS 拟合。
    w = 1.0 / (sigma_init[bins] ** 2)
    beta = _wls_solve(X_v, y_v, w)

    # Step 5: 用 WLS 残差重新估计最终局部 sigma。
    resid_wls = y_v - X_v @ beta
    global_sigma = _robust_sigma(resid_wls, min_sigma)
    sigma_final = np.empty(n_actual_bins, dtype=np.float64)
    for b in range(n_actual_bins):
        m = bins == b
        sigma_final[b] = _robust_sigma(resid_wls[m], min_sigma) if int(m.sum()) >= min_rows_per_bin else global_sigma
    sigma_final = np.maximum(sigma_final, min_sigma)

    result = {
        "beta": beta.tolist(),
        "sigma_per_bin": sigma_final.tolist(),
        "bin_edges": bin_edges.tolist(),
        "n_valid": int(len(x_v)),
        "global_sigma": float(global_sigma),
        "x_min": float(np.nanmin(x_v)),
        "x_max": float(np.nanmax(x_v)),
    }

    if return_full:
        pred_all = np.full(len(x), np.nan, dtype=np.float64)
        pred_all[valid] = X_v @ beta
        resid_all = np.full(len(x), np.nan, dtype=np.float64)
        resid_all[valid] = resid_wls
        scores = score_wls(x, y, beta, sigma_final, bin_edges, min_sigma=min_sigma)
        result.update({
            "predictions": pred_all,
            "residuals": resid_all,
            "scores": scores,
        })

    return result
