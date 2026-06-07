"""WLS 拟合模块：加权最小二乘回归（报告 §4）。

数学原理
--------
工业泵系统的压差与转速平方之间在理想条件下呈线性关系:
    ΔP = β₀ + β₁ · (speed/100)² + ε

但残差方差在低转速区间通常较大（测量噪声有限），高转速区间较小，
违背 OLS 同方差假设。WLS 通过分箱估计局部方差 σ²_b，以 1/σ² 为权重
给予低噪声区间更大的影响。

五步流程（报告 §4）:
    Step 1: OLS 初拟合 y = β₀ + β₁x → 获取初残差
    Step 2: 按 x 等频分箱，每箱用 MAD 估计局部 σ
    Step 3: 构造权重 w_i = 1/σ²_{bin(i)}
    Step 4: WLS 拟合 β = (XᵀWX)⁻¹XᵀWy
    Step 5: 基于 WLS 残差重新估计 σ → 标准化残差评分

公式:
    MAD = 1.4826 × median(|r_i - median(r)|)    稳健尺度估计（正态分布一致性常数）
    w_i = 1 / σ²_{bin(i)}                        异方差权重
    score_i = |r_i| / σ_{bin(i)}                 标准化残差（用作异常分数）

数值稳定性:
    - 1.4826 = 1 / Φ⁻¹(0.75)，使 MAD 对正态分布数据与 σ 一致
    - min_sigma = 1e-6: 局部 σ 下界，防止除零
    - 每箱最小样本数 5: 避免小样本导致 MAD 估计不稳定
    - edges[0] -= 1e-10 / edges[-1] += 1e-10: 确保 digitize 包含边界值
"""

from typing import Dict, Tuple

import numpy as np


def _wls_solve(X: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> np.ndarray:
    """求解 WLS 正规方程: β = (XᵀWX)⁻¹ XᵀWy（报告 §4 Step 4）。

    使用 np.linalg.solve（基于 LU 分解），比显式求逆更数值稳定。

    当 w=None 时退化为 OLS: β = (XᵀX)⁻¹ Xᵀy。

    参数:
        X (ndarray): (N × 2) 设计矩阵 [1, x]，第一列为截距项。
        y (ndarray): (N,) 因变量向量。
        w (ndarray or None): (N,) 权重向量，若为 None 则使用等权（退化为 OLS）。

    返回:
        ndarray: (2,) 回归系数 [β₀, β₁]。
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # 如果没有传入权重，就退化为普通最小二乘
    if w is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
    # 过滤非法值
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

    # 不要构遣W=np.diag(w)
    # X.T @ W @ X价于 X.T @ (w[:, None] * X)
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtWX) @ XtWy
    return beta


def _robust_sigma(residuals: np.ndarray) -> float:
    """MAD-based 稳健尺度估计（报告 §4 Step 2 和 Step 5）。

    公式: σ_MAD = 1.4826 × median(|r - median(r)|)

    常数 1.4826 = 1/Φ⁻¹(0.75) 是 MAD 到正态分布 σ 的一致性校正因子。
    相比传统标准差，MAD 对离群点的崩溃点达 50%（标准差崩溃点为 0%）。

    数值稳定性:
        - 使用 nanmedian 处理 NaN 残差
        - max(mad, 1e-8): 防止所有残差相同时 mad=0 导致后续除零

    参数:
        residuals (ndarray): 残差数组。

    返回:
        float: 稳健标准差估计。(3,) 回归系数 [β0, β1]。
    """
    med = np.nanmedian(residuals)
    mad = np.nanmedian(np.abs(residuals - med))
    # 1.4826: 正态分布一致性常数
    return 1.4826 * max(mad, 1e-8)


def _make_bin_edges(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """等频分箱边界。

    返回 n_bins+1 个边界；后续通过 _assign_bins 分配箱号。单独保留边界，
    是为了在训练后对同一控制压差组的测试样本使用同一套局部 sigma。
    """
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-10
    edges[-1] += 1e-10
    return edges


def _assign_bins(x: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """按训练阶段保存的边界分配箱号，并裁剪到 [0, n_bins-1]。"""
    bins = np.digitize(x, edges) - 1
    return np.clip(bins, 0, n_bins - 1)


def fit_wls(
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
        min_sigma: float = 1e-6,
) -> Dict:
    """执行完整 WLS 五步拟合（报告 §4 完整算法流程）。

    参数:
        x (ndarray): (N,) 自变量，归一化泵转速平方 = (speed/100)²。
        y (ndarray): (N,) 因变量，压差指标（M1=泵压差, M2=板换+供回水压差, M3=板换+过滤器压差）。
        n_bins (int): 分箱数量（默认 10，报告 §4）。
        min_sigma (float): 最小局部 σ（默认 1e-6），防除零。

    返回:
        dict: 结果字典，包含以下键:
            - beta (list): [β₀, β₁] WLS 回归系数
            - sigma_per_bin (list): 每箱的最终局部 σ（Step 5 结果）
            - bin_assignments (list): 每个有效样本的箱号
            - predictions (list): 全量预测值（无效样本位置为 NaN）
            - residuals (list): 全量 WLS 残差（无效样本位置为 NaN）
            - scores (list): 全量标准化残差（|残差|/σ_local），用作异常分数
    """
    # 构造设计矩阵 X = [1, x]（报告 §4 线性模型 y = β₀ + β₁x）
    X = np.column_stack([np.ones(len(x)), x])
    valid = np.isfinite(x) & np.isfinite(y)
    x_v = x[valid]
    y_v = y[valid]
    X_v = X[valid]

    # ================================================================
    # Step 1: OLS 初拟合（报告 §4 Step 1）
    # y = β₀ + β₁x + ε，等权回归获取初残差
    # ================================================================
    beta_ols = _wls_solve(X_v, y_v)
    resid_ols = y_v - X_v @ beta_ols

    # ================================================================
    # Step 2: 按 x 等频分箱估计局部 σ（报告 §4 Step 2）
    # 用 MAD（而非传统标准差）稳健估计每箱的残差离散度
    # ================================================================
    bin_edges = _make_bin_edges(x_v, n_bins)
    bins = _assign_bins(x_v, bin_edges, n_bins)
    sigma_per_bin = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() >= 5:  # 最小样本数 5: 保证 MAD 估计可靠性
            sigma_per_bin[b] = _robust_sigma(resid_ols[mask])
        else:
            # 小箱: 使用前一个箱的 σ 或默认值 1.0
            sigma_per_bin[b] = max(sigma_per_bin.max(), 1.0) if b > 0 else 1.0

    # 下界裁剪: 防止 σ 过小导致权重过大
    sigma_per_bin = np.maximum(sigma_per_bin, min_sigma)

    # ================================================================
    # Step 3: 构造异方差权重（报告 §4 Step 3）
    # w_i = 1/σ²_{bin(i)}，低噪声区间（小σ）获得更大权重
    # ================================================================
    w = np.zeros(len(x_v))
    for b in range(n_bins):
        w[bins == b] = 1.0 / sigma_per_bin[b] ** 2

    # ================================================================
    # Step 4: WLS 拟合（报告 §4 Step 4）
    # β = (XᵀWX)⁻¹XᵀWy
    # ================================================================
    beta = _wls_solve(X_v, y_v, w)

    # ================================================================
    # Step 5: 基于 WLS 残差重新估计 σ（报告 §4 Step 5）
    # 使用更新后的残差重新计算局部 σ，提高估计精度
    # ================================================================
    resid_wls = y_v - X_v @ beta
    sigma_final = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() >= 5:
            sigma_final[b] = _robust_sigma(resid_wls[mask])
        else:
            # 小箱直接复用 Step 2 的 σ
            sigma_final[b] = sigma_per_bin[b]

    sigma_final = np.maximum(sigma_final, min_sigma)

    # 全量预测和标准化残差（包括无效样本位置填充 NaN）
    pred_all = np.full(len(x), np.nan)
    pred_all[valid] = X_v @ beta
    resid_all = np.full(len(x), np.nan)
    resid_all[valid] = resid_wls

    # 分配每个样本的局部 σ
    # 注意：不能写 sigma_all[valid][bins == b] = ...，那是链式索引，
    # 会写入临时副本，导致 sigma_all 仍为 0，score 被错误放大到 |resid|/1e-6。
    sigma_all = np.full(len(x), np.nan, dtype=float)
    valid_idx = np.flatnonzero(valid)
    for b in range(n_bins):
        sigma_all[valid_idx[bins == b]] = sigma_final[b]

    # 标准化残差 = |r| / σ_local（报告 §4 异常评分公式）
    scores = np.abs(resid_all) / np.maximum(sigma_all, min_sigma)

    return {
        "beta": beta.tolist(),
        "sigma_per_bin": sigma_final.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_assignments": bins.tolist(),
        "predictions": pred_all.tolist(),
        "residuals": resid_all.tolist(),
        "scores": scores.tolist(),
    }
