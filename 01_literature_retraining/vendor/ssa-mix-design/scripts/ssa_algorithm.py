# -*- coding: utf-8 -*-
r"""
SSA (Sparrow Search Algorithm) — 纯Python/numpy实现
参考: SSA_SSA 函数的历史 MATLAB 实现
"""
import numpy as np


def ssa_optimize(fitness_func, lb, ub, pop=30, max_iter=50,
                 p_percent=0.2, safety_threshold=0.8, sentinel_ratio=0.2,
                 verbose=True, model_name="Model"):
    """
    麻雀搜索算法 (Sparrow Search Algorithm) 超参数优化

    Parameters
    ----------
    fitness_func : callable
        适应度函数, 签名为 f(params_1d_array) -> float (MAE越小越好)
    lb : list or np.array
        每个超参数的下界
    ub : list or np.array
        每个超参数的上界
    pop : int
        种群大小 (麻雀数量)
    max_iter : int
        最大迭代次数
    p_percent : float
        生产者(发现者)占比
    safety_threshold : float
        安全阈值 ST
    sentinel_ratio : float
        警戒者占比
    verbose : bool
        是否打印过程
    model_name : str
        模型名称, 用于打印

    Returns
    -------
    best_params : np.array
        最优超参数
    best_fitness : float
        最优适应度值 (最小MAE)
    convergence_curve : np.array
        收敛曲线
    """
    dim = len(lb)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    # 生产者数量
    p_num = max(1, int(pop * p_percent))

    # 初始化种群
    x = lb + (ub - lb) * np.random.rand(pop, dim)

    # 计算初始适应度
    fit = np.full(pop, np.inf)
    models = [None] * pop
    for i in range(pop):
        fit[i] = fitness_func(x[i])

    # 个体最佳
    p_fit = fit.copy()
    p_x = x.copy()

    # 全局最佳
    best_idx = np.argmin(fit)
    best_fitness = fit[best_idx]
    best_x = x[best_idx].copy()

    convergence_curve = np.zeros(max_iter)

    for t in range(max_iter):
        # 按适应度排序
        sort_idx = np.argsort(p_fit)

        # 最差个体
        worst_idx = np.argmax(p_fit)
        worst_x = x[worst_idx].copy()

        # 当前全局最优
        current_best_idx = np.argmin(fit)
        current_best_x = x[current_best_idx].copy()

        r2 = np.random.rand()  # 预警值

        # ===== 1. 生产者(发现者)位置更新 =====
        if r2 < safety_threshold:
            # 安全: 广泛搜索
            for i in range(p_num):
                r1 = np.random.rand()
                idx = sort_idx[i]
                x[idx] = p_x[idx] * np.exp(-(i + 1) / (r1 * max_iter))
                x[idx] = np.clip(x[idx], lb, ub)
                fit[idx] = fitness_func(x[idx])
        else:
            # 危险: 飞往安全区域
            for i in range(p_num):
                idx = sort_idx[i]
                x[idx] = p_x[idx] + np.random.randn() * np.ones(dim)
                x[idx] = np.clip(x[idx], lb, ub)
                fit[idx] = fitness_func(x[idx])

        # ===== 2. 跟随者(加入者)位置更新 =====
        for i in range(p_num, pop):
            idx = sort_idx[i]
            A = np.floor(np.random.rand(dim) * 2) * 2 - 1  # 1或-1

            if i > pop / 2:
                # 适应度差的跟随者: 飞往其他地方觅食
                x[idx] = np.random.randn() * np.exp((worst_x - p_x[idx]) / (i ** 2))
            else:
                # 适应度好的跟随者: 在最优附近觅食
                try:
                    A_inv = A.T @ np.linalg.inv(A.reshape(-1, 1) @ A.reshape(1, -1))
                    x[idx] = current_best_x + np.abs(p_x[idx] - current_best_x) * A_inv
                except np.linalg.LinAlgError:
                    x[idx] = current_best_x + np.random.randn(dim) * np.abs(p_x[idx] - current_best_x)

            x[idx] = np.clip(x[idx], lb, ub)
            fit[idx] = fitness_func(x[idx])

        # ===== 3. 警戒者位置更新 =====
        c = np.random.permutation(pop)
        n_sentinels = max(1, int(sentinel_ratio * pop))
        sentinel_indices = sort_idx[c[:n_sentinels]]

        for idx in sentinel_indices:
            if p_fit[idx] > best_fitness:
                # 处于群体边缘, 向最优移动
                x[idx] = best_x + np.random.randn(dim) * np.abs(p_x[idx] - best_x)
            else:
                # 处于群体中心, 向周围移动
                fmax = np.max(p_fit)
                x[idx] = p_x[idx] + (2 * np.random.rand() - 1) * np.abs(p_x[idx] - worst_x) / (p_fit[idx] - fmax + 1e-50)

            x[idx] = np.clip(x[idx], lb, ub)
            fit[idx] = fitness_func(x[idx])

        # ===== 更新个体和全局最优 =====
        for i in range(pop):
            if fit[i] < p_fit[i]:
                p_fit[i] = fit[i]
                p_x[i] = x[i].copy()

            if p_fit[i] < best_fitness:
                best_fitness = p_fit[i]
                best_x = p_x[i].copy()

        convergence_curve[t] = best_fitness

        if verbose and (t + 1) % 10 == 0:
            print(f"  [{model_name}] SSA iter {t+1:3d}/{max_iter}, best MAE={best_fitness:.4f}")

    return best_x, best_fitness, convergence_curve


def Bounds(x, lb, ub):
    """边界约束"""
    return np.clip(x, lb, ub)
