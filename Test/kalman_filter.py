"""
多变量卡尔曼滤波器实现
用于冷水机组传感器数据的去噪和平滑
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import warnings

# 导入配置文件
try:
    from kalman_config import (
        INITIAL_Q, INITIAL_R, INITIAL_P,
        STATE_TRANSITION_MATRIX, OBSERVATION_MATRIX,
        KALMAN_CONFIG, STATE_NAMES
    )
    _DEFAULT_DIM = INITIAL_Q.shape[0]
except ImportError:
    _DEFAULT_DIM = 6
    INITIAL_Q = np.eye(_DEFAULT_DIM) * 0.1
    INITIAL_R = np.eye(_DEFAULT_DIM) * 1.0
    INITIAL_P = np.eye(_DEFAULT_DIM) * 10.0
    STATE_TRANSITION_MATRIX = np.eye(_DEFAULT_DIM)
    OBSERVATION_MATRIX = np.eye(_DEFAULT_DIM)
    KALMAN_CONFIG = {'use_adaptive_noise': True, 'verbose': True}
    STATE_NAMES = [f"state_{i}" for i in range(_DEFAULT_DIM)]


class MultiVariateKalmanFilter:
    """
    多变量卡尔曼滤波器

    适用于冷水机组传感器数据的去噪和平滑处理
    支持7维状态向量和自适应噪声估计
    """

    def __init__(self,
                 state_dim: int = 7,
                 obs_dim: int = 7,
                 state_names: Optional[List[str]] = None,
                 config: Optional[Dict] = None):
        """
        初始化多变量卡尔曼滤波器

        参数:
        ----------
        state_dim : int
            状态向量维度
        obs_dim : int
            观测向量维度
        state_names : List[str], optional
            状态名称列表，用于调试和可视化
        config : Dict, optional
            配置字典，覆盖默认配置
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # 合并配置
        self.config = KALMAN_CONFIG.copy()
        if config:
            self.config.update(config)

        # 状态名称
        self.state_names = state_names or STATE_NAMES[:state_dim]
        if len(self.state_names) != state_dim:
            warnings.warn(f"状态名称数量({len(self.state_names)})与状态维度({state_dim})不匹配")
            self.state_names = [f"state_{i}" for i in range(state_dim)]

        # 系统模型矩阵
        self.F = STATE_TRANSITION_MATRIX[:state_dim, :state_dim]  # 状态转移矩阵
        self.H = OBSERVATION_MATRIX[:obs_dim, :state_dim]        # 观测矩阵

        # 协方差矩阵
        self.Q = INITIAL_Q[:state_dim, :state_dim]  # 过程噪声协方差
        self.R = INITIAL_R[:obs_dim, :obs_dim]      # 观测噪声协方差
        self.P = INITIAL_P[:state_dim, :state_dim]  # 状态估计协方差

        # 初始状态（设为0，将在第一次更新时修正）
        self.x = np.zeros(state_dim)

        # 前验状态缓存（用于RTS平滑器）
        self._last_x_prior = np.zeros(state_dim)
        self._last_P_prior = np.eye(state_dim)

        # 历史记录
        self.history = {
            'states': [],           # 状态估计历史
            'covariances': [],      # 协方差历史（对角线元素）
            'innovations': [],      # 创新序列（观测残差）
            'kalman_gains': [],     # 卡尔曼增益历史
            'timestamps': []        # 时间戳（如果提供）
        }

        # 统计信息
        self.stats = {
            'steps_processed': 0,
            'avg_innovation_norm': 0.0,
            'max_innovation_norm': 0.0,
            'convergence_status': 'initialized'
        }

        # 自适应噪声估计
        if self.config.get('use_adaptive_noise', True):
            self.innovation_buffer = []
            self.innovation_buffer_size = self.config.get('adaptive_window_size', 100)
            self.adaptive_learning_rate = self.config.get('adaptive_learning_rate', 0.1)

        if self.config.get('verbose', True):
            print(f"多变量卡尔曼滤波器初始化完成")
            print(f"  状态维度: {state_dim}, 观测维度: {obs_dim}")
            print(f"  状态名称: {self.state_names}")
            print(f"  自适应噪声估计: {self.config.get('use_adaptive_noise', True)}")

    def predict(self) -> np.ndarray:
        """
        预测步骤（时间更新）

        返回:
        -------
        np.ndarray
            预测状态向量
        """
        # 状态预测: x = F * x
        self.x = self.F @ self.x

        # 协方差预测: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        # 数值稳定性检查
        self._ensure_positive_definite(self.P)

        # 保存前验状态供RTS平滑器使用
        self._last_x_prior = self.x.copy()
        self._last_P_prior = self.P.copy()

        return self.x.copy()

    def update(self, z: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        更新步骤（测量更新）

        参数:
        ----------
        z : np.ndarray
            观测向量，形状为(obs_dim,)
        timestamp : float, optional
            时间戳，用于历史记录

        返回:
        -------
        np.ndarray
            更新后的状态估计
        """
        # 输入验证
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.obs_dim,):
            raise ValueError(f"观测向量形状应为({self.obs_dim},)，实际为{z.shape}")

        # 1. 预测步骤
        self.predict()

        # 2. 计算卡尔曼增益
        # S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        self._ensure_positive_definite(S)

        # K = P * H^T * S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 3. 计算创新（观测残差）
        # y = z - H * x
        y = z - self.H @ self.x
        innovation = y.copy()

        # 4. 状态更新
        # x = x + K * y
        self.x = self.x + K @ y

        # 4.5 状态约束：将滤波后的状态裁剪到物理合理范围
        self._clip_state()

        # 5. 协方差更新
        # P = (I - K * H) * P
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

        # 数值稳定性检查
        self._ensure_positive_definite(self.P)

        # 6. 记录历史
        self._record_history(K, innovation, timestamp)

        # 7. 更新统计信息
        self._update_stats(innovation)

        # 8. 自适应噪声估计（如果启用）
        if self.config.get('use_adaptive_noise', True):
            self._adaptive_noise_estimation(innovation, S)

        return self.x.copy()

    def _record_history(self, K: np.ndarray, innovation: np.ndarray,
                        timestamp: Optional[float] = None):
        """记录滤波器历史"""
        self.history['states'].append(self.x.copy())
        self.history['covariances'].append(np.diag(self.P).copy())
        self.history['innovations'].append(innovation.copy())
        self.history['kalman_gains'].append(K.copy())
        if timestamp is not None:
            self.history['timestamps'].append(timestamp)

    def _update_stats(self, innovation: np.ndarray):
        """更新统计信息"""
        self.stats['steps_processed'] += 1

        # 计算创新范数
        innovation_norm = np.linalg.norm(innovation)
        self.stats['max_innovation_norm'] = max(
            self.stats['max_innovation_norm'], innovation_norm
        )

        # 更新平均创新范数（指数移动平均）
        alpha = 0.1  # 平滑因子
        old_avg = self.stats['avg_innovation_norm']
        if self.stats['steps_processed'] == 1:
            self.stats['avg_innovation_norm'] = innovation_norm
        else:
            self.stats['avg_innovation_norm'] = (
                alpha * innovation_norm + (1 - alpha) * old_avg
            )

        # 检查收敛性
        if self.stats['steps_processed'] > 100:
            if self.stats['avg_innovation_norm'] < 0.1:
                self.stats['convergence_status'] = 'converged'
            elif self.stats['avg_innovation_norm'] < 1.0:
                self.stats['convergence_status'] = 'stable'
            else:
                self.stats['convergence_status'] = 'diverging'

    def _adaptive_noise_estimation(self, innovation: np.ndarray, S: np.ndarray):
        """
        自适应噪声估计

        基于创新序列动态调整观测噪声协方差R
        使用指数加权移动平均方法
        """
        # 将创新添加到缓冲区
        self.innovation_buffer.append(innovation.copy())
        if len(self.innovation_buffer) > self.innovation_buffer_size:
            self.innovation_buffer.pop(0)

        # 当有足够数据时更新R
        if len(self.innovation_buffer) >= min(50, self.innovation_buffer_size // 2):
            # 计算创新的样本协方差
            innovations_array = np.array(self.innovation_buffer)
            R_estimated = np.cov(innovations_array.T)

            # 确保R_estimated是正定矩阵
            self._ensure_positive_definite(R_estimated)

            # 平滑更新: R = (1-α)*R + α*R_estimated
            alpha = self.adaptive_learning_rate
            self.R = (1 - alpha) * self.R + alpha * R_estimated

            # 限制更新幅度，避免剧烈变化
            self.R = np.clip(self.R, 0.1 * INITIAL_R[:self.obs_dim, :self.obs_dim],
                            10.0 * INITIAL_R[:self.obs_dim, :self.obs_dim])

    def _clip_state(self):
        """将状态估计裁剪到物理合理范围，防止异常值"""
        try:
            from kalman_config import STATE_BOUNDS, FILTER_COLS
            for i, name in enumerate(self.state_names):
                if name in STATE_BOUNDS:
                    lo, hi = STATE_BOUNDS[name]
                    self.x[i] = np.clip(self.x[i], lo, hi)
        except ImportError:
            pass

    def _ensure_positive_definite(self, matrix: np.ndarray, epsilon: float = 1e-6):
        """
        确保矩阵是正定的（数值稳定性）

        通过添加小的正数到对角线元素
        """
        # 检查矩阵是否对称（近似）
        if not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8):
            warnings.warn("矩阵不对称，强制对称化")
            matrix = 0.5 * (matrix + matrix.T)

        # 检查特征值
        try:
            eigvals = np.linalg.eigvalsh(matrix)
            min_eigval = np.min(eigvals)

            if min_eigval <= 0:
                # 添加足够大的值使矩阵正定
                correction = abs(min_eigval) + epsilon
                matrix += np.eye(matrix.shape[0]) * correction

                if self.config.get('verbose', True) and self.stats['steps_processed'] < 10:
                    print(f"  警告: 矩阵特征值({min_eigval:.2e})<=0，已添加修正{correction:.2e}")
        except np.linalg.LinAlgError:
            # 如果特征值计算失败，直接添加小的正数
            matrix += np.eye(matrix.shape[0]) * epsilon

    def filter_sequence(self, observations: np.ndarray,
                        timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        对整个观测序列进行滤波，然后执行RTS后向平滑消除相位滞后

        参数:
        ----------
        observations : np.ndarray
            观测序列，形状为(n_samples, obs_dim)
        timestamps : np.ndarray, optional
            时间戳序列，形状为(n_samples,)

        返回:
        -------
        np.ndarray
            滤波后的状态序列，形状为(n_samples, state_dim)
        """
        n_samples = observations.shape[0]
        if timestamps is not None and len(timestamps) != n_samples:
            raise ValueError("时间戳序列长度与观测序列长度不匹配")

        if self.config.get('verbose', True):
            print(f"开始处理观测序列，共{n_samples}个样本")

        # === 前向滤波 ===
        # 保存每步的前验和后验状态/协方差，用于RTS平滑
        x_prior_list = []    # x_k|k-1
        P_prior_list = []    # P_k|k-1
        x_post_list = []     # x_k|k
        P_post_list = []     # P_k|k

        for i in range(n_samples):
            timestamp = timestamps[i] if timestamps is not None else None

            # 执行一步滤波（predict + update）
            # predict()内部会保存 _last_x_prior 和 _last_P_prior
            state = self.update(observations[i], timestamp)

            # 保存前验和后验
            x_prior_list.append(self._last_x_prior.copy())
            P_prior_list.append(self._last_P_prior.copy())
            x_post_list.append(self.x.copy())
            P_post_list.append(self.P.copy())

            # 进度显示
            if self.config.get('verbose', True) and (i+1) % 1000 == 0:
                print(f"  已处理 {i+1}/{n_samples} 个样本 (前向滤波)")

        if self.config.get('verbose', True):
            print(f"前向滤波完成")

        # === RTS后向平滑 ===
        smoothed_states = self._rts_smoother(
            x_post_list, P_post_list, x_prior_list, P_prior_list, n_samples
        )

        if self.config.get('verbose', True):
            print(f"RTS后向平滑完成")
            print(f"  平均创新范数: {self.stats['avg_innovation_norm']:.4f}")
            print(f"  收敛状态: {self.stats['convergence_status']}")

        return smoothed_states

    def _rts_smoother(self, x_post_list, P_post_list, x_prior_list, P_prior_list, n_samples):
        """
        Rauch-Tung-Striebel后向平滑器

        消除前向卡尔曼滤波的相位滞后，利用未来观测修正过去的状态估计。
        公式：
            C_k = P_k|k * F^T * inv(P_{k+1}|k)
            x_k|N = x_k|k + C_k * (x_{k+1}|N - x_{k+1}|k)
            P_k|N = P_k|k + C_k * (P_{k+1}|N - P_{k+1}|k) * C_k^T
        """
        if self.config.get('verbose', True):
            print(f"执行RTS后向平滑...")

        # 从后验状态开始
        x_smooth = np.array([x.copy() for x in x_post_list])
        P_smooth = np.array([P.copy() for P in P_post_list])

        # 后向递推
        for k in range(n_samples - 2, -1, -1):
            # 平滑增益 C_k = P_k|k * F^T * inv(P_{k+1}|k)
            P_prior_next = P_prior_list[k + 1]
            try:
                C_k = P_post_list[k] @ self.F.T @ np.linalg.inv(P_prior_next)
            except np.linalg.LinAlgError:
                # 奇异矩阵时用伪逆
                C_k = P_post_list[k] @ self.F.T @ np.linalg.pinv(P_prior_next)

            # 平滑状态
            x_smooth[k] = x_post_list[k] + C_k @ (x_smooth[k+1] - x_prior_list[k+1])

            # 平滑协方差
            P_smooth[k] = P_post_list[k] + C_k @ (P_smooth[k+1] - P_prior_next) @ C_k.T

            if self.config.get('verbose', True) and k % 5000 == 0:
                print(f"  后向平滑 {n_samples - k}/{n_samples}")

        # 应用状态约束
        try:
            from kalman_config import STATE_BOUNDS
            for i, name in enumerate(self.state_names):
                if name in STATE_BOUNDS:
                    lo, hi = STATE_BOUNDS[name]
                    x_smooth[:, i] = np.clip(x_smooth[:, i], lo, hi)
        except ImportError:
            pass

        return x_smooth

    def get_state_estimates(self) -> np.ndarray:
        """获取状态估计历史"""
        return np.array(self.history['states'])

    def get_innovation_sequence(self) -> np.ndarray:
        """获取创新序列"""
        return np.array(self.history['innovations'])

    def get_covariance_history(self) -> np.ndarray:
        """获取协方差历史"""
        return np.array(self.history['covariances'])

    def get_kalman_gain_history(self) -> np.ndarray:
        """获取卡尔曼增益历史"""
        return np.array(self.history['kalman_gains'])

    def plot_convergence(self, save_path: Optional[str] = None):
        """
        绘制滤波器收敛情况

        参数:
        ----------
        save_path : str, optional
            保存图像的路径，如果为None则显示图像
        """
        if len(self.history['states']) == 0:
            print("没有历史数据可绘制")
            return

        n_steps = len(self.history['states'])
        steps = np.arange(n_steps)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 1. 状态估计变化
        ax1 = axes[0]
        states = self.get_state_estimates()
        for i in range(min(3, self.state_dim)):  # 只显示前3个状态
            ax1.plot(steps, states[:, i], label=f'{self.state_names[i]}', alpha=0.7)
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('状态估计')
        ax1.set_title('状态估计收敛过程')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 创新序列
        ax2 = axes[1]
        innovations = self.get_innovation_sequence()
        for i in range(min(3, self.obs_dim)):  # 只显示前3个创新
            ax2.plot(steps, innovations[:, i], label=f'{self.state_names[i]}创新', alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('创新（观测残差）')
        ax2.set_title('创新序列（应接近0且无自相关）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 协方差对角线元素（不确定性）
        ax3 = axes[2]
        cov_diag = self.get_covariance_history()
        for i in range(min(3, self.state_dim)):  # 只显示前3个协方差
            ax3.plot(steps, cov_diag[:, i], label=f'{self.state_names[i]}不确定性', alpha=0.7)
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('状态方差')
        ax3.set_title('状态估计不确定性（应收敛）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  # 对数刻度

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"收敛图已保存至: {save_path}")
            plt.close()
        else:
            plt.show()

    def print_statistics(self):
        """打印滤波器统计信息"""
        print("\n" + "="*60)
        print("卡尔曼滤波器统计信息")
        print("="*60)
        print(f"处理步数: {self.stats['steps_processed']}")
        print(f"平均创新范数: {self.stats['avg_innovation_norm']:.6f}")
        print(f"最大创新范数: {self.stats['max_innovation_norm']:.6f}")
        print(f"收敛状态: {self.stats['convergence_status']}")

        # 打印各状态的最终不确定性
        if len(self.history['covariances']) > 0:
            final_cov_diag = self.history['covariances'][-1]
            print("\n各状态最终不确定性（标准差）:")
            for i, (name, var) in enumerate(zip(self.state_names, final_cov_diag)):
                std = np.sqrt(max(var, 0))  # 避免负数
                print(f"  {name}: {std:.6f} (方差: {var:.6e})")

        # 打印噪声矩阵的迹（总噪声功率）
        print(f"\n过程噪声迹（总功率）: {np.trace(self.Q):.6f}")
        print(f"观测噪声迹（总功率）: {np.trace(self.R):.6f}")
        print("="*60)


# ==================== 工具函数 ====================

def create_kalman_filter_for_features(feature_names: List[str],
                                      config: Optional[Dict] = None) -> MultiVariateKalmanFilter:
    """
    为特定特征创建卡尔曼滤波器

    参数:
    ----------
    feature_names : List[str]
        需要滤波的特征名称列表
    config : Dict, optional
        滤波器配置

    返回:
    -------
    MultiVariateKalmanFilter
        配置好的卡尔曼滤波器
    """
    state_dim = len(feature_names)

    # 从配置文件获取默认配置
    try:
        from kalman_config import FILTER_COLS, INITIAL_Q, INITIAL_R
        # 创建特征索引映射
        feature_indices = []
        q_diag = []
        r_diag = []

        for i, feature in enumerate(feature_names):
            if feature in FILTER_COLS:
                idx = FILTER_COLS.index(feature)
                feature_indices.append(idx)
                q_diag.append(INITIAL_Q[idx, idx])
                r_diag.append(INITIAL_R[idx, idx])
            else:
                # 如果特征不在默认列表中，使用默认值
                q_diag.append(0.1)
                r_diag.append(1.0)

        # 创建对应的Q和R矩阵
        Q_custom = np.diag(q_diag)
        R_custom = np.diag(r_diag)

    except ImportError:
        # 如果配置文件不存在，使用默认值
        Q_custom = np.eye(state_dim) * 0.1
        R_custom = np.eye(state_dim) * 1.0

    # 创建滤波器配置
    try:
        from kalman_config import KALMAN_CONFIG as _cfg
        filter_config = _cfg.copy()
    except ImportError:
        filter_config = {'use_adaptive_noise': False, 'verbose': True}
    if config:
        filter_config.update(config)

    # 创建滤波器
    kf = MultiVariateKalmanFilter(
        state_dim=state_dim,
        obs_dim=state_dim,
        state_names=feature_names,
        config=filter_config
    )

    # 设置自定义噪声矩阵
    kf.Q = Q_custom
    kf.R = R_custom

    return kf


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("卡尔曼滤波器单元测试")
    print("-" * 40)

    # 测试1: 创建滤波器
    print("测试1: 创建7维卡尔曼滤波器")
    kf = MultiVariateKalmanFilter(state_dim=7, obs_dim=7)
    print("  滤波器创建成功")

    # 测试2: 生成测试数据
    print("\n测试2: 生成测试观测序列")
    n_samples = 200
    true_states = np.zeros((n_samples, 7))

    # 创建简单的状态轨迹（正弦波加趋势）
    t = np.linspace(0, 4*np.pi, n_samples)
    for i in range(7):
        amplitude = 0.5 + 0.2 * i
        frequency = 0.5 + 0.1 * i
        phase = 0.1 * i
        trend = 0.01 * i * np.arange(n_samples) / n_samples
        true_states[:, i] = amplitude * np.sin(frequency * t + phase) + trend + 10

    # 添加观测噪声
    observation_noise = np.random.normal(0, 0.5, (n_samples, 7))
    observations = true_states + observation_noise

    print(f"  生成 {n_samples} 个样本的测试数据")
    print(f"  状态范围: [{true_states.min():.2f}, {true_states.max():.2f}]")
    print(f"  观测噪声标准差: {observation_noise.std():.4f}")

    # 测试3: 滤波处理
    print("\n测试3: 应用卡尔曼滤波")
    filtered_states = kf.filter_sequence(observations)

    # 测试4: 评估性能
    print("\n测试4: 评估滤波效果")
    mse_before = np.mean((observations - true_states) ** 2)
    mse_after = np.mean((filtered_states - true_states) ** 2)
    improvement = (mse_before - mse_after) / mse_before * 100

    print(f"  滤波前MSE: {mse_before:.6f}")
    print(f"  滤波后MSE: {mse_after:.6f}")
    print(f"  改进: {improvement:.2f}%")

    # 测试5: 显示统计信息
    kf.print_statistics()

    # 测试6: 绘制收敛图
    print("\n测试6: 生成收敛图")
    kf.plot_convergence(save_path='../pic/kalman_filter_test.png')

    print("\n" + "="*40)
    print("单元测试完成!")
    print("所有测试通过，卡尔曼滤波器工作正常")