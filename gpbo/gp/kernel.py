import numpy as np
from gpbo.gp import utils

class EXP:
    """指数カーネル

    Attributes:
        theta (float): スケールパラメータ
    """
    def __init__(self, theta:float)->None:
        self.theta = theta
    def get_kernel_matrix(self, Xi:np.ndarray, Xj:np.ndarray)->np.ndarray:
        """カーネル行列を計算

        Args:
            Xi (np.ndarray): 第一引数バッチデータ。shapeは(Ni, dim)
            Xj (np.ndarray): 第二引数バッチデータ。shapeは(Nj, dim)

        Returns:
            np.ndarray: カーネル行列。shapeは(Ni, Nj)。
        """
        norm = utils.get_norm(Xi, Xj)
        return np.exp(-norm/self.theta)