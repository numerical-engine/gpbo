import numpy as np

class GP:
    """ガウス過程回帰用モデル

    Attributes:
        kernel (any): カーネル関数クラス。
        Xf (np.ndarray): 入力データ。shapeは(Ni, dim)
        Yf (np.ndarray): 出力データ。shapeは(Ni., 1)
        sigma (float): 観測ノイズ。
    """
    def __init__(self, Xf:np.ndarray, Yf:np.ndarray, kernel:any, sigma:float = 0.)->None:
        data_num, _ = Xf.shape
        assert Yf.shape == (data_num, 1)

        self.kernel = kernel
        self.Xf = Xf
        self.Yf = Yf
        self.sigma = sigma
    
    def get_kernel_matrix(self, Xm:np.ndarray, cat:bool = False)->np.ndarray|tuple[np.ndarray]:
        """カーネル行列の計算

        Args:
            Xm (np.ndarray): 第二引数。*第一引数はself.Xf。
            cat (bool, optional): カーネル行列で返すか、部分行列のタプルで返すか。

        Returns:
            np.ndarray: カーネル行列。もしくは部分行列。
        """
        Kff = self.kernel.get_kernel_matrix(self.Xf, self.Xf)
        Kmf = self.kernel.get_kernel_matrix(Xm, self.Xf)
        Kfm = Kmf.T
        Kmm = self.kernel.get_kernel_matrix(Xm, Xm)

        if cat:
            K1 = np.concatenate((Kff, Kmf), axis = 0)
            K2 = np.concatenate((Kfm, Kmm), axis = 0)
            K = np.concatenate((K1, K2), axis = 1)
            return K
        else:
            return Kff, Kfm, Kmf, Kmm

    
    def pred(self, Xm:np.ndarray)->tuple[np.ndarray]:
        """入力データに対する予測

        Args:
            Xm (np.ndarray): 入力データ。shapeは(M, dim)
        Returns:
            tuple[np.ndarray]: 平均関数(shapeは(M, 1))および分散共分散行列(shapeは(M, M))。
        """
        Y_mean = np.mean(self.Yf)
        Yf = self.Yf - Y_mean
        Kff, Kfm, Kmf, Kmm = self.get_kernel_matrix(Xm, cat = False)

        mu = Kmf@np.linalg.inv(Kff + (self.sigma**2)*np.eye(len(self.Xf)))@Yf
        Sigma = Kmm - Kmf@np.linalg.inv(Kff + (self.sigma**2)*np.eye(len(self.Xf)))@Kfm

        return mu, Sigma