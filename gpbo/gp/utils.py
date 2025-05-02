import numpy as np

def get_norm(Xi:np.ndarray, Xj:np.ndarray)->np.ndarray:
    """ノルムを計算

    Args:
        Xi (np.ndarray): 第一引数バッチデータ。shapeは(Ni, dim)。
        Xj (np.ndarray): 第二引数バッチデータ。shapeは(Nj, dim)
    Returns:
        np.ndarray: ノルム。shapeは(Ni, Nj)。
    """
    diff = Xi[:, np.newaxis, :] - Xj[np.newaxis, :, :]
    Z = np.sqrt(np.sum(diff ** 2, axis=2))
    return Z