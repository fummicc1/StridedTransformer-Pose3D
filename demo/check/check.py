import numpy as np

def distance_from_line(point, line_point, line_dir, eps: float = 1e-8, verbose: bool = False):
    # 点と直線の距離の公式を使用
    if verbose:
        print("point", point)
        print("line_point", line_point)
        # print("line_dir", line_dir)
    return np.linalg.norm(np.cross(line_dir, line_point - point)) / np.linalg.norm(line_dir)

def check_points(points, threshold, eps: float = 1e-8, verbose: bool = False):
    # 3点を通る直線の方向ベクトルを計算
    direction = np.cross(points[1] - points[0], points[2] - points[0]) + eps
    direction /= np.linalg.norm(direction)

    # 各点と直線との距離を計算
    distances = [distance_from_line(points[i], points[0], direction, verbose=verbose) for i in range(3)]

    # 距離が閾値より大きいかどうかをチェック
    return all(d <= threshold for d in distances), distances

def check_has_almost_single_line(points: np.ndarray, threshold: float = 0.1, verbose: bool = False) -> bool:
    on_line, distances = check_points(points, threshold, verbose=verbose)
    if verbose:
        print("distances", distances)
    if on_line:
        if verbose:
            print("3点は近似的に一直線上にある")
        return True
    else:
        if verbose:
            print("3点は一直線上にない")
        return False

def check_did_fail(head: np.ndarray, waist: np.ndarray, foot: np.ndarray, threshold: float = 0.4, verbose: bool = False) -> bool:
    axis = 2
    diff = max(map(lambda num: abs(num), [head[axis] - waist[axis], foot[axis] - waist[axis], head[axis] - foot[axis]]))
    if verbose:
        print(head[axis], waist[axis], foot[axis])
        print("diff", diff)
    return diff < threshold
