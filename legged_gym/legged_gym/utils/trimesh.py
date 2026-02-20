""" This file defines a mesh as a tuple of (vertices, triangles)
All operations are based on numpy ndarray
- vertices: np ndarray of shape (n, 3) np.float32
- triangles: np ndarray of shape (n_, 3) np.uint32
"""
import numpy as np

def get_rotation_matrix(rpy):
    """
    根据 rpy (roll, pitch, yaw) 计算 3x3 旋转矩阵
    对应 x, y, z 轴的旋转
    """
    roll, pitch, yaw = rpy
    
    # 预计算 sin 和 cos
    c_r, c_p, c_y = np.cos([roll, pitch, yaw])
    s_r, s_p, s_y = np.sin([roll, pitch, yaw])

    # 绕 X 轴旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, c_r, -s_r],
        [0, s_r, c_r]
    ])

    # 绕 Y 轴旋转矩阵
    Ry = np.array([
        [c_p, 0, s_p],
        [0, 1, 0],
        [-s_p, 0, c_p]
    ])

    # 绕 Z 轴旋转矩阵 (通常平面旋转只用到这个)
    Rz = np.array([
        [c_y, -s_y, 0],
        [s_y, c_y, 0],
        [0, 0, 1]
    ])

    # 旋转顺序通常是 Z * Y * X (也可根据需要调整)
    return Rz @ Ry @ Rx

def box_trimesh(
        size, # float [3] for x, y, z axis length (in meter) under box frame
        center_position, # float [3] position (in meter) in world frame
        rpy=np.zeros(3), # euler angle (in rad). If you have a single angle theta, pass [0, 0, theta]
    ):

    rpy = np.array(rpy, dtype=np.float32)
    # 1. 创建相对于中心点 (0,0,0) 的顶点
    # 我们先在局部坐标系下生成盒子，不加 center_position
    vertices = np.zeros((8, 3), dtype=np.float32)
    
    # X axis
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    # Y axis
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    # Z axis
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2

    # 2. 处理旋转
    # 如果 rpy 不全为 0，则应用旋转
    if not (rpy == 0).all():
        R = get_rotation_matrix(rpy)
        # 矩阵乘法: (N, 3) dot (3, 3).T -> (N, 3)
        # 或者简单的: vertices = vertices @ R.T
        vertices = np.dot(vertices, R.T)

    # 3. 处理平移
    # 将旋转后的顶点移动到 center_position
    vertices += center_position

    # 4. 定义三角形面 (索引不变，因为拓扑结构没变)
    triangles = -np.ones((12, 3), dtype=np.uint32)
    triangles[0] = [0, 2, 1]
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2]
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6]
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5]
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4]
    triangles[9] = [1, 5, 4]
    triangles[10]= [2, 6, 3]
    triangles[11]= [3, 6, 7]

    return vertices.astype(np.float32), triangles

def combine_trimeshes(*trimeshes):
    if len(trimeshes) > 2:
        return combine_trimeshes(
            trimeshes[0],
            combine_trimeshes(*trimeshes[1:])
        )

    # only two trimesh to combine
    trimesh_0, trimesh_1 = trimeshes
    if trimesh_0[1].shape[0] < trimesh_1[1].shape[0]:
        trimesh_0, trimesh_1 = trimesh_1, trimesh_0
    
    trimesh_1 = (trimesh_1[0], trimesh_1[1] + trimesh_0[0].shape[0])
    vertices = np.concatenate((trimesh_0[0], trimesh_1[0]), axis= 0)
    triangles = np.concatenate((trimesh_0[1], trimesh_1[1]), axis= 0)

    return vertices, triangles

def move_trimesh(trimesh, move: np.ndarray):
    """ inplace operation """
    trimesh[0] += move