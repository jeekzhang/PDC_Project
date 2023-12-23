import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpi4py import MPI

w, h = 400, 300
O = np.array([0.0, 0.35, -1.0])
Q = np.array([0.0, 0.0, 0.0])
img = np.zeros((h, w, 3))
r = float(w) / h
S = (-1.0, -1.0 / r + 0.25, 1.0, 1.0 / r + 0.25)

start_time = MPI.Wtime()


# 归一化
def normalize(x):
    return x / np.linalg.norm(x)


# 射线与球的相交测试
def intersect_sphere(origin, dir, obj):
    OC = obj["position"] - origin
    radius = obj["radius"]
    if (np.linalg.norm(OC) < radius) or (np.dot(OC, dir) < 0):
        return np.inf
    l = np.linalg.norm(np.dot(OC, dir))
    m_square = np.linalg.norm(OC) * np.linalg.norm(OC) - l * l
    q_square = radius * radius - m_square
    return (l - np.sqrt(q_square)) if q_square >= 0 else np.inf


# 获得物体表面某点处的单位法向量
def get_normal(obj, point):
    return normalize(point - obj["position"])


def get_color(obj, M):
    color = obj["color"]
    if not hasattr(color, "__len__"):
        color = color(M)
    return color


def sphere(
    position, radius, color, reflection=0.5, diffuse=0.5, specular_c=0.6, specular_k=50
):
    return dict(
        type="sphere",
        position=np.array(position),
        radius=np.array(radius),
        color=np.array(color),
        reflection=reflection,
        diffuse=diffuse,
        specular_c=specular_c,
        specular_k=specular_k,
    )


# 定义场景中的球体：球心位置，半径，颜色
scene = [
    sphere([0.75, 0.1, 1.0], 0.6, [1, 0, 0]),
    sphere([-0.3, 0.01, 0.2], 0.3, [0, 1, 0]),
    sphere([-2.75, 0.1, 3.5], 0.6, [0, 0, 1]),
]

light_point = np.array([5.0, 5.0, -10.0])  # 点光源位置
light_color = np.array([1.0, 1.0, 1.0])  # 点光源的颜色值
ambient = 0.05  # 环境光


# 像素点着色
def intersect_color(origin, dir, intensity):
    min_distance = np.inf
    for i, obj in enumerate(scene):
        current_distance = intersect_sphere(origin, dir, obj)
        if current_distance < min_distance:
            min_distance, obj_index = current_distance, i  # 记录最近的交点距离和对应的物体
    if (min_distance == np.inf) or (intensity < 0.01):
        return np.array([0.0, 0.0, 0.0])

    obj = scene[obj_index]
    P = origin + dir * min_distance  # 交点坐标
    color = get_color(obj, P)
    N = get_normal(obj, P)  # 交点处单位法向量
    PL = normalize(light_point - P)
    PO = normalize(origin - P)

    c = ambient * color

    # 阴影测试
    l = [
        intersect_sphere(P + N * 0.0001, PL, obj_shadow_test)
        for i, obj_shadow_test in enumerate(scene)
        if i != obj_index
    ]
    if not (l and min(l) < np.linalg.norm(light_point - P)):
        c += obj["diffuse"] * max(np.dot(N, PL), 0) * color * light_color
        c += (
            obj["specular_c"]
            * max(np.dot(N, normalize(PL + PO)), 0) ** obj["specular_k"]
            * light_color
        )

    reflect_ray = dir - 2 * np.dot(dir, N) * N  # 计算反射光线
    c += obj["reflection"] * intersect_color(
        P + N * 0.0001, reflect_ray, obj["reflection"] * intensity
    )
    return np.clip(c, 0, 1)


for i, x in enumerate(tqdm(np.linspace(S[0], S[2], w), desc="Processing rows")):
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        Q[:2] = (x, y)
        img[h - j - 1, i, :] = intersect_color(O, normalize(Q - O), 1)

end_time = MPI.Wtime()
print(end_time - start_time)
plt.imsave("img_serial.png", img)
