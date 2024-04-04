import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

# 设置 Seaborn 风格
sns.set(style="whitegrid", palette="husl")

def lorenz(w, t):
    sigma = 10
    rho = 28
    beta = 8/3
    x, y, z = w
    return np.array([sigma*(y-x), rho*x-y-x*z, x*y-beta*z])

t = np.arange(0, 50, 0.01)  # 创建时间点
sol1 = odeint(lorenz, [0.0, 1.0, 0.0], t)  # 第一个初值问题求解
sol2 = odeint(lorenz, [0.0, 1.0001, 0.0], t)  # 第二个初值问题求解

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
fig.suptitle('Lorenz Attractor and Difference between Trajectories', fontsize=16)

# 美化第一个子图
points = np.array([sol1[:, 0], sol1[:, 1], sol1[:, 2]]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, len(sol1))
colors = plt.cm.Blues(norm(np.arange(0, len(sol1))))
lc = Line3DCollection(segments, cmap='Blues', norm=norm)
lc.set_array(np.arange(0, len(sol1)))
line = axes[0].add_collection3d(lc)
axes[0].set_xlabel('$x$', fontsize=14)
axes[0].set_ylabel('$y$', fontsize=14)
axes[0].set_zlabel('$z$', fontsize=14)
axes[0].set_title('Lorenz Attractor', fontsize=16)
axes[0].legend()

axes[0].set_xlim([-15, 15])
axes[0].set_ylim([-25, 25])
axes[0].set_zlim([0, 50])

points = np.array([(sol1[:, 0] - sol2[:, 0]), (sol1[:, 1] - sol2[:, 1]), (sol1[:, 2] - sol2[:, 2])]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, len(sol1))
colors = plt.cm.Greens(norm(np.arange(0, len(sol1))))
lc = Line3DCollection(segments, cmap='Greens', norm=norm)
lc.set_array(np.arange(0, len(sol1)))
line = axes[1].add_collection3d(lc)
axes[1].set_xlabel('$x$', fontsize=14)
axes[1].set_ylabel('$y$', fontsize=14)
axes[1].set_zlabel('$z$', fontsize=14)
axes[1].set_title('Difference between Trajectories', fontsize=16)
axes[1].legend()

axes[1].set_xlim([-30, 30])
axes[1].set_ylim([-50, 50])
axes[1].set_zlim([-10, 30])

plt.subplots_adjust(left=0.05, right=0.9)
plt.show()
