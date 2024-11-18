import numpy as np
import matplotlib.pyplot as plt

# 定义n的范围
n_values = np.linspace(0.1, 10, 400)  # 从0.1到10，避免log(0)的情况

# 设置底数b的范围
b_values = np.linspace(1, 10, 10)  # 底数从1到10之间

# 创建一个图形
plt.figure(figsize=(10, 6))

# 为每个底数绘制曲线
for b in b_values:
    y_values = np.log(n_values) / np.log(b)  # 计算log_b(n) 使用change-of-base公式
    plt.plot(n_values, y_values, label=f'b={b:.1f}')

# 设置图形标题和标签
plt.title('y = log_b(n) for different b values')
plt.xlabel('n')
plt.ylabel('y')

# 添加图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
