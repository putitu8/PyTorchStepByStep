import numpy as np
import matplotlib.pyplot as plt

# 定义 n 的范围为 [1, 10]
n = np.linspace(0.1, 10, 500)
y = np.log(n)

# 绘制曲线
plt.figure(figsize=(8, 5))
plt.plot(n, y, label='y = log(n)', color='purple')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # x轴
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # y轴
plt.title("Function Curve: y = log(n) (n > 1)", fontsize=14)
plt.xlabel("n", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.show()