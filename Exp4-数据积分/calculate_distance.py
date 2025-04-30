import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid
import matplotlib.pyplot as plt
import os


def main():
    try:
        # 1. 获取数据文件路径（使用相对路径）
        data_file = 'data.txt'

        # 2. 读取数据（使用 numpy.loadtxt）
        data = np.loadtxt(data_file)

        if data.size == 0:
            print("错误：数据文件为空，请检查文件内容。")
            return

        if data.ndim == 1:
            # 如果是一维数组，假设数据是交替存储时间和速度
            t = data[::2]
            v = data[1::2]
        else:
            t = data[:, 0]  # 时间列
            v = data[:, 1]  # 速度列

        # 3. 计算总距离（使用 scipy.integrate.trapezoid）
        total_distance = trapezoid(v, t)
        print(f"总运行距离: {total_distance:.2f} 米")

        # 4. 计算累积距离（使用 scipy.integrate.cumulative_trapezoid）
        distance = cumulative_trapezoid(v, t, initial=0)

        # 5. 绘制图表
        plt.figure(figsize=(10, 6))
        plt.plot(t, v, 'b-', label='Velocity (m/s)')
        plt.plot(t, distance, 'r--', label='Distance (m)')
        plt.title('Velocity and Distance vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s) / Distance (m)')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print("错误：找不到数据文件")
        print("请确保数据文件存在于正确路径")


if __name__ == '__main__':
    main()
    
    
