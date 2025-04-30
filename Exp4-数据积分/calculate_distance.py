import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid
import matplotlib.pyplot as plt
import os


def calculate_distance(data=None):
    """
    此函数用于计算总距离和累积距离，并绘制速度和距离随时间变化的图表。
    :param data: 包含时间和速度数据的 numpy 数组，默认为 None
    :return: 总距离和累积距离
    """
    if data is None:
        try:
            data_file = 'Velocities.txt'
            data = np.loadtxt(data_file)
            if data.size == 0:
                print("错误：数据文件为空，请检查文件内容。")
                return None, None
        except FileNotFoundError:
            print("错误：找不到数据文件")
            print("请确保数据文件存在于正确路径")
            return None, None

    if data.ndim == 1:
        t = data[::2]
        v = data[1::2]
    else:
        t = data[:, 0]
        v = data[:, 1]

    total_distance = trapezoid(v, t)
    print(f"总运行距离: {total_distance:.2f} 米")

    distance = cumulative_trapezoid(v, t, initial=0)

    plt.figure(figsize=(10, 6))
    plt.plot(t, v, 'b-', label='Velocity (m/s)')
    plt.plot(t, distance, 'r--', label='Distance (m)')
    plt.title('Velocity and Distance vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s) / Distance (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return total_distance, distance

    
