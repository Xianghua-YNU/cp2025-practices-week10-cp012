import numpy as np


# 定义被积函数
def f(x):
    """
    定义被积函数 f(x) = x^4 - 2x + 1
    """
    return x ** 4 - 2 * x + 1


# 梯形法则积分函数
def trapezoidal(f, a, b, N):
    """
    梯形法数值积分函数
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param N: 子区间数
    :return: 积分近似值
    """
    # 计算区间宽度
    h = (b - a) / N

    # 生成均匀划分的子区间点
    x_values = np.linspace(a, b, N + 1)

    # 初始积分值为首项和末项的和的一半
    integral = (f(x_values[0]) + f(x_values[-1])) * h / 2

    # 遍历中间的所有点并累加
    for i in range(1, N):
        integral += f(x_values[i]) * h

    return integral


# Simpson法则积分函数
def simpson(f, a, b, N):
    """
    Simpson法则数值积分函数
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param N: 子区间数（必须为偶数）
    :return: 积分近似值
    """
    # 检查N是否为偶数
    if N % 2 != 0:
        raise ValueError("Simpson 法则要求 N 必须为偶数")

    # 计算区间宽度
    h = (b - a) / N

    # 生成均匀划分的子区间点
    x_values = np.linspace(a, b, N + 1)

    # 初始化积分值，包括首项和末项
    integral = (f(x_values[0]) + f(x_values[-1])) * h / 3

    # 遍历中间点，分奇数和偶数索引处理
    for i in range(1, N):
        if i % 2 == 0:  # 偶数索引的权重为2
            weight = 2
        else:  # 奇数索引的权重为4
            weight = 4
        integral += weight * f(x_values[i]) * h / 3

    return integral


def main():
    # 定义积分区间和精确解
    a, b = 0, 2
    exact_integral = 4.4  # 精确积分值为4.4

    # 设定不同的子区间数
    for N in [100, 1000]:
        print(f"子区间数 N = {N}\n")

        # 使用梯形法则计算积分
        trapezoidal_result = trapezoidal(f, a, b, N)

        # 使用Simpson法则计算积分
        simpson_result = simpson(f, a, b, N)

        # 计算相对误差
        trapezoidal_error = abs(trapezoidal_result - exact_integral) / exact_integral
        simpson_error = abs(simpson_result - exact_integral) / exact_integral

        # 格式化输出结果
        print(f"梯形法则结果: {trapezoidal_result:.8f}")
        print(f"梯形法则相对误差: {trapezoidal_error:.2e}")
        print("\n")
        print(f"Simpson法则结果: {simpson_result:.8f}")
        print(f"Simpson法则相对误差: {simpson_error:.2e}")
        print("-" * 40)


if __name__ == '__main__':
    main()
