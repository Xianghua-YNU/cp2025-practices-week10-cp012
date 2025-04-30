import numpy as np


# 待积分函数
def f(x):
    """
    被积函数 f(x) = x^4 - 2x + 1
    :param x: 自变量，可以是单个值或 NumPy 数组
    :return: 函数值 f(x)
    """
    return x ** 4 - 2 * x + 1


# 梯形法则积分函数（供参考比较用）
def trapezoidal(f, a, b, N):
    """
    梯形法数值积分
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param N: 子区间数
    :return: 积分近似值
    """
    h = (b - a) / N  # 计算步长 h，即每个小梯形的宽度
    x = np.linspace(a, b, N + 1)  # 生成 N+1 个等间距点，从 a 到 b
    y = f(x)  # 计算每个点的函数值
    # 梯形公式：h/2 * (首项 + 2*中间项和 + 末项)
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral


# Simpson法则积分函数
def simpson(f, a, b, N):
    """
    Simpson法数值积分
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param N: 子区间数（必须为偶数）
    :return: 积分近似值
    """
    # 检查 N 是否为偶数，若不是则抛出错误
    if N % 2 != 0:
        raise ValueError("N 必须为偶数，以便分成抛物线段")

    h = (b - a) / N  # 计算步长 h，即每个小段的宽度
    x = np.linspace(a, b, N + 1)  # 生成 N+1 个等间距点，从 a 到 b
    y = f(x)  # 计算每个点的函数值

    # Simpson 公式：h/3 * (首项 + 4*奇数项和 + 2*偶数项和 + 末项)
    odd_sum = np.sum(y[1::2])  # 奇数索引项的和 (i=1,3,5,...,N-1)
    even_sum = np.sum(y[2:-1:2])  # 偶数索引项的和 (i=2,4,6,...,N-2)
    integral = (h / 3) * (y[0] + 4 * odd_sum + 2 * even_sum + y[-1])
    return integral


def main():
    a, b = 0, 2  # 积分区间 [0, 2]
    exact_integral = 4.4  # 精确解，通过解析计算得到

    for N in [100, 1000]:  # 测试不同的子区间数
        # 使用梯形法计算积分
        trapezoidal_result = trapezoidal(f, a, b, N)
        # 计算梯形法的相对误差
        trapezoidal_error = abs(trapezoidal_result - exact_integral) / exact_integral

        # 使用 Simpson 法计算积分
        simpson_result = simpson(f, a, b, N)
        # 计算 Simpson 法的相对误差
        simpson_error = abs(simpson_result - exact_integral) / exact_integral

        # 输出结果，保留 8 位小数显示积分值，误差用科学计数法显示
        print(f"N = {N}")
        print(f"梯形法则结果: {trapezoidal_result:.8f}, 相对误差: {trapezoidal_error:.2e}")
        print(f"Simpson法则结果: {simpson_result:.8f}, 相对误差: {simpson_error:.2e}")
        print("-" * 40)


if __name__ == '__main__':
    main()
