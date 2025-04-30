import numpy as np

# 定义被积函数
def f(x):
    """
    被积函数 f(x) = x^4 - 2x + 1
    """
    return x**4 - 2*x + 1

# 梯形法则积分函数
def trapezoidal(f, a, b, N):
    """
    梯形法数值积分
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param N: 子区间数
    :return: 积分近似值
    """
    h = (b - a) / N  # 区间宽度
    x_values = np.linspace(a, b, N + 1)  # 生成均匀划分的点
    
    integral = (h / 2) * (f(x_values[0]) + f(x_values[-1]))
    for i in range(1, N):
        integral += h * f(x_values[i])
    
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
    if N % 2 != 0:
        raise ValueError("Simpson法要求N为偶数")
    
    h = (b - a) / N
    x_values = np.linspace(a, b, N + 1)
    
    integral = h / 3 * (f(x_values[0]) + f(x_values[-1]))
    
    # 计算奇数次和偶数次节点的权重
    for i in range(1, N):
        if i % 2 == 0:
            weight = 2
        else:
            weight = 4
        integral += h / 3 * weight * f(x_values[i])
    
    return integral

def main():
    a, b = 0, 2  # 积分区间
    exact_integral = 4.4  # 准确积分值
    
    for N in [100, 1000]:
        print(f"子区间数 N = {N}")
        
        # 梯形法则
        trapezoidal_result = trapezoidal(f, a, b, N)
        trapezoidal_error = abs(trapezoidal_result - exact_integral) / exact_integral
        
        # Simpson法则
        simpson_result = simpson(f, a, b, N)
        simpson_error = abs(simpson_result - exact_integral) / exact_integral
        
        # 格式化输出结果
        print(f"梯形法则结果: {trapezoidal_result:.8f}")
        print(f"梯形法则相对误差: {trapezoidal_error:.2e}")
        print(f"Simpson法则结果: {simpson_result:.8f}")
        print(f"Simpson法则相对误差: {simpson_error:.2e}")
        print("-" * 40)

if __name__ == '__main__':
    main()
