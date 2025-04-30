import numpy as np
from sympy import symbols, tanh, diff, lambdify

# 定义函数 f(x) = 1 + 0.5 * tanh(2x)
def f(x):
    return 1 + 0.5 * np.tanh(2 * x)

# 使用 Sympy 获取解析导数函数
def get_analytical_derivative():
    x_sym = symbols('x')
    f_sym = 1 + 0.5 * tanh(2 * x_sym)
    df_sym = diff(f_sym, x_sym)
    return lambdify(x_sym, df_sym, 'numpy')

# 使用中心差分法计算数值导数
def calculate_central_difference(x, f, h=0.1):
    if isinstance(x, np.ndarray):
        x = x[1:-1]  # 中心差分法只能计算内部点的导数
    return (f(x + h) - f(x - h)) / (2 * h)

# 使用 Richardson 外推法计算不同阶数的导数值
def richardson_derivative_all_orders(x, f, h, max_order=3):
    D = np.zeros((max_order + 1, max_order + 1))
    for i in range(max_order + 1):
        D[i, 0] = (f(x + h) - f(x - h)) / (2 * h)
        h /= 2
        for j in range(1, i + 1):
            D[i, j] = D[i, j - 1] + (D[i, j - 1] - D[i - 1, j - 1]) / (4**j - 1)
    return D[:, -1]  # 返回所有阶数的结果

# 主函数
def main():
    # 设置实验参数
    x = np.linspace(-2, 2, 100)
    h = 0.1
    max_order = 3
    
    # 获取解析导数函数
    df_analytical = get_analytical_derivative()
    
    # 计算中心差分导数
    x_central = x[1:-1]  # 中心差分法只能计算内部点的导数
    dy_central = calculate_central_difference(x, f, h)
    
    # 计算 Richardson 外推导数
    dy_richardson = richardson_derivative_all_orders(x, f, h, max_order)
    
    # 打印结果
    print("解析导数：", df_analytical(x))
    print("中心差分法导数：", dy_central)
    print("Richardson 外推法导数：", dy_richardson)

if __name__ == '__main__':
    main()
