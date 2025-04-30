import numpy as np
import matplotlib.pyplot as plt
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
def calculate_central_difference(x, f, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# 使用 Richardson 外推法计算不同阶数的导数值
def richardson_derivative_all_orders(x, f, h, max_order=3):
    D = np.zeros((max_order + 1, max_order + 1))
    for i in range(max_order + 1):
        D[i, 0] = (f(x + h) - f(x - h)) / (2 * h)
        h /= 2
        for j in range(1, i + 1):
            D[i, j] = D[i, j - 1] + (D[i, j - 1] - D[i - 1, j - 1]) / (4**j - 1)
    return D

# 创建对比图，展示导数计算结果和误差分析
def create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical):
    df_true = df_analytical(x)
    
    # 创建四个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. 导数对比图
    ax1.plot(x, df_true, label='Analytical Derivative', color='black', linestyle='--')
    ax1.plot(x_central, dy_central, label='Central Difference', marker='o')
    ax1.plot(x, dy_richardson, label='Richardson Extrapolation', marker='x')
    ax1.set_title('Derivative Comparison')
    ax1.set_xlabel('x')
    ax1.set_ylabel("f'(x)")
    ax1.legend()
    
    # 2. 误差分析图（对数坐标）
    error_central = np.abs(dy_central - df_analytical(x_central))
    error_richardson = np.abs(dy_richardson - df_analytical(x))
    ax2.loglog(x_central, error_central, label='Central Difference Error', marker='o')
    ax2.loglog(x, error_richardson, label='Richardson Extrapolation Error', marker='x')
    ax2.set_title('Error Analysis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Error (log scale)')
    ax2.legend()
    
    # 3. Richardson 外推不同阶数误差对比图（对数坐标）
    h = 0.1
    errors = []
    for order in range(1, 6):
        dy_richardson_order = richardson_derivative_all_orders(x[1], f, h, max_order=order)[-1, -1]
        error = np.abs(dy_richardson_order - df_analytical(x[1]))
        errors.append(error)
    ax3.loglog(range(1, 6), errors, label='Richardson Extrapolation Error', marker='o')
    ax3.set_title('Richardson Extrapolation Error by Order')
    ax3.set_xlabel('Order')
    ax3.set_ylabel('Error (log scale)')
    ax3.legend()
    
    # 4. 步长敏感性分析图（双对数坐标）
    h_values = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
    central_errors = []
    richardson_errors = []
    for h in h_values:
        dy_central_h = calculate_central_difference(x[1], f, h)
        dy_richardson_h = richardson_derivative_all_orders(x[1], f, h, max_order=3)[-1, -1]
        central_errors.append(np.abs(dy_central_h - df_analytical(x[1])))
        richardson_errors.append(np.abs(dy_richardson_h - df_analytical(x[1])))
    ax4.loglog(h_values, central_errors, label='Central Difference Sensitivity', marker='o')
    ax4.loglog(h_values, richardson_errors, label='Richardson Extrapolation Sensitivity', marker='x')
    ax4.set_title('Step Size Sensitivity Analysis')
    ax4.set_xlabel('Step Size (log scale)')
    ax4.set_ylabel('Error (log scale)')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 设置实验参数
    x = np.linspace(-2, 2, 100)
    h_values = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
    max_order = 3
    
    # 获取解析导数函数
    df_analytical = get_analytical_derivative()
    
    # 计算中心差分导数
    x_central = x[1:-1]  # 中心差分法只能计算内部点的导数
    dy_central = np.zeros_like(x_central)
    for i, h in enumerate(h_values):
        dy_central = calculate_central_difference(x_central, f, h)
    
    # 计算 Richardson 外推导数
    dy_richardson = np.zeros_like(x)
    for i, h in enumerate(h_values):
        dy_richardson = richardson_derivative_all_orders(x, f, h, max_order)[-1, -1]
    
    # 绘制结果对比图
    create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical)

if __name__ == '__main__':
    main()
