import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, tanh, diff, lambdify

# 定义函数 f(x) = 1 + 0.5 * tanh(2x)
def f(x):
    """原始函数 f(x) = 1 + 0.5*tanh(2x)"""
    return 1 + 0.5 * np.tanh(2 * x)

# 使用 Sympy 获取解析导数函数
def get_analytical_derivative():
    """获取解析导数函数"""
    x = symbols('x')
    expr = diff(1 + 0.5 * tanh(2 * x), x)
    return lambdify(x, expr)

# 使用中心差分法计算数值导数
def calculate_central_difference(x, f):
    """使用中心差分法计算数值导数
    
    参数：
        x: numpy数组，输入点
        f: 可调用函数，要求导的函数
    
    返回：
        numpy数组，中心差分法计算的导数值
    """
    dy = []
    for i in range(1, len(x)-1):
        h = x[i+1] - x[i]  # 计算步长
        dy.append((f(x[i+1]) - f(x[i-1])) / (2 * h))
    return np.array(dy)

# 使用 Richardson 外推法计算不同阶数的导数值
def richardson_derivative_all_orders(x, f, h, max_order=3):
    """使用 Richardson 外推法计算不同阶数的导数值
    
    参数：
        x: 标量或numpy数组，输入点
        f: 可调用函数，要求导的函数
        h: 浮点数，初始步长
        max_order: 整数，最大外推阶数
    
    返回：
        列表，不同阶数计算的导数值
    """
    # 初始化 Richardson 外推表
    R = np.zeros((max_order + 1, max_order + 1))
    
    # 计算第一列（不同步长的中心差分）
    for i in range(max_order + 1):
        hi = h / (2**i)
        R[i, 0] = (f(x + hi) - f(x - hi)) / (2 * hi)
    
    # Richardson 外推过程
    for j in range(1, max_order + 1):
        for i in range(max_order - j + 1):
            R[i, j] = (4**j * R[i+1, j-1] - R[i, j-1]) / (4**j - 1)
    
    # 返回不同阶数的结果
    return [R[0, j] for j in range(1, max_order + 1)]

# 创建对比图，展示导数计算结果和误差分析
def create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical):
    """创建对比图，展示导数计算结果和误差分析
    
    参数：
        x: numpy数组，所有x坐标点
        x_central: numpy数组，中心差分法使用的x坐标点
        dy_central: numpy数组，中心差分法计算的导数值
        dy_richardson: numpy数组，Richardson外推法计算的导数值
        df_analytical: 可调用函数，解析导数函数
    """
    # 创建四个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # 计算解析导数
    analytical = df_analytical(x)
    analytical_central = df_analytical(x_central)
    
    # 1. 导数对比图
    ax1.plot(x, analytical, 'b-', label='Analytical Solution')
    ax1.plot(x_central, dy_central, 'ro', markersize=4, label='Central Difference')
    ax1.plot(x, dy_richardson[:, 1], 'g^', markersize=4, label='Richardson (2nd Order)')
    ax1.set_title('Derivative Comparison')
    ax1.set_xlabel('x')
    ax1.set_ylabel('dy/dx')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 误差分析图
    error_central = np.abs(dy_central - analytical_central)
    error_richardson = np.abs(dy_richardson[:, 1] - analytical)
    
    ax2.plot(x_central, error_central, 'ro', markersize=4, label='Central Difference Error')
    ax2.plot(x, error_richardson, 'g^', markersize=4, label='Richardson Error')
    ax2.set_yscale('log')
    ax2.set_title('Error Analysis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Absolute Error (log scale)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Richardson 外推不同阶数误差对比图
    for i, order in enumerate(['1st', '2nd', '3rd']):
        error = np.abs(dy_richardson[:, i] - analytical)
        ax3.plot(x, error, marker='^', markersize=4, label=f'Richardson {order}')
    ax3.set_yscale('log')
    ax3.set_title('Richardson Extrapolation Error Comparison')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Absolute Error (log scale)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 步长敏感性分析图
    h_values = np.logspace(-6, -1, 20)
    x_test = 0.0  # 在x=0处分析步长敏感性
    central_errors = []
    richardson_errors = []
    expected = df_analytical(x_test)
    
    for h in h_values:
        # 中心差分法误差
        central_result = (f(x_test + h) - f(x_test - h)) / (2 * h)
        central_errors.append(abs(central_result - expected))
        
        # Richardson 外推法误差（2阶）
        rich_result = richardson_derivative_all_orders(x_test, f, h, max_order=3)[1]
        richardson_errors.append(abs(rich_result - expected))
    
    ax4.loglog(h_values, central_errors, 'ro-', label='Central Difference')
    ax4.loglog(h_values, richardson_errors, 'g^-', label='Richardson (2nd Order)')
    ax4.set_title('Step Size Sensitivity Analysis')
    ax4.set_xlabel('Step Size h (log scale)')
    ax4.set_ylabel('Absolute Error (log scale)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# 主函数，运行数值微分实验
def main():
    """主函数，运行数值微分实验"""
    # 设置实验参数
    h_initial = 0.1  # 初始步长
    max_order = 3     # 最大外推阶数
    N_points = 200    # 采样点数
    x = np.linspace(-2, 2, N_points)
    
    # 获取解析导数函数
    df_analytical = get_analytical_derivative()
    
    # 计算中心差分导数
    dy_central = calculate_central_difference(x, f)
    x_central = x[1:-1]
    
    # 计算 Richardson 外推导数
    dy_richardson = np.array([
        richardson_derivative_all_orders(xi, f, h_initial, max_order=max_order)
        for xi in x
    ])
    
    # 绘制结果对比图
    create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical)

if __name__ == '__main__':
    main()
