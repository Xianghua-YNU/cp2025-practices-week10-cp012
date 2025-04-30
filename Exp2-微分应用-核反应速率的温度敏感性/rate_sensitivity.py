import numpy as np
import matplotlib.pyplot as plt

def q3a(T):
    """
    计算 3-alpha 反应速率中与温度相关的部分 q / (rho^2 Y^3)
    输入: T - 温度 (K)
    返回: 速率因子 (erg * cm^6 / (g^3 * s))
    """
    # 将温度转换为以 10^8 K 为单位
    T8 = T / 1e8
    
    # 处理温度为零或接近零的特殊情况
    if T8<= 0:
        return 0.0
    
    # 计算速率因子
    rate = 5.09e11 * (T8 ** -3) * np.exp(-44.027 / T8)
    return rate

def plot_rate(filename="rate_vs_temp.png"):
    """绘制速率因子随温度变化的 log-log 图"""
    # 生成温度数据点 (从 1e7 K 到 1e10 K)
    temperatures = np.logspace(np.log10(3.0e8), np.log10(5.0e9), 100)
    
    # 计算对应的速率值
    rates = [q3a(T) for T in temperatures]
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.loglog(temperatures, rates, linewidth=2)
    
    # 添加标签和标题
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Reaction Rate Factor (erg·cm$^6$·g$^{-3}$·s$^{-1}$)', fontsize=12)
    plt.title('3-alpha Reaction Rate vs Temperature', fontsize=14)
    plt.grid(True, which="both", ls="--")
    
    # 保存图形
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 计算并打印 nu 值
    print("   温度 T (K)    :   ν (敏感性指数)")
    print("--------------------------------------")

    temperatures_K = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    h = 1.0e-8  # 扰动因子

    for T in temperatures_K:
        # 计算中心点的速率
        q = q3a(T)
        
        if q == 0:
            nu = 0.0
        else:
            # 计算扰动后的速率
            q_perturbed = q3a(T * (1 + h))
            
            # 计算对数导数 (敏感性指数)
            dq_dT = (q_perturbed - q) / (T * h)
            nu = (T * dq_dT) / q
        
        print(f"{T:12.1e} : {nu:15.4f}")

    # 调用绘图函数展示结果
    plot_rate()
