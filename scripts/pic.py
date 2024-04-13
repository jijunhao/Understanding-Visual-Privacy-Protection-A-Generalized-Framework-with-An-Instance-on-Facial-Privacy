import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import seaborn as sns

matplotlib.rc('font', family='Times New Roman')

# 三个指标值，假设在 0 到 100 之间
d1 = 68.48
d2 = 46.87
D = d1 / (d1 + d2) * 100

plt.figure(figsize=(4, 3))
# 创建一个新的 Figure
fig, ax = plt.subplots()

# 设置柱状图的宽度
width = 0.6

# 将 values 转换为在 0 到 100 之间
values = [d1, d2]

# 绘制 d1 和 d2 的条形图，使用淡一点的颜色
indicators = ['$D_1$', '$D_2$']
colors = [sns.color_palette("pastel")[0], sns.color_palette("pastel")[1]]
bars = ax.bar(indicators, values, width, color=colors)

# 添加 d1 和 d2 的数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height - 20, f'{height:.2f}%', ha='center', va='bottom', fontsize=30)

# 添加 D 的标题，调整位置和字体
ax.set_title(f'D-R = {D:.4f}%', loc='center', fontsize=30)

# 设置 y 轴刻度范围为 0 到 100，步长为 10
ax.yaxis.set_major_locator(ticker.MultipleLocator(base=10))

# 去掉右边和上面的边框
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# 设置刻度字体为 "DejaVu Serif"
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.set_yticks([])
plt.savefig('box.png',dpi=500 ,bbox_inches='tight')
# 显示图形
plt.show()
