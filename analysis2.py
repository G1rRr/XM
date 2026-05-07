# ============================================================
# 第二阶段：深度分析 + 优化可视化
# 新增：流失漏斗细化 + 商品群画像标签 + 渠道效率对比
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 读取数据（和第一阶段一样）──
df = pd.read_csv("data.csv", encoding="utf-8", parse_dates=["日期"])
df_clean = df[df["浏览次数"] > 0].copy()

# 构造转化率字段
df_clean["加购转化率"] = df_clean["加购物车人次"] / df_clean["网页访客数"].replace(0, np.nan)
df_clean["下单转化率"] = df_clean["下单人数"] / df_clean["网页访客数"].replace(0, np.nan)
df_clean["成交转化率"] = df_clean["成交人次"] / df_clean["下单人数"].replace(0, np.nan)
df_clean["客单价"]    = df_clean["成交金额"] / df_clean["成交人次"].replace(0, np.nan)

# ── 重新聚类（列名已修正）──
cluster_features = ["浏览次数", "成交金额", "成交件数（目标变量）", "加购物车人次", "下单人数"]
df_cluster = df_clean[cluster_features].dropna().copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster["聚类标签"] = kmeans.fit_predict(X_scaled)

# 给每个群贴业务标签（根据成交金额均值排序）
profile = df_cluster.groupby("聚类标签")["成交金额"].mean().sort_values()
label_map = {
    profile.index[0]: "长尾冷门品",
    profile.index[1]: "潜力成长品",
    profile.index[2]: "核心爆款",
    profile.index[3]: "超级爆款",
}
df_cluster["商品类型"] = df_cluster["聚类标签"].map(label_map)

# 把商品类型合并回主表
df_clean = df_clean.reset_index(drop=True)
df_clean["商品类型"] = df_cluster["商品类型"]

print("各商品群数量分布:")
print(df_cluster["商品类型"].value_counts())

# ============================================================
# 可视化：稳定布局版
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("电商商品销售深度分析报告", fontsize=15, fontweight="bold")

# ── 图1：漏斗图（左上）──
ax1 = axes[0][0]

funnel_labels = ["曝光", "访问", "加购", "下单", "成交"]
funnel_values = [
    df_clean["浏览次数"].sum(),
    df_clean["网页访客数"].sum(),
    df_clean["加购物车人次"].sum(),
    df_clean["下单人数"].sum(),
    df_clean["成交人次"].sum(),
]
loss_rates = [(funnel_values[i-1] - funnel_values[i]) / funnel_values[i-1]
              for i in range(1, len(funnel_values))]
worst_idx = loss_rates.index(max(loss_rates)) + 1

bar_colors = ["#4C8BF5"] * len(funnel_labels)
bar_colors[worst_idx] = "#E84040"

bars = ax1.bar(funnel_labels, funnel_values, color=bar_colors, width=0.45, alpha=0.85)
for bar, val in zip(bars, funnel_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
             f"{val/1e6:.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")

for i, loss in enumerate(loss_rates):
    color = "#E84040" if (i+1) == worst_idx else "#666"
    ax1.text(i + 0.5, max(funnel_values) * 0.4,
             f"▼{loss:.1%}",
             ha="center", fontsize=9, color=color, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor=color, alpha=0.85))

ax1.set_title("购买漏斗各层流失分析", fontsize=11)
ax1.set_ylabel("用户量 / 次数")
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

# ── 图2：渠道对比（右上）──
ax2 = axes[0][1]

channel_stats = pd.DataFrame({
    "渠道": ["直通车", "淘宝客", "聚划算", "搜索"],
    "引导浏览次数": [
        df_clean["直通车引导浏览次数"].sum(),
        df_clean["淘宝客引导浏览次数"].sum(),
        df_clean["聚划算引导浏览次数"].sum(),
        df_clean["搜索引导浏览次数"].sum(),
    ],
})
channel_stats["占比"] = (channel_stats["引导浏览次数"] /
                         channel_stats["引导浏览次数"].sum() * 100).round(1)
channel_stats = channel_stats.sort_values("引导浏览次数", ascending=True)

bar_colors2 = ["#CCCCCC", "#CCCCCC", "#4C8BF5", "#E84040"]
bars2 = ax2.barh(channel_stats["渠道"], channel_stats["引导浏览次数"],
                  color=bar_colors2, height=0.4, alpha=0.88)
for bar, row in zip(bars2, channel_stats.itertuples()):
    ax2.text(bar.get_width() + channel_stats["引导浏览次数"].max() * 0.02,
             bar.get_y() + bar.get_height()/2,
             f"{row.引导浏览次数:,.0f}次 ({row.占比}%)",
             va="center", fontsize=9)

ax2.set_title("各渠道引导浏览次数对比", fontsize=11)
ax2.set_xlabel("引导浏览次数")
ax2.spines[["top", "right"]].set_visible(False)
ax2.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax2.set_xlim(0, channel_stats["引导浏览次数"].max() * 1.4)

# ── 图3：商品分层气泡图（左下）──
ax3 = axes[1][0]

type_colors = {
    "长尾冷门品": "#CCCCCC",
    "潜力成长品": "#4ECDC4",
    "核心爆款":   "#4C8BF5",
    "超级爆款":   "#E84040",
}
for t, color in type_colors.items():
    sub = df_cluster[df_cluster["商品类型"] == t]
    ax3.scatter(
        sub["浏览次数"], sub["成交金额"],
        c=color,
        label=f"{t}（{len(sub):,}个）",
        alpha=0.55,
        s=sub["成交件数（目标变量）"].clip(upper=500) + 10,
        edgecolors="none"
    )

# 标注超级爆款
super_item = df_cluster[df_cluster["商品类型"] == "超级爆款"]
if not super_item.empty:
    ax3.annotate("超级爆款",
        xy=(super_item["浏览次数"].values[0], super_item["成交金额"].values[0]),
        xytext=(super_item["浏览次数"].values[0] * 0.65,
                super_item["成交金额"].values[0] * 0.82),
        fontsize=9, color="#E84040",
        arrowprops=dict(arrowstyle="->", color="#E84040"))

ax3.set_title("商品分层气泡图（气泡大小=成交件数）", fontsize=11)
ax3.set_xlabel("浏览次数")
ax3.set_ylabel("成交金额（元）")
ax3.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax3.spines[["top", "right"]].set_visible(False)
ax3.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e4:.0f}万"))

# ── 图4：各商品群均值对比条形图（右下）──
ax4 = axes[1][1]

profile_means = df_cluster.groupby("商品类型")[["浏览次数", "成交金额", "加购物车人次"]].mean()
# 只显示成交金额均值，按大小排序
profile_means = profile_means.sort_values("成交金额", ascending=True)
colors4 = ["#CCCCCC", "#4ECDC4", "#4C8BF5", "#E84040"]
bars4 = ax4.barh(profile_means.index,
                  profile_means["成交金额"],
                  color=colors4, height=0.4, alpha=0.88)
for bar, val in zip(bars4, profile_means["成交金额"]):
    ax4.text(bar.get_width() + profile_means["成交金额"].max() * 0.02,
             bar.get_y() + bar.get_height()/2,
             f"{val/1e4:.1f}万元", va="center", fontsize=9)

ax4.set_title("各商品群平均成交金额对比", fontsize=11)
ax4.set_xlabel("平均成交金额（元）")
ax4.spines[["top", "right"]].set_visible(False)
ax4.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e4:.0f}万"))
ax4.set_xlim(0, profile_means["成交金额"].max() * 1.3)

plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.55, wspace=0.38)
plt.savefig("analysis_report2.png", dpi=150, bbox_inches="tight", facecolor="white")
print("图表已保存为 analysis_report2.png")
plt.show()
print("✅ 第二阶段完成！")