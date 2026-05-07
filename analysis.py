# ============================================================
# 电商商品销售数据分析
# 目标：漏斗分析 + 渠道归因 + 商品分层 + 可视化报告
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ---------- 0. 中文字体设置 ----------
# Windows 用户使用 SimHei（黑体），Mac 用户改成 Arial Unicode MS
matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ============================================================
# 第一部分：读取数据
# ============================================================

# 定义列名（因为你的CSV第一行是中文列名，直接读取即可）
df = pd.read_csv(
    "data.csv",
    encoding="utf-8",          # 如果报错改成 "gbk" 或 "gb18030"
    parse_dates=["日期"],       # 自动把日期列解析为datetime格式
)

# 快速查看数据基本情况
print("=" * 50)
print("数据维度（行数, 列数）:", df.shape)
print("\n前3行数据:")
print(df.head(3))
print("\n数据类型:")
print(df.dtypes)
print("\n缺失值统计:")
print(df.isnull().sum())
print("=" * 50)


# ============================================================
# 第二部分：数据清洗
# ============================================================

# 删除浏览次数为0的行（无效曝光，不进入漏斗分析）
df_clean = df[df["浏览次数"] > 0].copy()

# 构造新字段：转化相关指标
# 点击转化率 = 网页访客数 / 浏览次数（每次浏览带来访客的比例）
df_clean["点击转化率"] = df_clean["网页访客数"] / df_clean["浏览次数"]

# 加购转化率 = 加购物车人次 / 网页访客数（访客中有多少人加购）
df_clean["加购转化率"] = df_clean["加购物车人次"] / df_clean["网页访客数"].replace(0, np.nan)

# 下单转化率 = 下单人数 / 网页访客数
df_clean["下单转化率"] = df_clean["下单人数"] / df_clean["网页访客数"].replace(0, np.nan)

# 成交转化率 = 成交人次 / 下单人数（下单后实际付款的比例）
df_clean["成交转化率"] = df_clean["成交人次"] / df_clean["下单人数"].replace(0, np.nan)

# 客单价 = 成交金额 / 成交人次
df_clean["客单价"] = df_clean["成交金额"] / df_clean["成交人次"].replace(0, np.nan)

# 各渠道总引导浏览次数（用于渠道归因分析）
df_clean["总渠道引导"] = (
    df_clean["直通车引导浏览次数"]
    + df_clean["淘宝客引导浏览次数"]
    + df_clean["搜索引导浏览次数"]
    + df_clean["聚划算引导浏览次数"]
)

print("清洗后数据量:", len(df_clean), "行")


# ============================================================
# 第三部分：整体漏斗分析
# ============================================================

# 计算漏斗各层的总量（整体汇总）
funnel_data = {
    "曝光（浏览次数）":  df_clean["浏览次数"].sum(),
    "访问（访客数）":    df_clean["网页访客数"].sum(),
    "加购（加购人次）":  df_clean["加购物车人次"].sum(),
    "下单（下单人数）":  df_clean["下单人数"].sum(),
    "成交（成交人次）":  df_clean["成交人次"].sum(),
}

funnel_df = pd.DataFrame(list(funnel_data.items()), columns=["阶段", "人数"])

# 计算相邻步骤转化率
funnel_df["步骤转化率"] = funnel_df["人数"] / funnel_df["人数"].shift(1)
funnel_df["步骤转化率"] = funnel_df["步骤转化率"].map(
    lambda x: f"{x:.1%}" if pd.notna(x) else "—"
)

print("\n漏斗分析结果:")
print(funnel_df.to_string(index=False))


# ============================================================
# 第四部分：渠道归因分析
# ============================================================

# 计算各渠道总引导浏览次数
channel_sum = {
    "直通车":  df_clean["直通车引导浏览次数"].sum(),
    "淘宝客":  df_clean["淘宝客引导浏览次数"].sum(),
    "搜索":    df_clean["搜索引导浏览次数"].sum(),
    "聚划算":  df_clean["聚划算引导浏览次数"].sum(),
}
channel_df = pd.DataFrame(list(channel_sum.items()), columns=["渠道", "引导浏览次数"])
channel_df["占比"] = channel_df["引导浏览次数"] / channel_df["引导浏览次数"].sum()
channel_df = channel_df.sort_values("引导浏览次数", ascending=False)

print("\n渠道归因分析:")
print(channel_df.to_string(index=False))


# ============================================================
# 第五部分：商品分层（K-means聚类）
# ============================================================

# 选取用于聚类的特征：浏览→成交全链路指标
cluster_features = ["浏览次数", "成交金额", "成交件数（目标变量）", "加购物车人次", "下单人数"]
# 过滤掉这几列有缺失值的行
df_cluster = df_clean[cluster_features].dropna().copy()

# 标准化（K-means对量纲敏感，必须标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# 用肘部法则选K值（画图观察，一般K=3或4是拐点）
inertia = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# 正式聚类，取K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster["聚类标签"] = kmeans.fit_predict(X_scaled)

# 查看各群特征均值
cluster_profile = df_cluster.groupby("聚类标签")[cluster_features].mean().round(1)
print("\n商品聚类各群画像:")
print(cluster_profile)


# ============================================================
# 第六部分：可视化（生成4张图，保存为PNG）
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("电商商品销售数据分析报告", fontsize=16, fontweight="bold", y=0.98)

# ------ 图1：购买漏斗图 ------
ax1 = axes[0][0]
colors_funnel = ["#4C8BF5", "#5FA4F5", "#85BCE8", "#AACFE0", "#C8E0EC"]
bars = ax1.barh(
    funnel_df["阶段"][::-1],   # 反转让"曝光"在顶部
    funnel_df["人数"][::-1],
    color=colors_funnel,
    height=0.5
)
# 在条形上标注数值
for bar, val in zip(bars, funnel_df["人数"][::-1]):
    ax1.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
             f"{int(val):,}", va="center", fontsize=9)
ax1.set_title("购买漏斗（各阶段用户量）", fontsize=12)
ax1.set_xlabel("人数 / 次数")
ax1.spines[["top", "right"]].set_visible(False)

# ------ 图2：渠道归因饼图 ------
ax2 = axes[0][1]
colors_pie = ["#FF7F50", "#63B8FF", "#90EE90", "#DDA0DD"]
wedges, texts, autotexts = ax2.pie(
    channel_df["引导浏览次数"],
    labels=channel_df["渠道"],
    autopct="%1.1f%%",
    colors=colors_pie,
    startangle=140,
    pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
ax2.set_title("各渠道引导浏览次数占比", fontsize=12)

# ------ 图3：肘部法则选K值 ------
ax3 = axes[1][0]
ax3.plot(list(K_range), inertia, marker="o", color="#4C8BF5", linewidth=2)
ax3.set_title("K-means肘部法则（选最优聚类数）", fontsize=12)
ax3.set_xlabel("聚类数 K")
ax3.set_ylabel("簇内误差平方和（Inertia）")
ax3.spines[["top", "right"]].set_visible(False)

# ------ 图4：商品聚类散点图（浏览次数 vs 成交金额）------
ax4 = axes[1][1]
scatter_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
for label in sorted(df_cluster["聚类标签"].unique()):
    subset = df_cluster[df_cluster["聚类标签"] == label]
    ax4.scatter(
        subset["浏览次数"],
        subset["成交金额"],
        c=scatter_colors[label],
        label=f"群{label}",
        alpha=0.6,
        s=40
    )
ax4.set_title("商品聚类分布（浏览次数 vs 成交金额）", fontsize=12)
ax4.set_xlabel("浏览次数")
ax4.set_ylabel("成交金额（元）")
ax4.legend(fontsize=9)
ax4.spines[["top", "right"]].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("analysis_report.png", dpi=150, bbox_inches="tight")
print("\n图表已保存为 analysis_report.png")
plt.show()

print("\n✅ 分析完成！")