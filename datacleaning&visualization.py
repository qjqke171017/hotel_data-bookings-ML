from HotelData import *

# 实例化并运行
hotel_data = HotelData(file_path='C:/Users/Lenovo/hotel_bookings.csv') #导入文件
hotel_data.load_data() #加载数据集
hotel_data.handle_missing_values() #显示缺失值情况

hotel_data.data.info() #数据具体信息展示
hotel_data.data.describe() #数据集各个字段总览

#数据清洗

# 删除重复的行
hotel_data.data.drop_duplicates(inplace=True)

#删除包含缺失值的行（如果缺失值较少）
hotel_data.data.dropna(subset=['children', 'country'], inplace=True)

#对于分类数据，填充为众数
hotel_data.data['agent'].fillna(hotel_data.data['agent'].mode()[0], inplace=True)

# 处理异常值， adr取值应该大于0， 房间均价不应太高，设置为1000以下。
hotel_data.data = hotel_data.data[(hotel_data.data['adr'] >= 0) & (hotel_data.data['adr'] <= 1000)]

hotel_data.data.info()#数据清洗后数据集总览

#数据可视化
flow_data = hotel_data.data.groupby(['hotel', 'is_canceled'])['is_canceled'].count().reset_index(name='count')

# 构造源和目标列表
labels = ['Resort Hotel', 'City Hotel', 'Cancelled', 'Not Cancelled']  # 标签：酒店类型和取消情况
sources = [0, 0, 1, 1]  # 源节点：度假酒店和城市酒店
targets = [2, 3, 2, 3]  # 目标节点：取消和未取消
values = [flow_data.loc[(flow_data['hotel'] == 'Resort Hotel') & (flow_data['is_canceled'] == 1), 'count'].values[0],
          flow_data.loc[(flow_data['hotel'] == 'Resort Hotel') & (flow_data['is_canceled'] == 0), 'count'].values[0],
          flow_data.loc[(flow_data['hotel'] == 'City Hotel') & (flow_data['is_canceled'] == 1), 'count'].values[0],
          flow_data.loc[(flow_data['hotel'] == 'City Hotel') & (flow_data['is_canceled'] == 0), 'count'].values[0]]  # 流动量

# 绘制桑基图
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
))

# 设置图表标题
fig.update_layout(title_text="Sankey Diagram: Booking Flow by Hotel Type and Cancellation Status", font_size=10)

# 显示图表
fig.show()

#数据可视化2

# 确保月份按正确的顺序显示
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
hotel_data.data['arrival_date_month'] = pd.Categorical(hotel_data.data['arrival_date_month'], categories=month_order, ordered=True)

# 计算每个月份的取消率
cancel_rate = hotel_data.data.groupby(['arrival_date_month', 'hotel'])['is_canceled'].mean().reset_index()

# 计算每个月份的入住人数
hotel_data.data['total_guests'] = hotel_data.data['adults'] + hotel_data.data['children'] + hotel_data.data['babies']
guests_count = hotel_data.data.groupby(['arrival_date_month', 'hotel'])['total_guests'].sum().reset_index()

# 创建画布
fig, ax1 = plt.subplots(figsize=(14, 8))

sns.barplot(data=guests_count, x='arrival_date_month', y='total_guests', hue='hotel', ax=ax1, palette='pastel', ci=None)
ax1.set_ylabel('Total Guests', fontsize=12, color='green')
ax1.set_yticks(range(0, int(guests_count['total_guests'].max()) + 1, 500))  # 设置入住人数刻度
ax1.tick_params(axis='y', labelcolor='green')

# 创建第二个 Y 轴，用于展示入住人数
ax2 = ax1.twinx()
sns.lineplot(data=cancel_rate, x='arrival_date_month', y='is_canceled', hue='hotel', marker='o', ax=ax2, palette='viridis')
ax2.set_ylabel('Cancellation Rate (%)', fontsize=12, color='blue')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylim(0, 1)  # 设置取消率的范围在 0 到 1 之间
ax2.set_yticks([i * 0.2 for i in range(6)])  # 设置取消率百分比刻度
ax2.tick_params(axis='y', labelcolor='blue')

# 设置标题
plt.title('Cancellation Rate and Guest Count by Month for Resort and City Hotels', fontsize=16)

# 优化布局
plt.tight_layout()

# 显示图表
plt.show()

