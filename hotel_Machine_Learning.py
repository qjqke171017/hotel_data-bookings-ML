from HotelData import *

# 实例化并运行
hotel_data = HotelData(file_path='C:/Users/Lenovo/hotel_bookings.csv') #导入文件
hotel_data.load_data() #加载数据集

#数据清洗

# 删除重复的行
hotel_data.data.drop_duplicates(inplace=True)

#删除包含缺失值的行（如果缺失值较少）
hotel_data.data.dropna(subset=['children', 'country'], inplace=True)

#对于分类数据，填充为众数
hotel_data.data['agent'].fillna(hotel_data.data['agent'].mode()[0], inplace=True)

# 处理异常值， adr取值应该大于0， 房间均价不应太高，设置为1000以下。
hotel_data.data = hotel_data.data[(hotel_data.data['adr'] >= 0) & (hotel_data.data['adr'] <= 1000)]

# 数据集划分准备
hotel_data.prepare_data()

# 随机森林与神经网络的参数控制示例，可以手动进行修改调参
rf_params = {
    'n_estimators':50,  # 决策树的数量
    'max_depth': 20,       # 每棵树的最大深度
    'min_samples_split': 2# 内部节点再划分所需的最小样本数
}

mlp_params = {
    'epochs': 20 ,        # 训练的最大轮次
    'dropout_rate': 0.2 ,
    'learning_rate': 0.001, # 学习率
    'patience': 5          # 提前停止的轮次
}

warnings.filterwarnings("ignore")
#按照默认值训练
hotel_data.random_forest_study(n_estimators = 50, max_depth = 20, min_samples_split = 2)
hotel_data.mlp_study(epochs = 15, dropout_rate = 0.2, learning_rate = 0.001, patience = 5) 

hotel_data.mlp_study(epochs = 15, dropout_rate = 0.2, learning_rate = 0.0005, patience = 5)

hotel_data.mlp_study(epochs = 15, dropout_rate = 0.2, learning_rate = 0.0001, patience = 5) #测试

warnings.filterwarnings("ignore")
hotel_data.random_forest_study(n_estimators = 75, max_depth = 25, min_samples_split = 2) #调整参数再次尝试

warnings.filterwarnings("ignore")
hotel_data.random_forest_study(n_estimators = 100, max_depth = 50, min_samples_split = 2) #更大的参数

#另一种方法——使用参数字典的方式训练
warnings.filterwarnings("ignore")
hotel_data.compare_methods(rf_params=rf_params, mlp_params=mlp_params)

# 总结
hotel_data.summarize()

#构造一条很可能退订的预订信息
high_cancellation_record = {
    'hotel': 'Resort Hotel',
    'is_repeated_guest': 0,          # 非回头客
    'lead_time': 365,               # 提前时间极长
    'arrival_date_year': 2024,
    'arrival_date_month': 'December',
    'arrival_date_week_number': 52,
    'arrival_date_day_of_month': 31,
    'stays_in_weekend_nights': 0,   # 周末天数少
    'stays_in_week_nights': 1,      # 工作日天数少
    'adults': 1,                    # 仅1个成人
    'children': 0,                  # 无儿童
    'babies': 0,                    # 无婴儿
    'meal': 'BB',                   # 早餐计划
    'country': 'PRT',               # 预订来源国家
    'market_segment': 'Online TA',  # 在线旅行社预订
    'distribution_channel': 'TA/TO',
    'is_repeated_guest': 0,
    'previous_cancellations': 3,    # 高取消历史记录
    'previous_bookings_not_canceled': 0,
    'reserved_room_type': 'A',
    'assigned_room_type': 'A',
    'booking_changes': 0,
    'deposit_type': 'Non Refund',   # 非退款政策
    'agent': 1,
    'days_in_waiting_list': 100,    # 较长的等待时间
    'customer_type': 'Transient',
    'adr': 200,                     # 高房价
    'required_car_parking_spaces': 0,
    'total_of_special_requests': 0  # 无特殊需求
}

# 使用训练好的模型预测退订概率
hotel_data.predict_cancellation_probability(high_cancellation_record)