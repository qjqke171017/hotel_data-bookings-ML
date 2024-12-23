from HotelData import *

# 实例化并运行
hotel_data = HotelData(file_path='C:/Users/Lenovo/hotel_bookings.csv') #导入文件
hotel_data.load_data() #加载数据集

#数据清洗

# 删除重复的行
hotel_data.drop_duplicates(inplace=True)

#删除包含缺失值的行（如果缺失值较少）
hotel_data.dropna(subset=['children', 'country'], inplace=True)

#对于分类数据，填充为众数
hotel_data['agent'].fillna(hotel_data['agent'].mode()[0], inplace=True)

# 处理异常值， adr取值应该大于0， 房间均价不应太高，设置为1000以下。
hotel_data = hotel_data[(hotel_data['adr'] >= 0) & (hotel_data['adr'] <= 1000)]

# 添加新数据示例
hotel_data.add_new_data(hotel="Resort Hotel", is_canceled=0, lead_time=45, adults=2, children=0) #添加含有缺失值的新预订记录
#添加完整的预订记录
new_hotel_data2 = { #新记录字典
    'hotel': 'Resort Hotel',
    'is_canceled': 0,
    'lead_time': 200,
    'arrival_date_year': 2021,
    'arrival_date_month': 'July',
    'arrival_date_week_number': 27,
    'arrival_date_day_of_month': 4,
    'stays_in_weekend_nights': 2,
    'stays_in_week_nights': 5,
    'adults': 2,
    'children': 1,
    'babies': 0,
    'meal': 'BB',
    'country': 'USA',
    'market_segment': 'Direct',
    'distribution_channel': 'Direct',
    'is_repeated_guest': 0,
    'previous_cancellations': 0,
    'previous_bookings_not_canceled': 1,
    'reserved_room_type': 'A',
    'assigned_room_type': 'A',
    'booking_changes': 0,
    'deposit_type': 'No Deposit',
    'agent': 'NULL',
    'company': 'NULL',
    'days_in_waiting_list': 0,
    'customer_type': 'Transient',
    'adr': 100.0,
    'required_car_parking_spaces': 0,
    'total_of_special_requests': 0,
    'reservation_status': 'Check-Out',
    'reservation_status_date': '2021-07-04'
}
hotel_data.add_new_data(**new_hotel_data2) 

# 计算字段统计信息
hotel_data.field_statistics('lead_time')
hotel_data.field_statistics('adr')
hotel_data.field_statistics('hotel')
hotel_data.field_statistics('test') #检验错误值

# 数据可视化示例
hotel_data.visualize_data('lead_time')
hotel_data.visualize_data('adr')
hotel_data.visualize_data('reservation_status')
hotel_data.visualize_data('test')#检验错误值



