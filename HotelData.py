import pandas as pd #加载pandas包
import numpy as np #加载numpy包
import matplotlib.pyplot as plt #加载matplot包
from sklearn.model_selection import train_test_split, GridSearchCV #加载sklearn包
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import tensorflow as tf #加载tensorflow包以深度学习
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping #早停
import warnings#取消警告
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px#可视化

class HotelData: #构造酒店类
    def __init__(self, file_path):
        self.file_path = file_path #初始化文件路径
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None #初始化训练集预测集
        self.rf_accuracy = None
        self.nn_accuracy = None
        self.rf_model = None
        self.nn_model = None
        self.rf_fpr = None
        self.rf_tpr = None
        self.nn_fpr = None
        self.nn_tpr = None  #初始化预测准确率

    # 1. 上传数据集
    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
            print(self.data.head())
        except Exception as e: #如果数据没有成功上传则抛出错误
            print(f"Error loading data: {e}")

    # 2. 简易的数据清洗，处理缺失值
    def handle_missing_values(self):
        print("Missing values before handling:\n", self.data.isnull().sum())
        self.data.fillna(self.data.median(), inplace=True)
        print("Missing values handled.")

    # 3. 添加新的酒店数据
    def add_new_data(self, **new_data): 
        #接收新的酒店预订记录。提供已知的信息即可，对于未提供的信息使用NA填充。但是不可以出现原数据集中没有的字段，否则抛出错误
        try:
            missing_columns = [col for col in self.data.columns if col not in new_data] 
            if missing_columns:
                print(f"Missing values for fields: {missing_columns}. These will be filled with NA.") #提示缺少的信息使用NA填充
            
            complete_data = {col: new_data.get(col, np.nan) for col in self.data.columns}#将新加入的信息汇总到原数据集当中
            new_data_df = pd.DataFrame([complete_data])
            self.data = pd.concat([self.data, new_data_df], ignore_index=True)
            print("New data added successfully.") #提示新预订记录成功添加
            print(new_data_df)
        except Exception as e: #抛出错误：不合规范的格式或者出现了未出现过的字段
            print(f"Error adding new data: {e}")
            
    
    # 4.研究字段信息
    def field_statistics(self, field_name):
        try:
            if field_name not in self.data.columns: #如果输入不是数据集中的字段，则抛出错误
                raise ValueError(f"Field '{field_name}' does not exist in the dataset.")

            # 如果是数值型的字段，则输出该字段的描述统计信息，包括最大值，最小值，平均数，众数，分位数，峰度偏度
            #存储在字典中
            if np.issubdtype(self.data[field_name].dtype, np.number):
                stats = {
                    'mean': self.data[field_name].mean(),
                    'max': self.data[field_name].max(),
                    'min': self.data[field_name].min(),
                    'mode': self.data[field_name].mode()[0],
                    '25th_percentile': self.data[field_name].quantile(0.25),
                    'median': self.data[field_name].median(),
                    '75th_percentile': self.data[field_name].quantile(0.75),
                    'kurtosis': self.data[field_name].kurtosis(),
                    'skewness': self.data[field_name].skew()
                }
                print(f"Statistics for numeric field '{field_name}': {stats}")
                return stats

            #如果是非数值型的字段，则当做因子型统计不同字符的出现次数。
            else:
                value_counts = self.data[field_name].value_counts()
                print(f"Value counts for non-numeric field '{field_name}':\n{value_counts}")
                return value_counts

        except Exception as e:
            print(f"Error calculating statistics for field '{field_name}': {e}") #若不是字段，抛出错误

    # 5.简易数据可视化
    def visualize_data(self, field_name):
        try:
            if field_name not in self.data.columns: #如果输入的参数不是字段名称，则报错
                raise ValueError(f"Field '{field_name}' does not exist in the dataset.")

            if np.issubdtype(self.data[field_name].dtype, np.number): #如果是数值型变量，则绘制直方图。
                plt.hist(self.data[field_name].dropna(), bins=30, edgecolor='k', alpha=0.7)
                plt.title(f"Histogram of {field_name}")
                plt.xlabel(field_name)
                plt.ylabel("Frequency")
                plt.show()
            else: #如果是非数值型变量，则绘制条形图。
                self.data[field_name].value_counts().plot(kind='bar', color='skyblue', edgecolor='k')
                plt.title(f"Bar Chart of {field_name}")
                plt.xlabel(field_name)
                plt.ylabel("Frequency")
                plt.show()
        except Exception as e: #不是字段名称，则报错
            print(f"Error visualizing data for field '{field_name}': {e}")

    # 6. 数据划分，为机器学习模型的训练做准备
    def prepare_data(self):
        print("Preparing data for machine learning...") #提示划分数据中
        df = self.data.copy() #将数据集进行复制
        df = pd.get_dummies(df, drop_first=True)  #onehot编码
        df.fillna(0, inplace=True) 
        self.X = df.drop('is_canceled', axis=1) #按照取消率标签分类
        self.y = df['is_canceled']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        #设置好随机状态随机划分训练集与测试机。80%作为训练集，20%作为测试机。
        print("Data prepared successfully.") #提示数据划分完成

     # 7. 随机森林方法
    def random_forest_study(self, n_estimators=100, max_depth=None, min_samples_split=2):
        print("Training Random Forest...")
        try:
            # 初始化随机森林模型
            self.rf_model = RandomForestClassifier(n_estimators=n_estimators, 
                                                   max_depth=max_depth, 
                                                   min_samples_split=min_samples_split, 
                                                   random_state=42)
            # 训练模型
            self.rf_model.fit(self.X_train, self.y_train)

            # 在测试集上预测
            rf_pred = self.rf_model.predict(self.X_test)
            self.rf_accuracy = accuracy_score(self.y_test, rf_pred)
            print("Random Forest Accuracy:", self.rf_accuracy)
            print(classification_report(self.y_test, rf_pred))

            # 计算 ROC 曲线
            rf_probs = self.rf_model.predict_proba(self.X_test)[:, 1]
            self.rf_fpr, self.rf_tpr, _ = roc_curve(self.y_test, rf_probs)
        
            # 输出特征重要性
            importances = self.rf_model.feature_importances_
            feature_names = self.X_train.columns
            important_features = sorted(zip(importances, feature_names), reverse=True)
            print("Top 10 important features:")
            print(important_features[:10])
        
            return self.rf_accuracy
        except ValueError as e:
            print(f"Error during Random Forest training: {e}")
            print("Check if there are invalid values in the dataset.")


    # 8. 神经网络方法
    def mlp_study(self, epochs=20, dropout_rate=0.2, learning_rate=0.0003, patience=5): #神经网络的参数默认值
        print("Training Neural Network...") #提示神经网络模型训练中
        self.nn_model = Sequential([ #构造神经网络模型层次
            Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)), #使用relu激活函数
            Dropout(dropout_rate), #dropuout百分之二十的神经元，预防过拟合发生
            Dense(64, activation='relu'), #使用relu激活函数全连接
            Dense(1, activation='sigmoid') #使用sigmoid激活函数进行最后的预测
        ])
        optimizer = Adam(learning_rate=learning_rate) #优化器使用adam
        self.nn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) #对于分类问题采用交叉熵损失函数

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True) #早停机制
        history = self.nn_model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])
        #对于过去历史训练的追溯
        nn_loss, self.nn_accuracy = self.nn_model.evaluate(self.X_test, self.y_test)
        print("Neural Network Accuracy:", self.nn_accuracy) #打印神经网络的预测准确率

        # 绘制ROC曲线
        nn_probs = self.nn_model.predict(self.X_test).ravel()
        self.nn_fpr, self.nn_tpr, _ = roc_curve(self.y_test, nn_probs)

        # 绘制训练集和测试机准确率随epoch的变化
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Neural Network Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        return self.nn_accuracy
    
    # 7. 比较两种方法 
    # 另一种训练方式，输入为接受两个参数字典，先后训练两个模型。
    def compare_methods(self, rf_params={}, mlp_params={}):
        print("Comparing Random Forest and Neural Network...")
        self.rf_accuracy = self.random_forest_study(**rf_params)
        self.nn_accuracy = self.mlp_study(**mlp_params)
        print(f"Random Forest Accuracy: {self.rf_accuracy:.4f}, Neural Network Accuracy: {self.nn_accuracy:.4f}")

    # 9. 总结
    def summarize(self):
        print("Summarizing model performance...") #总结此次机器学习任务

        #绘制直方图展示两个模型准确率
        accuracies = {'Random Forest': self.rf_accuracy, 'Neural Network': self.nn_accuracy}
        plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'orange'])
        plt.title("Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.show()

        # 绘制两个模型的ROC曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.rf_fpr, self.rf_tpr, label=f"Random Forest (AUC = {auc(self.rf_fpr, self.rf_tpr):.2f})", color='blue')
        plt.plot(self.nn_fpr, self.nn_tpr, label=f"Neural Network (AUC = {auc(self.nn_fpr, self.nn_tpr):.2f})", color='orange')
        plt.plot([0, 1], [0, 1], 'k--', label="Chance")
        plt.title("ROC Curve Comparison")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.show()

        # 总结哪个模型表现更好
        if self.rf_accuracy > self.nn_accuracy:
            result_message = f"Random Forest performed better with an accuracy of {self.rf_accuracy:.4f} compared to Neural Network's {self.nn_accuracy:.4f}."
        elif self.rf_accuracy < self.nn_accuracy:
            result_message = f"Neural Network performed better with an accuracy of {self.nn_accuracy:.4f} compared to Random Forest's {self.rf_accuracy:.4f}."
        else:
            result_message = f"Both models performed equally well with an accuracy of {self.rf_accuracy:.4f}."

        print(result_message)
        return result_message
    
    #10.对新的预定记录做出退订概率的预测
    def predict_cancellation_probability(self, new_record): #输出要求是一条完整的预订记录，不可以有缺失值。
        try:
            if any(pd.isnull(value) for value in new_record.values()):
                raise ValueError("The input record contains missing values. Please provide complete data.") #有预测值导致抛出错误
            
            if self.rf_model is None:
                raise ValueError("Random Forest model has not been trained. Please run random_forest_study first.")
            if self.nn_model is None:
                raise ValueError("Neural Network model has not been trained. Please run mlp_study first.")


            # 加入数据集中
            record_df = pd.DataFrame([new_record])
            record_df = pd.get_dummies(record_df, drop_first=True)
            missing_cols = set(self.X_train.columns) - set(record_df.columns)
            for col in missing_cols:
                record_df[col] = 0

            record_df = record_df[self.X_train.columns]

            #预测取消率
            rf_prob = self.rf_model.predict_proba(record_df)[:, 1][0]
            nn_prob = self.nn_model.predict(record_df).ravel()[0]

            print(f"Random Forest model predicts cancellation probability: {rf_prob:.4f}")
            print(f"Neural Network model predicts cancellation probability: {nn_prob:.4f}")

        except Exception as e:
            print(f"Error predicting cancellation probability: {e}")



