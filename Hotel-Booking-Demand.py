#!/usr/bin/env python
# coding: utf-8

# # Hotel Booking Demand 酒店预定需求

# In[1]:


import pandas as pd

df = pd.read_csv("hotel_bookings.csv")
df.head()


# ## 1. 基本情况：城市酒店和假日酒店预订需求和入住率比较

# 统计 hotel 列的值域情况，结论是城市酒店的预订信息比假日酒店的预订信息更多。

# In[3]:


df.hotel.value_counts()


# 统计缺失值情况，children、country 缺失较少，agent、company 缺失较多，并进行缺失值处理。

# In[5]:


df.isnull().sum()


# In[7]:


df_cln = df.fillna({"children": 0,"country": "Unknown", "agent": 0, "company": 0})
df_cln["meal"].replace("Undefined", "SC", inplace=True)
zero_guests = list(df_cln.loc[df_cln["adults"]
                   + df_cln["children"]
                   + df_cln["babies"]==0].index)
df_cln.drop(df_cln.index[zero_guests], inplace=True)
df = df_cln

df.isnull().sum()


# 比较酒店预定需求以及入住率（入住率=入住总数/预定总数），可以看到就酒店预定来说城市酒店比假日酒店更受欢迎，人们更喜欢预定城市酒店。

# In[8]:


import seaborn as sns

sns.countplot(df.hotel)


# In[9]:


city_count_book = df.hotel.value_counts()['City Hotel']
resort_count_book = df.hotel.value_counts()['Resort Hotel']
resort_check_in = df[df['hotel'] == 'Resort Hotel'].is_canceled.value_counts()[0]
city_check_in = df[df['hotel'] == 'City Hotel'].is_canceled.value_counts()[0]

print('城市酒店入住率：', city_check_in/city_count_book)
print('假日酒店入住率：', resort_check_in/resort_count_book)


# ## 2. 用户行为：提前预订时间、入住时长、预订间隔、餐食预订情况

# ### 2.1 提前预订时间

# 统计 lead_time 的数值属性，可以看出顾客平均提前预定时间为 104 天左右，预定最久的天数为 737 天，将近两年多，但大部分顾客都是当天预定当天入住。

# In[14]:


import numpy as np

time_list = list(df['lead_time'])
print('均值:', np.mean(time_list))
print("中位数：",np.median(time_list))
print("最小值：",min(time_list))
print("最大值：",max(time_list))
print("四分位数:",np.percentile(time_list, (25, 50, 75), interpolation='midpoint'))
counts = np.bincount(time_list)
print("众数：",np.argmax(counts))

sns.countplot(df.lead_time)


# ### 2.2 入住时长

# 统计 stay_time 的数值属性，可以看出平均入住晚数为 3 晚左右，最大入住晚数为 69 天，两个多月，其中大部分顾客入住 2 晚。

# In[15]:


df['stay_time'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
stay_list = list(df['stay_time'])
print('均值:', np.mean(stay_list))
print("中位数：",np.median(stay_list))
print("最小值：",min(stay_list))
print("最大值：",max(stay_list))
print("四分位数:",np.percentile(stay_list, (25, 50, 75), interpolation='midpoint'))
counts = np.bincount(stay_list)
print("众数：",np.argmax(counts))

sns.countplot(df.stay_time)


# ### 2.3 餐食预订情况

# 统计 meal 标签属性，可以看出大多数人会在酒店中订餐，其中大部分人预定了 BB 这个套餐类型，很少人订 FB 这个套餐类型。当然不是所有人都会在酒店订餐，少部分人不需要订餐服务

# In[19]:


sns.countplot(df.meal)


# ## 3. 一年中最佳预订酒店时间

# 统计各时间段酒店入住情况，可以看出最佳预定酒店的时间应为每年的 1、2 月和 11、12 月，这几个时间段的酒店的入住人数少且价格较低，是最佳的酒店预定入住时间。

# In[20]:


df.groupby(['arrival_date_month','arrival_date_year'])['children'].sum().plot.bar(figsize=(15,5))


# ## 4. 利用 Logistic 预测酒店预订

# 计算矩阵关联矩阵，并按相关系数排序，并使用逻辑回归预测。

# In[21]:


cancel_corr = df.corr()["is_canceled"]
cancel_corr.abs().sort_values(ascending=False)[1:]


# In[22]:


df.groupby("is_canceled")["reservation_status"].value_counts()


# 其它属性按相关系数权重作为特征标签，选择酒店预定情况属性 is_canceled 作为预测标签，使用 Logic Regression 模型训练、预测，结果的 acc 指标达到 0.79 左右！

# In[24]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score

# 使用的 ic_canceled 属性作为预测标签，其它属性按相关系数权重作为特征标签
num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
                "stays_in_weekend_nights","stays_in_week_nights","adults","children",
                "babies","is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled","agent","company",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]

cat_features = ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]

features = num_features + cat_features
X = df.drop(["is_canceled"], axis=1)[features]
y = df["is_canceled"]

# 使用 Logic Regression 模型训练、预测
num_transformer = SimpleImputer(strategy="constant")

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                               ("cat", cat_transformer, cat_features)])
base_models = [("LR_model", LogisticRegression(random_state=42,n_jobs=-1))]

# 划分训练集与测试集
kfolds = 4
split = KFold(n_splits=kfolds, shuffle=True, random_state=42)

for name, model in base_models:
    model_steps = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
    
    # 使用交叉熵损失函数迭代优化模型
    cv_results = cross_val_score(model_steps, 
                                 X, y, 
                                 cv=split,
                                 scoring="accuracy",
                                 n_jobs=-1)

    min_score = round(min(cv_results), 4)
    max_score = round(max(cv_results), 4)
    mean_score = round(np.mean(cv_results), 4)
    std_dev = round(np.std(cv_results), 4)
    print(f"{name} cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")


# In[ ]:




