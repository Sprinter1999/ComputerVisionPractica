
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
PATH = "kaggle/datasets/" #where you put the files
df_train = pd.read_csv(f'{PATH}train.csv', index_col='Id')
df_test = pd.read_csv(f'{PATH}test.csv', index_col='Id')
'''
导入数据集
'''


# In[2]:


target = df_train['SalePrice'] #target variable
df_train = df_train.drop('SalePrice', axis=1)
df_train['training_set'] = True
df_test['training_set'] = False
df_full = pd.concat([df_train, df_test])
df_full = df_full.interpolate()
df_full = pd.get_dummies(df_full)
df_train = df_full[df_full['training_set']==True]
df_train = df_train.drop('training_set', axis=1)
df_test = df_full[df_full['training_set']==False]
df_test = df_test.drop('training_set', axis=1)
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(df_train, target)
preds = rf.predict(df_test)
'''
随机森林训练法
'''


# In[3]:


my_submission = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds})
my_submission.to_csv(f'{PATH}submission.csv', index=False)
'''
写入文本，准备提交
'''

