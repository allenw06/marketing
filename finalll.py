#!/usr/bin/env python
# coding: utf-8

# In[198]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import seaborn as sns


# # Data Structure 

# In[199]:


df=pd.read_pickle(r'subscribers')


# In[200]:


df.head()


# In[170]:


print(df['attribution_technical'])


# In[108]:


df.shape


# In[171]:


df.isnull().sum()


# In[42]:


#sub['creation_until_cancel_days'].value_counts(dropna=False)


# In[172]:


def get_data(df,column_name,fill_value,with_null=True):
    if with_null is True:
        if df[column_name].isnull().sum()>0:
            df[column_name] = df[column_name].fillna(fill_value)
    data = df.groupby(['paid_TF',column_name])
    results = {}
    for name,group in data:
        results[name] = len(group)
    return results


# In[173]:


def trans_data(data):
    results = {}
    keys = set()
    for key in data:
        first = key[0]
        second = key[1]
        keys.add(second)
        if first not in results:
            results[first] = {}
        results[first][second] = data[key]
    # 补齐两边的key
    for key in results:
        for inner_key in keys:
            if inner_key not in results[key]:
                results[key][inner_key] = 0
    return results


# In[174]:


def remove_null(data):
    '''
    将null值去掉，不在结果中显示
    '''
    results = {}
    for key in data:
        if key not in results:
            results[key] = {}
        for inner_key in data[key]:
            if inner_key == "null":
                continue
            results[key][inner_key] = data[key][inner_key]
    return results


# In[175]:


def percent_data(data):
    for key in data:
        sum_values = sum(data[key].values())
        for inner_key in data[key]:
#             print(key,inner_key,data[key][inner_key],sum(data[key].values()))
            data[key][inner_key] = round(data[key][inner_key] / sum_values* 100,2) 
    return data


# In[176]:


def to_percent(temp, position):
    return '%1.0f'%(temp) + '%'


# In[177]:


df['paid_TF'].value_counts(dropna=False)


# In[292]:


df['age'].value_counts(dropna=False)


# In[179]:


df['age']=pd.to_numeric(df.age, errors='coerce')


# In[291]:


def trans_age(data):
    age = data['age']
    if age is None :
        return "null"
    if age >= 70:
        return "Over 70"
    elif age >= 50:
        return "50-70"
    elif age >= 35:
        return "35-50"
    elif age >= 25:
        return "25-35"
    elif age >= 18:
        return "18-25"
    elif age >= 0:
        return "below_18"
    else:
        return "null"

    return age
df['age'] = df.apply(trans_age,axis=1)


# In[67]:


#df['age'].value_counts(dropna=False)


# In[27]:


# 年龄的数据
data = get_data(df,'age','null')
# 转换数据
data = trans_data(data)
# 去掉null值
data = remove_null(data)
# 转化为百分比
data = percent_data(data)
data


# In[10]:


df['package_type'].value_counts(dropna=False)


# In[11]:


data = get_data(df,'package_type','null')
# 转换数据
data = trans_data(data)
# 去掉null值
#data = remove_null(data)
# 转化为百分比
data = percent_data(data)
data


# In[12]:


data = get_data(df,'preferred_genre','null')
# 转换数据
data = trans_data(data)
# 去掉null值
#data = remove_null(data)
# 转化为百分比
data = percent_data(data)
data


# In[13]:


#intended_use
data = get_data(df,'intended_use','null')
# 转换数据
data = trans_data(data)
# 去掉null值
#data = remove_null(data)
# 转化为百分比
data = percent_data(data)
data


# In[33]:


data = get_data(df,'male_TF','null')
# 转换数据
data = trans_data(data)
# 去掉null值
#data = remove_null(data)
# 转化为百分比
data = percent_data(data)
data


# In[53]:


data = get_data(df,'plan_type','null')
# 转换数据
data = trans_data(data)
# 去掉null值
#data = remove_null(data)
# 转化为百分比
#data = percent_data(data)
data


# In[183]:


def trans_hour(data):
    hour = data['weekly_consumption_hour']
    if hour is None :
        return "null"
    if hour >= 60:
        return "Over 60"
    if hour >= 40:
        return "40-60"
    elif hour >= 20:
        return "20-40"
    elif hour >= 10:
        return "10-20"
    elif hour >= 0:
        return "0-10"
    
    else:
        return "null"

    return hour
#df['weekly_consumption_hour'] = df.apply(trans_hour,axis=1)


# In[184]:


df['weekly_consumption_hour'].value_counts(dropna=False)


# In[63]:


df['weekly_consumption_hour']=pd.to_numeric(df.age, errors='coerce')


# In[66]:


# 年龄的数据
data = get_data(df,'weekly_consumption_hour','null')
# 转换数据
data = trans_data(data)
# 去掉null值
data = remove_null(data)
# 转化为百分比
data = percent_data(data)
data


# In[53]:


df['cancel_before_trial_end'].value_counts(dropna=False)


# In[56]:


# 年龄的数据
data = get_data(df,'cancel_before_trial_end','null')
# 转换数据
data = trans_data(data)
# 去掉null值
#data = remove_null(data)
# 转化为百分比
#data = percent_data(data)
data


# In[61]:



# 年龄的数据
data = get_data(df,'attribution_technical','null')
# 转换数据
data = trans_data(data)
# 去掉null值
data = remove_null(data)
# 转化为百分比
data = percent_data(data)
data  


# In[122]:


EGG=pd.read_pickle(r'engagement')


# In[123]:


EGG.head()


# In[152]:


EGG.isnull().sum()


# ## ABTEST

# In[227]:


CSR=pd.read_pickle(r'customer_service_reps')


# In[228]:


CSR.head()


# In[229]:


#sort
sortCSR = CSR.sort_values(by=['account_creation_date'])
sortCSR =sortCSR[['subid','current_sub_TF']]


# In[230]:


sortCSR



# In[231]:


#churn
churn = sortCSR.drop_duplicates(keep = 'last')
churn


# In[232]:


#merge
Merged = df.merge(churn, how = 'left',left_on='subid', right_on='subid')
CH_SUB = Merged[['current_sub_TF']]
Merged[['current_sub_TF']] = CH_SUB.fillna(True)


# In[247]:


Merged


# In[234]:


Merged[['num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','revenue_net','join_fee']] = Merged[['num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','revenue_net','join_fee']].fillna(0)
Merged[['package_type','preferred_genre','intended_use','male_TF','attribution_survey','payment_type']] = Merged[['package_type','preferred_genre','intended_use','male_TF','attribution_survey','payment_type']].fillna('Unknown')


# In[235]:


drop = ['op_sys','num_ideal_streaming_services' ,'creation_until_cancel_days','payment_type','country', 'account_creation_date', 'trial_end_date', 'language'] 
for i in drop:
    Merged = Merged[Merged.columns.drop(i)]


# In[236]:


uae14 = Merged.loc[Merged['plan_type'] == 'base_uae_14_day_trial']
no_trial = Merged.loc[Merged['plan_type'] == 'low_uae_no_trial']


# In[237]:


uae14= uae14[['current_sub_TF','plan_type']]
no_trial= no_trial[['current_sub_TF','plan_type']]


# In[238]:


ABT = pd.concat([uae14 , no_trial], axis=0)


# In[239]:


ABT= pd.get_dummies(ABT, prefix=['plan_type'],drop_first= True)


# In[240]:


ABT


# In[ ]:


#import HW1


# In[241]:


import HW1 as ABTest


# In[99]:


#pip install -U imbalanced-learn


# In[242]:


import imblearn
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='majority')

X_under, y_under = undersample.fit_resample(ABT[['plan_type_low_uae_no_trial']],ABT[['current_sub_TF']])


# In[243]:


ABT


# In[244]:


A =  ABT.loc[ABT['plan_type_low_uae_no_trial'] == 0]
B =  ABT.loc[ABT['plan_type_low_uae_no_trial'] == 1] 


# In[245]:



A1 = list(A['current_sub_TF'])
B1= list(B['current_sub_TF'])


# In[246]:


import scipy
norm = scipy.stats.norm()
ABTest.t_test(A_list, B_list,0.95)


# ### CLUSTERING

# In[293]:


dfdummy = df[[ 'preferred_genre','age','weekly_consumption_hour','intended_use','revenue_net']]

dfdummy = pd.get_dummies(dfdummy)

dfdummy
df1 = pd.merge(df[['subid']], dfdummy, left_index=True, right_index = True, how='left')

df1


# In[294]:


List=['num_videos_completed','num_series_started',"app_opens"]
df2 = pd.pivot_table(EGG, values=List, index='subid', aggfunc=np.mean)
df2.reset_index(drop=False, inplace=True)
df2


# In[295]:


dfall = pd.merge(df1, df2, on= 'subid', how='left')

dfall


# In[296]:


dfall.dropna(axis=0, inplace=True)

dfall.set_index('subid',inplace=True)

dfall.shape


# In[297]:


dfall.columns


# In[298]:


#optimal K
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def fitting(df):
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(df)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    return 


# In[299]:


fitting(dfall)


# In[301]:


kmeans = KMeans(n_clusters = 4, random_state=0).fit(dfall)
a = list(kmeans.cluster_centers_)

seg_result = pd.DataFrame(a, columns=dfall.columns)

seg_result


# In[303]:



seg_result.to_excel("output2.xlsx")  


# ### CHURN MODEL

# In[248]:


Merged = pd.get_dummies(Merged, prefix=['package_type', 'preferred_genre', 'intended_use','male_TF',  'attribution_technical',  'plan_type'], columns=['package_type', 'preferred_genre', 'intended_use','male_TF',  'attribution_technical', 'plan_type'])


# In[257]:


Merged


# In[256]:


Merged = Merged[Merged.columns.drop('num_weekly_services_utilized')]


# In[252]:


drop1 = ['attribution_survey','monthly_price' ,'months_per_bill_period','cancel_before_trial_end','discount_price'] 
for i in drop1:
    Merged = Merged[Merged.columns.drop(i)]


# In[254]:


Merged.columns


# In[255]:


drop2 = ['join_fee','initial_credit_card_declined'] 
for i in drop2:
    Merged = Merged[Merged.columns.drop(i)]


# In[265]:


Merged.dropna(axis=0, inplace=True)


# In[268]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
def contacted_data_test(model):
    model.fit(X_train, y_train)
    #print("The coefficient of each independent variable is {}".format(model.coef_))
    print("The Mean test cross-validation score (5-folds) for contacted dataset: {}".format(np.mean(cross_val_score(model, X_train, y_train, cv=5))))

    prediction = model.predict(X_test)
    prediction1= model.predict_proba(X_test)[:,1]
    test_score = accuracy_score(y_test, prediction)
    print("The accuracy score for contacted dataset's test set: {}".format(test_score))

    CM = confusion_matrix(y_test,prediction)
    print("The confusion matrix for contacted dataset's test set: {}".format(CM))
    
    print('AUC:{}'.format(metrics.roc_auc_score(y_test,prediction1)))


# In[267]:


from sklearn.model_selection import train_test_split
X = Merged[Merged.columns.drop('current_sub_TF')]
y = Merged['current_sub_TF'].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)


# In[269]:


# built decision tree model with max_depth= [3,4,5,6,7], using cross validation to evaluate the models for contacted dataset

from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier(random_state=13)

cv_scores_4 = cross_val_score(dct, X_train, y_train, cv=3)  # parameter cv is number of folds you want to split

print('Cross-validation scores for contacted dataset(3-fold):', cv_scores_4)
print('Mean cross-validation score for contacted dataset(3-fold): {:.3f}'
     .format(np.mean(cv_scores_4)))

param_range = [3,4,5,6,7]
train_scores_2, test_scores_2 = validation_curve(dct, X_train, y_train,
                                            param_name='max_depth',
                                            param_range=param_range, cv=5)

print('Mean train cross-validation score (5-folds) for contacted dataset with max_depth = [3,4,5,6,7]: {}'.format(np.mean(train_scores_2, axis=1)))
print('Mean test cross-validation score (5-folds) for contacted dataset with max_depth = [3,4,5,6,7]: {}'.format(np.mean(test_scores_2, axis=1)))


# In[272]:


dct = DecisionTreeClassifier(random_state=13, max_depth=7)
contacted_data_test(dct)


# In[277]:




dct.fit(X_train,y_train)
y_pred = dct.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))


# In[278]:


def plot_feature_importances(clf, feature_names):
    c_features = len(feature_names)
    feat_importances = pd.Series(clf.feature_importances_, index=feature_names)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    
plt.figure(figsize=(10,4), dpi=80)
plot_feature_importances(dct, X_train.columns)
plt.show()


# CLV

# In[285]:


y_pred = dct.predict(X)
y_pred = y_pred.astype(int)
REV = np.array(X['revenue_net'])
PredictRev = np.dot(y_pred,REV)
CLV=PredictRev/len(y_pred)
print("CLV is" + str(CLV) + " And expected revenue is " +str(PredictRev))

