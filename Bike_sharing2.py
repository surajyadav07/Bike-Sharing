# -*- coding: utf-8 -*-
"""
Created on Mon May  4 08:07:34 2020

@author: -
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

from datetime import datetime
#import missingno as msno
from scipy import stats

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv('sampleSubmission.csv')

df['date'] = df.datetime.apply(lambda x:x.split()[0])
df["hour"] = df.datetime.apply(lambda x : x.split()[1].split(":")[0])
df['weekday'] = df.date.apply(lambda x:calendar.day_name[datetime.strptime(x,"%Y-%m-%d").weekday()])
df['month'] = df.date.apply(lambda x:calendar.month_name[datetime.strptime(x,'%Y-%m-%d').month])


df["season"] = df.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
df["weather"] = df.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\
                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \
                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \
                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })
    
df = df.drop('datetime',1)
msno.matrix(df,figsize=(12,5))

fig, axes = plt.subplots(2,2)
fig.set_size_inches(12,10)

sns.boxplot(data=df,y='count',ax=axes[0][0])
sns.boxplot(data=df, x='season',y='count',orient="v",ax=axes[0][1])
sns.boxplot(data=df, x='hour',y='count',orient="v",ax=axes[1][0])
sns.boxplot(data=df, x='workingday',orient="v",y='count',ax=axes[1][1])

axes[0][0].set(ylabel='Count',title='Box Plot on Count')
axes[0][1].set(xlabel='Season',ylabel='Count',title='Box plot on Count Across Seaborn')
axes[1][0].set(xlabel='Hour of the day',ylabel='Count',title='Boc plot on Count Across Hour Of The Day')
axes[1][1].set(ylabel='Count',xlabel='Working Day',title='Box Plot On Count Across Working Day')


dailyDataWithoutOutliers = df[np.abs(df["count"]-
                                            df["count"].mean())<=(3*df["count"].std())]

fig, axes = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(df.corr(),vmax=.8,square=True,annot=True)

fig, ax = plt.subplots(ncols=3)
fig.set_size_inches(12,5)

sns.regplot(data=df,x='temp',y='count',ax=ax[0])
sns.regplot(data=df,x='windspeed',y='count',ax=ax[1])
sns.regplot(data=df,x='humidity',y='count',ax=ax[2])

fig, ax = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(12,10)

sns.distplot(df['count'],ax=ax[0][0])
stats.probplot(df["count"], dist='norm', fit=True, plot=ax[0][1])
sns.distplot(np.log(dailyDataWithoutOutliers['count']),ax=ax[1][0])
stats.probplot(np.log1p(dailyDataWithoutOutliers['count']),dist='norm',fit='True',plot=ax[1][1])

fig, ax = plt.subplots(nrows=3)
fig.set_size_inches(12,20)

sortOrder=['January','February','March','April','May','June','July','August','September','October','November','December']
hueOrder = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

monthAgregated = pd.DataFrame(df.groupby('month')['count'].mean()).reset_index()
sns.barplot(data=monthAgregated,x='month',y='count',order=sortOrder,ax=ax[0])
ax[0].set(title='Average Count By Month')

hourAgregated = pd.DataFrame(df.groupby(['hour','season'])['count'].mean()).reset_index()
sns.pointplot(x = hourAgregated['hour'], y=hourAgregated['count'],hue=hourAgregated['season'],
              join=True,ax=ax[1])
ax[1].set(title='Average Users Count By Hour Of The Day Across Season')

hourAgregated = pd.DataFrame(df.groupby(['hour','weekday'])['count'].mean()).reset_index()
sns.pointplot(x = hourAgregated['hour'],y=hourAgregated['count'],hue=hourAgregated['weekday'],
              join=True,ax=ax[2])
ax[2].set(title='Average Users Count By Hour Of The Day Across Weekdays')

'''
hourAgregated = pd.DataFrame(df.groupby(['hour','variable'])['count'].mean()).reset_index()
sns.pairplot(x=hourAgregated['hour'],y=hourAgregated['count'],hue=hourAgregated['variable'],
             join=True,ax=ax[3])
ax[3].set(title='Average Users Count By Hour Of The Day Across User Type')'''

del ax,axes,hourAgregated,monthAgregated,sortOrder,hueOrder,fig

dataTrain = pd.read_csv("train.csv")
dataTest = pd.read_csv("test.csv")

data = dataTrain.append(dataTest)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)

data["date"] = data.datetime.apply(lambda x : x.split()[0])
data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)


from sklearn.ensemble import RandomForestRegressor

dataWind0 = data[data["windspeed"]==0]
dataWindNot0 = data[data["windspeed"]!=0]
rfModel_wind = RandomForestRegressor()
windColumns = ["season","weather","humidity","month","temp","year","atemp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
dataWind0["windspeed"] = wind0Values
data = dataWindNot0.append(dataWind0)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)

categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]
numericalFeatureNames = ["temp","humidity","windspeed","atemp"]
dropFeatures = ['casual',"count","datetime","date","registered"]

for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")
    
#Splitting Train And Test Data

dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])
datetimecol = dataTest["datetime"]
yLabels = dataTrain["count"]
yLablesRegistered = dataTrain["registered"]
yLablesCasual = dataTrain["casual"]

#Dropping Unncessary Variables

dataTrain  = dataTrain.drop(dropFeatures,axis=1)
dataTest  = dataTest.drop(dropFeatures,axis=1)

#RMSLE Scorer

def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#Linear Regression Model¶

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logistic regression model
lModel = LinearRegression()

# Train the model
yLabelsLog = np.log1p(yLabels)
lModel.fit(X = dataTrain,y = yLabelsLog)

from sklearn.metrics import mean_squared_log_error as msle
# Make predictions
preds = lModel.predict(X= dataTrain)
print ("RMSLE Value For Linear Regression: ",np.sqrt(msle(np.exp(yLabelsLog),np.exp(preds))))

#Regularization Model - Ridge¶

ridge_m_ = Ridge()
ridge_params_ = { 'max_iter':[3000],'alpha':[0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
grid_ridge_m = GridSearchCV( ridge_m_,
                          ridge_params_,
                          scoring = rmsle_scorer,
                          cv=5)
yLabelsLog = np.log1p(yLabels)
grid_ridge_m.fit( dataTrain, yLabelsLog )
preds = grid_ridge_m.predict(X= dataTrain)
print (grid_ridge_m.best_params_)
print ("RMSLE Value For Ridge Regression: ",np.sqrt(msle(np.exp(yLabelsLog),np.exp(preds))))

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_ridge_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmsle"] = df["mean_validation_score"].apply(lambda x:-x)
sns.pointplot(data=df,x="alpha",y="rmsle",ax=ax)

#Regularization Model - Lasso¶
lasso_m_ = Lasso()

alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])
lasso_params_ = { 'max_iter':[3000],'alpha':alpha}

grid_lasso_m = GridSearchCV( lasso_m_,lasso_params_,scoring = rmsle_scorer,cv=5)
yLabelsLog = np.log1p(yLabels)
grid_lasso_m.fit( dataTrain, yLabelsLog )
preds = grid_lasso_m.predict(X= dataTrain)
print (grid_lasso_m.best_params_)
print ("RMSLE Value For Lasso Regression: ",np.sqrt(msle(np.exp(yLabelsLog),np.exp(preds))))

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_lasso_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmsle"] = df["mean_validation_score"].apply(lambda x:-x)
sns.pointplot(data=df,x="alpha",y="rmsle",ax=ax)

#Ensemble Models - Random Forest¶

from sklearn.ensemble import RandomForestRegressor
rfModel = RandomForestRegressor(n_estimators=100)
yLabelsLog = np.log1p(yLabels)
rfModel.fit(dataTrain,yLabelsLog)
preds = rfModel.predict(X= dataTrain)
print ("RMSLE Value For Random Forest: ",np.sqrt(msle(np.exp(yLabelsLog),np.exp(preds))))


#Ensemble Model - Gradient Boost¶
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41
yLabelsLog = np.log1p(yLabels)
gbm.fit(dataTrain,yLabelsLog)
preds = gbm.predict(X= dataTrain)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))


predsTest = gbm.predict(X= dataTest)
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(yLabels,ax=ax1,bins=50)
sns.distplot(np.exp(predsTest),ax=ax2,bins=50)

















