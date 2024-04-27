import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

my_data= pd.read_csv("sales data file.csv")
my_data.dropna(inplace=True)
my_data

my_data.info

my_data.hist(figsize=(10,8),bins = 50)

plt.figure(figsize=(10,8))
sns.heatmap(my_data.corr(),annot=True,cmap="spring")


sns.pairplot(my_data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')


sns.pairplot(my_data)
plt.show()


X=my_data.drop(["Sales"],axis=1)
X

y=my_data["Sales"]
y=pd.DataFrame(y)
y


#Standard Scaler for Data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X= scaler.fit_transform(X)
X

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
y= scaler.fit_transform(y)
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=33, shuffle =True)
from sklearn.ensemble import RandomForestRegressor
reg_moduel=RandomForestRegressor(n_estimators=250,random_state=0)
reg_moduel.fit(X_train,y_train)
#Calculating Details
print('Random Forest Regressor Train Score is : ' ,  reg_moduel.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , reg_moduel.score(X_test, y_test))


