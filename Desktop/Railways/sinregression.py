import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"C:\Users\om\Desktop\sinproject\d.csv")

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l_x=LabelEncoder()
data['Train Name']=l_x.fit_transform(data['Train Name'])
l_x1=LabelEncoder()
data['Station Code']=l_x1.fit_transform(data['Station Code'])
l_x2=LabelEncoder()
data['Station Name']=l_x2.fit_transform(data['Station Name'])
l_x3=LabelEncoder()
data['Source Station']=l_x3.fit_transform(data['Source Station'])
l_x4=LabelEncoder()
data['Source Station Name']=l_x4.fit_transform(data['Source Station Name'])
l_x5=LabelEncoder()
data['Destination Station']=l_x5.fit_transform(data['Destination Station'])
l_x6=LabelEncoder()
data['Destination Station Name']=l_x6.fit_transform(data['Destination Station Name'])

x=data.drop(['Distance'],axis=1)
y=data['Distance']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#polynomial regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
reg=PolynomialFeatures(degree=4)
x_poly=reg.fit_transform(x_train)
reg2=LinearRegression()
reg2.fit(x_poly,y_train)
y_pred=reg2.predict(x_poly)
plt.scatter(x_train[:,0],y_train,color='red')
plt.plot(np.sort(x_test[:,0]),np.sort(y_pred),color='blue')
plt.title("railways with poly reg")
plt.show()

#linear regression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
plt.scatter(x_train[:,3],y_train,color='red')
plt.plot(np.sort(x_test[:,3]),np.sort(y_pred),color='blue')
plt.title("railways with linear reg")
plt.show()
print('The accuracy of the linear reg is {:.2f} out of 1 on training data'.format(reg.score(x_train, y_train)))
print('The accuracy of the linear reg is {:.2f} out of 1 on test data'.format(reg.score(x_test, y_test)))
#reg.coef_
#reg.intercept_
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
#KNN
from sklearn.neighbors import KNeighborsClassifier
reg=KNeighborsClassifier()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
plt.scatter(x_train[:,0],y_train,color='red')
plt.plot(x_test[:,0],y_pred,color='blue')
plt.title("railways with KNN reg")
plt.show()
from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(y_test,y_pred)
print(cm2)
print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(reg.score(x_train, y_train)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(reg.score(x_test, y_test)))

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=0)
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
plt.figure()
plt.scatter(x_train[:,0], y_train, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(np.sort(x_test[:,0]), y_pred, color="yellowgreen", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
print('The accuracy of the dec tree regr is {:.2f} out of 1 on training data'.format(reg.score(x_train, y_train)))
print('The accuracy of the dec tree regr is {:.2f} out of 1 on test data'.format(reg.score(x_test, y_test)))
#random forest regression
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
reg= RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
x_grid = np.arange(np.min(x_train[:,2]), np.max(x_train), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_grid, reg.predict(x_grid), color ='blue')
plt.title('Random Forest Regression Model')
plt.xlabel('Years')
plt.ylabel('Account Balance')
plt.show()
