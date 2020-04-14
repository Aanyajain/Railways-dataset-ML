import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"C:\Users\om\Desktop\ML\Social_Network_Ads.csv")

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l_x=LabelEncoder()
data['Gender']=l_x.fit_transform(data['Gender'])

x=data.drop(['Purchased'],axis=1)
y=data['Purchased']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
#SVM
from sklearn.svm import SVC
reg=SVC(kernel="linear",random_state=0)
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
#print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(reg.score(x_train, y_train)))
#print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(reg.score(x_test, y_test)))
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
x1, x2 = np.meshgrid(np.arange(start=x1_min, stop = x1_max, step = 0.01),
                     np.arange(start = x2_min, stop = x2_max, step = 0.01))
x_pred = np.array([x1.ravel(), x2.ravel()] + [np.repeat(0, x1.ravel().size) for _ in range(2)]).T
# x_pred now has a grid for x1 and x2 and average value (0) for x3 through x9
pred = reg.predict(x_pred).reshape(x1.shape)   # is a matrix of 0's and 1's !
plt.contourf(x1, x2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],cmap=ListedColormap(("red","green")))
plt.title("SVM classification training")
plt.show()
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
cls=GaussianNB()
cls.fit(x_train,y_train)
y_pred=cls.predict(x_test)
# training data
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
x1, x2 = np.meshgrid(np.arange(start=x1_min, stop = x1_max, step = 0.01),
                     np.arange(start = x2_min, stop = x2_max, step = 0.01))
x_pred = np.array([x1.ravel(), x2.ravel()] + [np.repeat(0, x1.ravel().size) for _ in range(2)]).T
# x_pred now has a grid for x1 and x2 and average value (0) for x3 through x9
pred = cls.predict(x_pred).reshape(x1.shape)   # is a matrix of 0's and 1's !
plt.contourf(x1, x2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],cmap=ListedColormap(("red","green")))
plt.title("Naive Bayes classification training")
plt.show()
print('The accuracy of the naive bayes classifier on training data is {:.2f} out of 1'.format(cls.score(x_train, y_train)))
print('The accuracy of the naive bayes classifier on test data is {:.2f} out of 1'.format(cls.score(x_test, y_test)))
#testing data
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
x1, x2 = np.meshgrid(np.arange(start=x1_min, stop = x1_max, step = 0.01),
                     np.arange(start = x2_min, stop = x2_max, step = 0.01))
x_pred = np.array([x1.ravel(), x2.ravel()] + [np.repeat(0, x1.ravel().size) for _ in range(2)]).T
# x_pred now has a grid for x1 and x2 and average value (0) for x3 through x9
pred = cls.predict(x_pred).reshape(x1.shape)   # is a matrix of 0's and 1's !
plt.contourf(x1, x2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],cmap=ListedColormap(("red","green")))
plt.title("Naive Bayes classification testing")
plt.show()
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
#Decision tree classification

