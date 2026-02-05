import pandas as pd

df = pd.read_csv('delaney_solubility_with_descriptors.csv')   #Loading Data
# print(df)

y = df['logS']  #takes only the last column "logS"
# print(y)

X = df.drop('logS', axis = 1)  #removes/drop last column and axis=1 means works on columns & 0 for rows
# print(X)

''' using Scikit-learn for Spliting data for training and testing '''
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100) #random_state to get consistent data split in every run
# print(X_train) #80% data for training 
# print(X_test) #20% data for testing 
# print(y_train)
# print(y_test)

''' Training the Model using Linear Regression '''
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
# print(LinearRegression)

'''applying model to make prediction'''
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
# print(y_lr_train_pred)
# print(y_lr_test_pred)
