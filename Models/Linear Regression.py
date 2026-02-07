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

'''Evaluate Model Performance'''
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2 = r2_score(y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2 = r2_score(y_test,y_lr_test_pred)

# print("LR MSE (Train)", lr_train_mse)
# print("LR R2 (Train)", lr_train_r2)
# print("LR MSE (Test)", lr_test_mse)
# print("LR R2 (Test)", lr_test_r2)
'''More Systematic Printing Method of above Results'''
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

print(lr_results)