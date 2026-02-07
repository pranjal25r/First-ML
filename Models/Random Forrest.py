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

''' Training the Model using Random Forest  '''
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train,y_train)
# print(RandomForestRegressor)

'''applying model to make prediction'''
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)
# print(y_rf_train_pred)
# print(y_rf_test_pred)

'''Evaluate Model Performance'''
from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2 = r2_score(y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2 = r2_score(y_test,y_rf_test_pred)

# print("RF MSE (Train)", rf_train_mse)
# print("RF R2 (Train)", rf_train_r2)
# print("RF MSE (Test)", rf_test_mse)
# print("RF R2 (Test)", rf_test_r2)
'''More Systematic Printing Method of above Results'''
rf_results = pd.DataFrame(['Random Forest ', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

print(rf_results)