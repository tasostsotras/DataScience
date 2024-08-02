from pickle import FALSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import requests
import csv
from io import BytesIO
import plotly.express as px
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import date, datetime
import holidays
df = pd.read_csv('C:/Users/tasos/Desktop/DataScience14Feb2024/PEMS-BAY.csv')
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.to_numpy())
print(df.info())
threshold = 1.5
df = df.drop(34548)
df = df.to_csv('C:/Users/tasos/Desktop/DataScience14Feb2024/PEMS-BAY.csv',index=False)
df = pd.read_csv('C:/Users/tasos/Desktop/DataScience14Feb2024/PEMS-BAY.csv')
df.describe()
print(df.describe())
q1=df.quantile(0.25)
q3=df.quantile(0.75)
IQR=q3-q1
outliers = (df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR))#Which are the outliers?
print(q1)
print(q3)
print(IQR)
print(outliers)
print("The maximum Inter-Quartile Range is: ",max(IQR))
max_iqr_column = IQR.idxmax() #Finds which column has the maximum Inter-Quartile Range
print("The column with the maximum IQR is: ", max_iqr_column)
print("The minimum Inter-Quartile Range is: ", min(IQR))
min_iqr_column = IQR.idxmin()
print("The column with the minimum IQR is: ", min_iqr_column)
clean_data = df[~outliers.any(axis=1)]#Eliminating outliers
print(clean_data)
df = pd.read_csv('C:/Users/tasos/Desktop/DataScience14Feb2024/PEMS-BAY.csv')
fig = px.line(df, x = '400971', y = '401817', title='Comparison of average speed between sensors with maximum and minimum IQR')
fig.show()
num_features = ['400971', '401817']
X = df[num_features]
scaler = StandardScaler()#Normalization
X_normalized = scaler.fit_transform(X)
print(X_normalized)
df2 = pd.read_csv('C:/Users/tasos/Desktop/DataScience14Feb2024/PEMS-BAY-META.csv')
df2.drop('District',inplace=True,axis=1)#We drop columns which aren't useful
df2.drop('County',inplace=True,axis=1)
df2.drop('Type',inplace=True,axis=1) 
df2.drop('User_ID_1',inplace=True,axis=1)
df2.drop('User_ID_2',inplace=True,axis=1)
df2.drop('User_ID_3',inplace=True,axis=1)
df2.drop('User_ID_4',inplace=True,axis=1)
df2.drop('Length',inplace=True,axis=1)
print(df2)
X2 = df.loc[:,['400001', '401327']]
print(X2.head())
y = df.loc[:,['400296']]
print(y.head())
print(df.mean(numeric_only=None))
plot_acf(df['400971'])#Auto-correlation plot for the column with the max IQR
plt.show()
columns = df.columns
for column in columns:#Trying to find trends of each column
    # Get the unique values in the column
    unique_values = df[column].unique()
    
    # Print the column name and its unique values
    print(f"Column: {column}")
    print(f"Unique Values: {unique_values}")
    print()
for date, name in sorted(holidays.US(subdiv='CA', years=2017).items()):#Getting all holidays till June 30th 
    print(date, name)
row_avg = df.mean(axis=1)
df['AVERAGE'] = row_avg
df.to_csv('your_file.csv', index=False)#PEMS-BAY csv but with the AVERAGE column
df_new = pd.read_csv('C:/Users/tasos/Desktop/DataScience14Feb2024/DataScience14Feb2024/your_file.csv')#Read the new file
df_new = df_new[['AVERAGE']]#Only select the "average" column from the 2nd file
df_new.to_csv('PEMS-BAY-AVERAGE.csv', index=False)#Saves only the average values in a third file for easier conclusions
for index, row in df_new.iterrows():
    cell_value = str(row['AVERAGE'])
    if len(cell_value) >= 3:
        new_value = cell_value[:2] +"," + cell_value[2:]
        df_new.at[index, 'AVERAGE'] = new_value

# Save the modified DataFrame to a new CSV file
df_new.to_csv('PEMS-BAY-AVERAGE.csv', index=False)
training_set = df[:25908]#Split the CSV into 3 datasets
validation_set = df[25908:34548]
testing_set = df[34548:]
print(f"Training set size: {len(training_set)}")
print(f"Validation set size: {len(validation_set)}")
print(f"Testing set size: {len(testing_set)}")
print("Training set is:")
print(training_set)
print("Validation set is:")
print(validation_set)
print("Testing set is:")
print(testing_set)
df = df.drop('Date',axis=1)
X_train = training_set.drop('Date', axis=1)
y_train = training_set['AVERAGE']
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Prepare the data for validation
X_validation = validation_set.drop(columns=['Date'])
y_validation = validation_set['AVERAGE']

# Make predictions on the validation set
y_pred = model.predict(X_validation)
print("Predictions of the validation set are: ",y_pred)
# Evaluate the model's performance
mae = mean_absolute_error(y_validation, y_pred)
mape = mean_absolute_percentage_error(y_validation, y_pred)
rmse = mean_squared_error(y_validation, y_pred, squared=False)
r2 = r2_score(y_validation, y_pred)

# Prepare the data for testing
X_test = testing_set.drop(columns=['Date'])
y_test = testing_set['AVERAGE']

# Make predictions on the testing set
y_pred_test = model.predict(X_test)
print("Predictions of the testing set are: ",y_pred_test)
# Evaluate the model's performance on the testing set
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)

# Create a table to display the performance metrics
metrics_table = pd.DataFrame({
    'Time Horizon (minutes)': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    'MAE': [mae, mae, mae, mae, mae, mae, mae, mae, mae, mae, mae, mae],
    'MAPE': [mape, mape, mape, mape, mape, mape, mape, mape, mape, mape, mape, mape],
    'RMSE': [rmse, rmse, rmse, rmse, rmse, rmse, rmse, rmse, rmse, rmse, rmse, rmse],
    'R2': [r2, r2, r2, r2, r2, r2, r2, r2, r2, r2, r2, r2]
})

# Display the table
print(metrics_table)

