import csv

from decimal import Decimal
import order as order
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from datetime import datetime

from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from geopy.distance import geodesic


########################################################################################################################
# Read data set and set the date format to 'yyyy-mm-dd'
dataset = pd.read_csv('br_traffic_incidents.csv', parse_dates=['CRASH DATE'])

# reduce dataset to only those wrecks in district 1 for simpler calculations
district1 = dataset[dataset['DISTRICT'] == '1']

# data cleaning -- drop unnecessary values
district1.drop(['FILE#', 'ZONE', 'SUBZONE', 'STREET TYPE', 'STREET#', 'STREET DIRECTION', 'STREET TYPE',
                'FORMATTED STREET', 'OCCURED ON', 'HIT&RUN', 'TRAIN INVOLVED', 'FATALITY', 'INJURY', 'PEDESTRIAN',
                'AT INTERSECTION', 'CLOSEST STREET', 'MANNER OF COLLISION', 'SURFACE CONDITION', 'SURFACE TYPE',
                'ROAD CONDITION', 'ROAD TYPE', 'ALIGNMENT', 'PRIMARY FACTOR', 'SECOND FACTOR', 'WEATHER',
                'LOCATION KIND', 'RELATION ROADWAY', 'ACCESS CONTROL', 'LIGHTING'], inplace=True, axis=1)

# get only Airline streets in district 1
ALstreet = district1[district1["STREET NAME"] == 'AIRLINE']

# drop more unnecessary data from dataset
ALstreet.drop(['CRASH DATE', 'DISTRICT', 'GEOLOCATION', 'STREET NAME'], inplace=True, axis=1)

# sort the data by crash date
sorted_by_date = dataset.sort_values(by='CRASH DATE')

# plot crash date vs total vehicles
sorted_by_date.plot(x='CRASH DATE', y='TOT VEH', style='o')
plt.title('Crash Event vs Total Vehicles')
plt.xlabel('Crash Date')
plt.ylabel('Total Vehicles')
plt.show()

# plot crash date vs total vehicles by year
# sorted_by_date.groupby(sorted_by_date['CRASH DATE'].dt.year)
sorted_by_date.groupby(sorted_by_date['CRASH DATE'].dt.year)['TOT VEH'].agg(['sum']).plot(kind='bar')
plt.ylabel('TOTAL VEHICLES')
plt.title('Crash Date vs Total Vehicles by Year')
plt.show()

# plot crash date vs total vehicles by month
sorted_by_date.groupby(sorted_by_date['CRASH DATE'].dt.month)['TOT VEH'].agg(['sum']).plot(kind='bar')
plt.ylabel('TOTAL VEHICLES')
plt.title('Crash Date vs Total Vehicles by Month')
plt.show()

'''
# plot crash date vs total vehicles by day
sorted_by_date.groupby(sorted_by_date['CRASH DATE'].dt.day)['TOT VEH'].agg(['sum']).plot(kind='bar')
plt.ylabel('TOTAL VEHICLES')
plt.title('Crash Date vs Total Vehicles by Day')
plt.show()
'''

# get a list of the top 10 streets with most wrecks
top10list = sorted_by_date['STREET NAME'].value_counts()[:10].index.tolist()

# reduce dataset to only include top 10 streets
top10 = sorted_by_date[(sorted_by_date['STREET NAME'] == top10list[0]) | (sorted_by_date['STREET NAME'] == top10list[5]) |
                       (sorted_by_date['STREET NAME'] == top10list[1]) | (sorted_by_date['STREET NAME'] == top10list[6]) |
                       (sorted_by_date['STREET NAME'] == top10list[2]) | (sorted_by_date['STREET NAME'] == top10list[7]) |
                       (sorted_by_date['STREET NAME'] == top10list[3]) | (sorted_by_date['STREET NAME'] == top10list[8]) |
                       (sorted_by_date['STREET NAME'] == top10list[4]) | (sorted_by_date['STREET NAME'] == top10list[9])]

'''
# plot top 10 streets on bar graph
top10.groupby(top10['STREET NAME'])['TOT VEH'].agg(['sum']).plot(kind='bar')
plt.title('Top 10 Most Dangerous Streets')
plt.xlabel('Street Name')
plt.ylabel('Total Vehicles Involved')
plt.show()
'''

'''
# plot all streets vs total vehicles as bar plot
sorted_by_date.groupby(sorted_by_date['STREET NAME'])['TOT VEH'].agg(['sum']).plot(kind='bar')
plt.xlabel('Street Name')
plt.ylabel('Total Vehicles Involved')
plt.title('All Streets')
plt.show()
'''

# graph of total wrecks per street
streetwrecks = dataset.groupby(dataset['STREET NAME'], as_index=False).size().sort_values(ascending=False)[:10]\
    .plot(kind='bar')
plt.xlabel('Street Name')
plt.ylabel('Total Wrecks')
plt.title('Top 10 Most Dangerous Streets')
plt.show()

# list of the top 10 most wrecks per day
datewrecks = dataset.groupby(dataset['CRASH DATE'], as_index=False).size().sort_values(ascending=False)[:10]
print('Top 10 Most Dangerous Dates Values:')
print(datewrecks.to_string())

# graph of the top 10 most wrecks per day
datewrecksplt = dataset.groupby(dataset['CRASH DATE'], as_index=False).size().sort_values(ascending=False)[:10].plot(kind='bar')
plt.xlabel('Street Name')
plt.ylabel('Total Wrecks')
plt.title('Top 10 Most Dangerous Dates')
plt.show()

# Linear Regression on Date of Crash and the number of crashes that day
condensed = dataset.pivot_table(index=['CRASH DATE'], aggfunc='size') # puts values in order for some reason
test = Series(data=condensed, name='Crashes')
new = pd.DataFrame(data=dataset['CRASH DATE'].sort_values())
new.drop_duplicates(subset=['CRASH DATE'], inplace=True)
new.insert(1, 'Date Time Float', value=new['CRASH DATE'].values.astype(float))
new.insert(2, 'Crashes', value=test.values)
X=new['Date Time Float'].values.reshape(-1, 1)
y=new['Crashes'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

# To retrieve the intercept:
#print(regressor.intercept_)

# For retrieving the slope:
#print(regressor.coef_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#######################################################################################################################
# EXTRA STUFF

'''
# Multiple Linear Regression

ds = new
ds.isnull().any()
ds = ds.fillna(method='ffill')
X = ds[['Date Time Float']].values #add more columns
y = ds['Crashes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# gets first 100 rows
simple = dataset[:100][:]


# Getting Distance between the first two geo locations
split1 = dataset['GEOLOCATION'][0].split('\n')[2].split('(')[1].split(', ')
split2 = dataset['GEOLOCATION'][1].split('\n')[2].split('(')[1].split(', ')
x1 = Decimal(split1[0])
x2 = Decimal(split2[0])
y1 = Decimal(split1[1].split(')')[0])
y2 = Decimal(split2[1].split(')')[0])
glocation1 = (x1, y1)
glocation2 = (x2, y2)
distance = geodesic(glocation1, glocation2).miles
#print(distance)

# get second column
dates = dataset.values[:, 1]

# get column data types
column_types = dataset.dtypes
#print(column_types)

# gets dimensions of dataset
dimensions = dataset.shape
#print(dimensions)

# gives count, mean, std deviation, and max values for total vehicles and street number
description = dataset.describe()
#print(description)
'''