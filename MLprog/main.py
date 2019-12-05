import csv

from decimal import Decimal
import order as order
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from geopy.distance import geodesic


########################################################################################################################
# read the dataset
dataset = pd.read_csv('br_traffic_incidents.csv', parse_dates=['CRASH DATE'])
#print(dataset)




'''
width = 1
plt.bar(x, y, width, label='Test')
plt.legend()
plt.xlabel('Street Name')
plt.ylabel('Total Vehicles')
plt.title('Testing')
plt.show()
'''
# get all rows with Airline street
ALstreet = dataset[dataset["STREET NAME"] == 'AIRLINE']
#print(ALstreet)


test = dataset[dataset['DISTRICT'] == '1']

#print(test.resample('Y', on='CRASH DATE'))

test.drop(['FILE#', 'ZONE', 'SUBZONE', 'STREET TYPE', 'STREET#', 'STREET DIRECTION','STREET TYPE', 'FORMATTED STREET',
           'OCCURED ON', 'HIT&RUN', 'TRAIN INVOLVED', 'FATALITY', 'INJURY', 'PEDESTRIAN', 'AT INTERSECTION',
           'CLOSEST STREET', 'MANNER OF COLLISION', 'SURFACE CONDITION', 'SURFACE TYPE', 'ROAD CONDITION', 'ROAD TYPE',
           'ALIGNMENT', 'PRIMARY FACTOR', 'SECOND FACTOR', 'WEATHER', 'LOCATION KIND', 'RELATION ROADWAY',
           'ACCESS CONTROL', 'LIGHTING'], inplace=True, axis=1)


sorted_by_date = test.sort_values(by='CRASH DATE')
print(sorted_by_date)

#year2010 = sorted_by_date[sorted_by_date['CRASH DATE'].split('-')[0] == '2010']
#print(year2010)


objects = (test[test['CRASH DATE'] == '01-01-2015'])
y_pos = np.arange(len(objects))
performance = test['TOT VEH']

# ultimate bar graph
'''
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')

plt.show()
'''

'''
x = test['CRASH DATE']
y = test['TOT VEH']
xpos = np.arange(len(y))
plt.bar(x, x)
plt.show()
print(test)
'''

# gets first 100 rows
simple = dataset[:100][:]
#print(simple)

#x = simple['DISTRICT']
#y = simple['TOT VEH']
#xpos = np.arange(len(y))
#plt.bar(xpos, x)

#dis = simple.groupby('DISTRICT')


#simple.plot(kind = 'bar', x = , y = 'TOT VEH')


#print(sorted_by_date)


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

# get column 'CRASH DATES'
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

# plots data of the crash date and total vehicles
#sorted_by_date.plot(x='DISTRICT', y='TOT VEH', style='o')
plt.title('Most Dangerous Streets')
plt.xlabel('District')
plt.ylabel('Total Vehicles')
#plt.show()

plt.figure()
plt.tight_layout()
seabornInstance.distplot(dataset['TOT VEH'])
#plt.show()

#x = sorted_by_date['DISTRICT'].values.reshape(-1, 1)
#y = sorted_by_date['TOT VEH'].values.reshape(-1, 1)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#regressor = LinearRegression()
#regressor.fit(x_train, y_train) #training the algorithm

#To retrieve the intercept:
#print(regressor.intercept_)
#For retrieving the slope:
#print(regressor.coef_)

#y_pred = regressor.predict(x_test)


########################################################################################################################

'''
#:time = pd.date_range('1/1/2000', periods=2000, freq='60min')

#times = pd.DatetimeIndex(dataset['CRASH TIME'].values)
#grouped = dataset.groupby([times.hour, times.minute])
#print(times)

#times = pd.to_datetime(dataset['CRASH TIME'])
#dataset.groupby([times.hour, times.minute]).value_col.sum()
#print(times)


# Series summary operations.
# We are selecting the column "Y2007", and performing various calculations.
data['Y2007'].sum(), # Total sum of the column values
data['Y2007'].mean(), # Mean of the column values
data['Y2007'].median(), # Median of the column values
data['Y2007'].nunique(), # Number of unique entries
data['Y2007'].max(), # Maximum of the column values
data['Y2007'].min()] # Minimum of the column values

'''