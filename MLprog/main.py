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
print(dataset)

sorted_by_date = dataset.sort_values(by='CRASH DATE')
print(sorted_by_date)

#time = pd.date_range('1/1/2000', periods=2000, freq='60min')

#times = pd.DatetimeIndex(dataset['CRASH TIME'].values)
#grouped = dataset.groupby([times.hour, times.minute])
#print(times)

#times = pd.to_datetime(dataset['CRASH TIME'])
#dataset.groupby([times.hour, times.minute]).value_col.sum()
#print(times)

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
print(distance)

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
sorted_by_date.plot(x='STREET NAME', y='TOT VEH', style='o')
plt.title('Most Dangerous Streets')
plt.xlabel('Street Name')
plt.ylabel('Total Vehicles')
plt.show()

plt.figure()
plt.tight_layout()
seabornInstance.distplot(dataset['TOT VEH'])
#plt.show()
