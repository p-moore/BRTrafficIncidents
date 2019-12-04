import csv

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

index = dataset['GEOLOCATION'][0]
#print(index)

split = index.split('\n')
print(split[2])

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
sorted_by_date.plot(x='CRASH DATE', y='TOT VEH', style='o')
plt.title('Crash Date by Year')
plt.xlabel('Crash Date')
plt.ylabel('Total Vehicles')
#plt.show()

plt.figure()
plt.tight_layout()
seabornInstance.distplot(dataset['TOT VEH'])
#plt.show()
