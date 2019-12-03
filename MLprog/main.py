import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as SeabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#%matplotlib inline

# read the dataset
dataset = pd.read_csv('br_traffic_incidents.csv')
dataset.head(10)

# get column names
column_names = dataset.columns
print(column_names)

# get column data types
column_types = dataset.dtypes
print(column_types)

# gives dimensions of dataset
var = dataset.shape
print(var)

# gives count, mean, std deviation, and max values for total vehicles and street number
print(dataset.describe())

# plots data of the crash date and total vehicles
dataset.plot(x='CRASH DATE', y='TOT VEH', style='o')
plt.title('Crash Date by Year')
plt.xlabel('Crash Date')
plt.ylabel('Total Vehicles')
plt.show()
