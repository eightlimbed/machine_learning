#!/usr/bin/python3

# Notes from pandas crash course
import pandas as pd
import numpy as np

# pandas has to main data types: Series (columns) and DataFrame (rows + columns)

# create two series: city_names and population
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacremento'])
population = pd.Series([852469, 1015785, 485199])

# combine these two series into a DataFrame
cities = pd.DataFrame({"Cities": city_names, "Pop": population})

# you can add more columns, similar to adding a key/val in a dict
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])

# you can create new Series based on other columns
cities['Population density'] = cities['Pop'] / cities['Area square miles']

print(type(cities['Area square miles']))  # pd.Series
print(type(cities['Population density'])) # pd.Series

# creating a new series that returns True if the name starts with 'San'
# and has an area larger than 50 square miles
cities['BigSaint'] = (cities['Area square miles'] > 50) & \
                     (cities['Cities'].apply(lambda x: x.startswith('San')))

# DataFrames are auto-indexed
print(city_names.index)
print(cities.index)

# you can reindex() them to rearrange the order
print(cities.reindex([2, 0, 1]))

# using NumPy's random.permutation() you can randomize the arrangement of data
# this is useful for training models
print(cities.reindex(np.random.permutation(cities.index)))
