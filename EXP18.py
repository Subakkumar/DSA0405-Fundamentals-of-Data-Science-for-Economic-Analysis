import pandas as pd
import numpy as np
data = {
    'City' : ['A','A','A','B','B','C','C','D','D','D'],
    'Date' : [23-1-1,23-1-1,23-1-1,23-1-2,23-1-2,23-1-3,23-1-3,23-1-4,23-1-4,23-1-4],
    'Temperature' : [34,45,34,23,45,23,23,45,40,33]
    }
temperature_data=pd.DataFrame(data)
temperature_data['Date'] = pd.to_datetime(temperature_data['Date'])

temperature_data['Year'] = temperature_data['Date'].dt.year
temperature_data['Month'] = temperature_data['Date'].dt.month
temperature_data['Day'] = temperature_data['Date'].dt.day

mean_temperatures = temperature_data.groupby('City')['Temperature'].mean()

std_dev_temperatures = temperature_data.groupby('City')['Temperature'].std()

temperature_range = temperature_data.groupby('City')['Temperature'].agg(lambda x: np.max(x) - np.min(x))
city_with_highest_range = temperature_range.idxmax()

most_consistent_city = std_dev_temperatures.idxmin()

print("Task 1: Mean Temperature for Each City")
print(mean_temperatures)
print("\nTask 2: Standard Deviation of Temperature for Each City")
print(std_dev_temperatures)
print(f"\nTask 3: City with the Highest Temperature Range: {city_with_highest_range}")
print(f"\nTask 4: City with the Most Consistent Temperature: {most_consistent_city}")
