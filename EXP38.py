import pandas as pd

df = pd.read_csv(r"C:\Users\kbala\Downloads\temp_data.csv")

print(df.head())

mean_temperatures = df.groupby('City')['Temperature'].mean()
print("\nMean Temperatures:")
print(mean_temperatures)

std_dev_temperatures = df.groupby('City')['Temperature'].std()
print("\nStandard Deviation of Temperatures:")
print(std_dev_temperatures)

temperature_range = df.groupby('City')['Temperature'].max() - df.groupby('City')['Temperature'].min()
city_with_highest_range = temperature_range.idxmax()
print(f"\nCity with the Highest Temperature Range: {city_with_highest_range}")

most_consistent_city = std_dev_temperatures.idxmin()
print(f"\nCity with the Most Consistent Temperature: {most_consistent_city}")


#data set
City,Date,Temperature
New York,2023-01-01,32
New York,2023-01-02,34
New York,2023-01-03,30
New York,2023-01-04,36
New York,2023-01-05,38
Los Angeles,2023-01-01,70
Los Angeles,2023-01-02,72
Los Angeles,2023-01-03,68
Los Angeles,2023-01-04,74
Los Angeles,2023-01-05,76
