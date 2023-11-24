import pandas as pd
df = pd.read_csv(r"C:\Users\kbala\Downloads\stock_data.csv")

print(df.head())

closing_prices = df['Closing Price']

std_deviation = closing_prices.std()

price_range = closing_prices.max() - closing_prices.min()

daily_percentage_change = closing_prices.pct_change().dropna()

average_daily_percentage_change = daily_percentage_change.mean()

print(f"\nVariability Metrics:")
print(f"Standard Deviation: {std_deviation:.2f}")
print(f"Price Range: {price_range:.2f}")
print(f"Average Daily Percentage Change: {average_daily_percentage_change:.2%}")

if std_deviation > 0:
    print("\nThe stock prices exhibit variability.")
else:
    print("\nThe stock prices show low variability.")

if average_daily_percentage_change > 0:
    print("There is an average daily positive price change.")
else:
    print("There is an average daily negative or zero price change.")
