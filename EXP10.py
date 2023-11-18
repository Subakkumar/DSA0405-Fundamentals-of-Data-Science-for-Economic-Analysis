import matplotlib.pyplot as plt
import numpy as np

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
temperature_values = [10, 12, 15, 18, 22, 25, 28, 26, 23, 20, 15, 12]
rainfall_values = [50, 40, 60, 80, 70, 90, 100, 110, 80, 70, 60, 45]

plt.figure(figsize=(10, 5))
plt.plot(months, temperature_values, marker='o', color='orange', label='Monthly Temperature')
plt.title('Monthly Temperature Data')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(months, rainfall_values, color='blue', label='Monthly Rainfall', s=100)
plt.title('Monthly Rainfall Data')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()
