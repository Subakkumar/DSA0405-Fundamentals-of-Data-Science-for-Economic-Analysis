import numpy as np
import matplotlib.pyplot as plt

temperature = np.random.randint(50, 100, 365)
rainfall = np.random.uniform(0, 10, 365)

correlation_coefficient = np.corrcoef(temperature, rainfall)[0, 1]

plt.scatter(temperature, rainfall, alpha=0.5)
plt.title('Temperature vs Rainfall')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.text(55, 9, f'Correlation Coefficient: {correlation_coefficient:.2f}', fontsize=10, color='red')
plt.show()
print(temperature)
print()
print(rainfall)
