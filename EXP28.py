import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

car_data = sns.load_dataset('mpg')

print(car_data.head())

sns.scatterplot(x='horsepower', y='mpg', hue='origin', size='weight', data=car_data)
plt.title('Multivariate Scatterplot')
plt.show()

sns.pairplot(car_data, vars=['mpg', 'horsepower', 'weight', 'acceleration'], hue='origin')
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()
