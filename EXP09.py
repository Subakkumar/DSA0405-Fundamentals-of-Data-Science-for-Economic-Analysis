import matplotlib.pyplot as plt
import numpy as np

months = ['January', 'February', 'March', 'April', 'May', 'June']
sales_values = [20000, 25000, 18000, 30000, 28000, 35000]

plt.figure(figsize=(8, 5))
plt.plot(months, sales_values, marker='o', color='b', label='Monthly Sales')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales Value')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(months, sales_values, color='green', alpha=0.7, label='Monthly Sales')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales Value')
plt.legend()
plt.grid(axis='y')
plt.show()
