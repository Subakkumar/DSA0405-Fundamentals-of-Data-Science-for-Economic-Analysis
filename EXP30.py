import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
n_sales = 500
shoe_sales = pd.DataFrame({
    'shoe_size': np.random.choice(['US 7', 'US 8', 'US 9', 'US 10', 'US 11'], n_sales),
    'quantity': np.random.randint(1, 20, n_sales)
})

print(shoe_sales.head())
size_distribution = shoe_sales.groupby('shoe_size')['quantity'].sum()
print("Frequency Distribution Table:")
print(size_distribution)
plt.figure(figsize=(10, 6))
size_distribution.plot(kind='bar', color='skyblue')
plt.title('Shoe Size Frequency Distribution')
plt.xlabel('Shoe Size')
plt.ylabel('Quantity Sold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
