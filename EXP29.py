import numpy as np
import pandas as pd
from scipy import stats
np.random.seed(42)
n_reviews = 500
customer_reviews = pd.DataFrame({
    'rating': np.random.choice([1, 2, 3, 4, 5], n_reviews, p=[0.05, 0.1, 0.2, 0.3, 0.35])
})
print(customer_reviews.head())
sample_mean = customer_reviews['rating'].mean()
standard_error = stats.sem(customer_reviews['rating'])
confidence_level = 0.95
degrees_of_freedom = len(customer_reviews['rating']) - 1
confidence_interval = stats.t.interval(confidence_level, degrees_of_freedom, loc=sample_mean, scale=standard_error)

print(f"Sample Mean Rating: {sample_mean:.2f}")
print(f"Confidence Interval ({confidence_level*100}%): {confidence_interval}")
