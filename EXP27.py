import numpy as np
from scipy.stats import t

revenue_data = np.random.normal(loc=100, scale=20, size=365)

sample_mean = np.mean(revenue_data)
sample_std = np.std(revenue_data, ddof=1)
sample_size = len(revenue_data)

confidence_level = 0.95

alpha = 1 - confidence_level
degrees_freedom = sample_size - 1
t_critical = t.ppf(1 - alpha / 2, degrees_freedom)
margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"Sample Mean: {sample_mean}")
print(f"Margin of Error: {margin_of_error}")
print(f"Confidence Interval ({confidence_level * 100}%): {confidence_interval}")
