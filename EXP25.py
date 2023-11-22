import numpy as np

dataset = np.random.normal(loc=50, scale=10, size=1000)

population_mean = np.mean(dataset)
population_variance = np.var(dataset, ddof=1)

sample_size = 100
sample = np.random.choice(dataset, size=sample_size, replace=False)

sample_mean = np.mean(sample)
sample_variance = np.var(sample, ddof=1)

print(f"True Population Mean: {population_mean}")
print(f"Estimated Sample Mean: {sample_mean}")
print(f"\nTrue Population Variance: {population_variance}")
print(f"Estimated Sample Variance: {sample_variance}")
