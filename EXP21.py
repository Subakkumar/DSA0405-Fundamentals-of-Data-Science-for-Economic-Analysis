import numpy as np
import scipy.stats
sample_size = int(input("Enter sample size: "))
confidence_level = float(input("Enter confidence level (between 0 and 1): "))
precision = float(input("Enter level of precision: "))
data = np.genfromtxt(r"C:\Users\kbala\Documents\rare elements.csv", delimiter=',')
sample_mean = np.mean(data)
standard_error = scipy.stats.sem(data)
margin_of_error = standard_error * scipy.stats.t.ppf((1 + confidence_level) / 2, sample_size - 1)

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"The sample mean is: {sample_mean}")
print(f"The margin of error is: {margin_of_error}")
print(f"The {confidence_level * 100}% confidence interval is: {confidence_interval}")


# data set
10.2
12.5
8.3
15.7
9.8
14.2
11.0
13.6
