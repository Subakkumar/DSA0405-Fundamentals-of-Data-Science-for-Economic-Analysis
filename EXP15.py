import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

np.random.seed(42)
control_group = np.random.normal(loc=30, scale=10, size=100)
treatment_group = np.random.normal(loc=35, scale=10, size=100)

plt.boxplot([control_group, treatment_group], labels=['Control (Placebo)', 'Treatment'])
plt.title('Boxplot of Control Group (Placebo) and Treatment Group')
plt.ylabel('Effectiveness')
plt.show()

t_stat, p_value = ttest_ind(control_group, treatment_group)

print(f'T-statistic: {t_stat}')
print(f'P-value: {p_value}')

alpha = 0.05
if p_value < alpha:
    print("The p-value is less than the significance level. Reject the null hypothesis.")
    print("There is evidence that the new treatment has a statistically significant effect.")
else:
    print("Fail to reject the null hypothesis. There is no significant evidence of a treatment effect.")
