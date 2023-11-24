import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {'Study Time (hours)': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Exam Score': [60, 65, 70, 75, 80, 85, 90, 95, 100]}

df = pd.DataFrame(data)

print(df.head())

correlation = df['Study Time (hours)'].corr(df['Exam Score'])
print(f"\nCorrelation Coefficient: {correlation:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Study Time (hours)', y='Exam Score', data=df)
plt.title('Study Time vs. Exam Score')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Score')
plt.show()

sns.jointplot(x='Study Time (hours)', y='Exam Score', data=df, kind='reg')
plt.show()

sns.pairplot(df)
plt.show()

