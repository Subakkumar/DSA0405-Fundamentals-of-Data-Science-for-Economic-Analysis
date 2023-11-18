import pandas as pd

student_data = pd.read_csv(r"C:\Users\suba\Downloads\student_data.csv")

correlation_matrix = student_data.groupby('Course').corr()
correlation_coefficients = correlation_matrix.loc[(slice(None), 'Hours_Studied'), 'Score']
 
strongest_correlations = correlation_coefficients.sort_values(ascending=False)
weakest_correlations = correlation_coefficients.sort_values()

course_data = student_data.groupby('Course').agg({'Score': 'mean', 'Hours_Studied': 'mean'})

print("Correlation coefficients:")
print(correlation_coefficients)
print("Courses with strongest correlation:")
print(strongest_correlations.head())
print("Courses with weakest correlation:")
print(weakest_correlations.head())
print("Course data:")
print(course_data)

##student_csv 
std_id Course Hours_Studied Score
1      Mat    10            95
2      Mat    12            75
3      Sci    8             88
4      Sci    15            79
5      Eng    11            80
6      Eng    15            85
