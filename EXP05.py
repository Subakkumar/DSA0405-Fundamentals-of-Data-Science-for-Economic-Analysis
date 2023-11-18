import numpy as np
student_scores = np.array([[90, 85, 88, 92],
                            [78, 89, 92, 87],
                            [94, 91, 87, 84],
                            [85, 88, 90, 82]])

average_scores_per_subject = np.mean(student_scores, axis=0)

subject_with_highest_average = np.argmax(average_scores_per_subject)

print("Average score:", average_scores_per_subject)
print("Subject with the highest average score:", subject_with_highest_average)
