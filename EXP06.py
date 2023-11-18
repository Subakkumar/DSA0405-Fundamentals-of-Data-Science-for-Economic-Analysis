import numpy as np
time_intervals = np.array([0, 1, 2, 3, 4])
vertical_positions = np.array([0, 10, 20, 15, 5])
change_in_vertical_position = np.diff(vertical_positions)
change_in_time = np.diff(time_intervals)
average_velocity = change_in_vertical_position / change_in_time
print("Change in Vertical Position:", change_in_vertical_position)
print("Change in Time:", change_in_time)
print("Average Velocity:", average_velocity)
