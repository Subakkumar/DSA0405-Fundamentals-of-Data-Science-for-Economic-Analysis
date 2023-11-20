import numpy as np
fuel_efficiency = np.array([25, 30, 28, 22, 35, 32, 26, 24])
average_fuel_efficiency = np.mean(fuel_efficiency)
fuel_efficiency_model_1 = fuel_efficiency[2]
fuel_efficiency_model_2 = fuel_efficiency[5]
percentage_improvement = ((fuel_efficiency_model_2 - fuel_efficiency_model_1) / fuel_efficiency_model_1) * 100
print("Average Fuel Efficiency:", average_fuel_efficiency)
print(f"Percentage Improvement between Model {2} and Model {5}:", percentage_improvement)
