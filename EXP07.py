import numpy as np
file_path = r"C:\Users\kbala\Downloads\house_data.csv"
house_data = np.genfromtxt(file_path, delimiter=',', skip_header=1) 

bedrooms = house_data[:, 0] 
sale_prices = house_data[:, -1]  

houses_more_than_four_bedrooms = np.where(bedrooms > 4)

average_sale_price_more_than_four_bedrooms = np.mean(sale_prices[houses_more_than_four_bedrooms])
print("Average Sale Price of Houses with More than Four Bedrooms:", average_sale_price_more_than_four_bedrooms)

# data set
bedrooms	sqft	price
3	        1500	200000
4	        1800	250000
5	        2000	280000
4	        1600	230000
3	        1400	190000
