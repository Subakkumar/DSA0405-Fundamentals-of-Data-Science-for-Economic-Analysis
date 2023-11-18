import pandas as pd

property_data = pd.read_csv(r"C:\Users\kbala\Downloads\product_data.csv")

average_price_per_location = property_data.groupby('location')['listing_price'].mean()

properties_more_than_four_bedrooms = property_data[property_data['number_of_bedrooms'] > 4]
number_of_properties_more_than_four_bedrooms = len(properties_more_than_four_bedrooms)

property_with_largest_area = property_data.loc[property_data['area_in_sqft'].idxmax()]

print("Average listing price of properties in each location:")
print(average_price_per_location)

print("Number of properties with more than four bedrooms:", number_of_properties_more_than_four_bedrooms)

print("Property with the largest area:")
print(property_with_largest_area)

#data set
location	listing_price	number_of_bedrooms	area_in_sqft
chennai	  100000	      5	                  1500
chennai	  90000	        3	                  1200
Madurai	  120000	      5	                  1400
Madurai	  230000	      5	                  1800
neyveli	  200000	      4 	                2000
neyveli	  150000	      3	                  1500
