import pandas as pd
#sales_data = pd.read_csv(r"C:\Users\kbala\Downloads\sales.csv")
data = {
    'Product' : ['A','B','C','D','E','F'],
    'Quantity' : [20,30,50,12,10,100]
    }
sales_data = pd.DataFrame(data)
product_sales = sales_data.groupby('Product')['Quantity'].sum()
sorted_products = product_sales.sort_values(ascending=False)
top_5_products = sorted_products.head(5)
print("Top 5 Products Sold the Most:")
print(top_5_products)

# data set
Product	Item_Price	Quantity
1	      200	        30
2	      300	        15
3	      150	        20
4	      135	        7
5	      100	        20
