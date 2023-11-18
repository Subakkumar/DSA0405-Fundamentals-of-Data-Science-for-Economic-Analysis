import pandas as pd

data = {
    "cust_id" : [1,1,1,2,3,3],
    'order_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2022-12-31', '2023-01-02', '2023-01-04'],
    "prdt_name" : ["A","A","B","B","C","C"],
    "order_quantity" : [9,10,6,4,2,3]
    }
order_data = pd.DataFrame(data)
total_orders = order_data.groupby('cust_id')['order_date'].count()

avg_order_quantity = order_data.groupby('prdt_name')['order_quantity'].mean()

earliest = order_data['order_date'].min()
latest = order_data['order_date'].max()

print("Total orders:")
print(total_orders)
print("Average order quantity:")
print(avg_order_quantity)
print("Earliest Order Date:", earliest)
print("Latest Order Date:", latest)
print(order_data)
