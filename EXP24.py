products_sold = {
    'product1': 50,
    'product2': 30,
    'product3': 80,
    'product4': 20,
    'product5': 60
}
frequency_distribution = {}

for product, quantity in products_sold.items():
    frequency_distribution[product] = quantity

most_popular_product = max(frequency_distribution, key=frequency_distribution.get)

print("Frequency Distribution of Products Sold:")
for product, quantity in frequency_distribution.items():
    print(f"{product}: {quantity} times")

print("\nThe most popular product is:", most_popular_product)
