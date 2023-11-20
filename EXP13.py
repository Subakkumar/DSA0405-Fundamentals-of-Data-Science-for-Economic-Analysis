import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
reviews_data = pd.read_csv(r"C:\Users\kbala\Downloads\reviews_data.csv")
all_reviews = ' '.join(reviews_data['Review'].astype(str))
words = np.array(all_reviews.split())
words_series = pd.Series(words)
word_freq = words_series.value_counts()
print("Top 10 Most Common Words:")
print(word_freq.head(10))
word_freq.head(30).plot(kind='bar', figsize=(10, 6), title='Frequency Distribution of Words in Customer Reviews')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()


# Link for data set :https://www.kaggle.com/datasets/harshalhonde/starbucks-reviews-dataset
