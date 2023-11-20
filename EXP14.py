import pandas as pd
import string
import matplotlib.pyplot as plt
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    stop_words = set(["the", "and", "is", "in", "to", "of", "it", "that", "for", "you", "on", "with", "are"])
    words = [word for word in words if word not in stop_words]
    return words

def analyze_feedback(data, top_n):
    data['processed_feedback'] = data['Review'].apply(preprocess_text)
    all_words = [word for sublist in data['processed_feedback'] for word in sublist]
    freq_dist = pd.Series(all_words).value_counts()
    print(f"Top {top_n} Most Frequent Words:")
    print(freq_dist.head(top_n))
    freq_dist.head(top_n).plot(kind='bar', figsize=(12, 6), title=f'Top {top_n} Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()

feedback_data = pd.read_csv(r"C:\Users\kbala\Downloads\reviews_data.csv")
top_n = int(input("Enter the number of top words to analyze: "))
analyze_feedback(feedback_data, top_n)


# Link for data set : https://www.kaggle.com/datasets/harshalhonde/starbucks-reviews-dataset
