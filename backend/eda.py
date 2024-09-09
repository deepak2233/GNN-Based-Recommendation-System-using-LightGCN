import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(file_path):
    df = pd.read_csv(file_path)
    
    # Ratings distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['overall'], kde=False, bins=5)
    plt.title('Ratings Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()
    
    # Most frequent reviewers
    top_reviewers = df['reviewerID'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_reviewers.index, y=top_reviewers.values)
    plt.title('Top 10 Reviewers')
    plt.xticks(rotation=90)
    plt.show()

if __name__ == '__main__':
    file_path = '../data/amazon_reviews.csv'
    perform_eda(file_path)
