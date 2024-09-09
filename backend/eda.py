import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(file_path, save_path='ratings_distribution.png'):
    # Load the dataset and assign column names if necessary
    df = pd.read_csv(file_path, header=None, names=['reviewerID', 'asin', 'overall', 'timestamp'])

    # Print column names to debug if 'overall' is missing or named differently
    print("Column names in the dataset:", df.columns)
    
    # Check if 'overall' exists and perform EDA, or adjust to use the correct column
    if 'overall' in df.columns:
        # Plot the distribution of ratings
        plt.figure(figsize=(8, 6))
        sns.histplot(df['overall'], kde=False, bins=5)
        plt.title('Ratings Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')

        # Save the plot to a file
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.show()
    else:
        print("'overall' column not found in dataset. Check the dataset for the correct column name.")

if __name__ == '__main__':
    file_path = '../data/amazon_reviews.csv'  # Adjust path as per your configuration
    save_path = 'ratings_distribution.png'  # Specify where to save the plot
    perform_eda(file_path, save_path)
