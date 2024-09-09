import pandas as pd
import torch

def load_amazon_reviews(file_path):
    # Load the dataset with no header and assign correct column names
    df = pd.read_csv(file_path, header=None, names=['reviewerID', 'asin', 'overall', 'timestamp'])
    
    # Keep only the relevant columns (reviewerID, asin, overall)
    df = df[['reviewerID', 'asin', 'overall']]

    # Convert user and item ids to categorical values
    df['userID'] = df['reviewerID'].astype('category').cat.codes
    df['itemID'] = df['asin'].astype('category').cat.codes

    # Keep interactions (user, item, rating)
    interactions = df[['userID', 'itemID', 'overall']].values

    return interactions, len(df['userID'].unique()), len(df['itemID'].unique())

def build_edge_index(interactions, num_users):
    """
    Build the edge index for PyTorch Geometric from the interactions.
    
    Arguments:
    - interactions: A numpy array or dataframe of user-item interactions.
    - num_users: Number of unique users.
    
    Returns:
    - edge_index: A tensor with shape [2, num_interactions].
    """
    # Get user and item indices from the interactions
    user_indices = torch.tensor(interactions[:, 0], dtype=torch.long)
    item_indices = torch.tensor(interactions[:, 1], dtype=torch.long) + num_users  # Shift item indices by the number of users

    # Create edge index by stacking user-item pairs
    edge_index = torch.stack([user_indices, item_indices], dim=0)

    return edge_index
