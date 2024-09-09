import torch
import torch.optim as optim
from backend.utils import bpr_loss  # Correct relative import for utils
from backend.model import LightGCNAttention  # Model import
from backend.data_loader import load_amazon_reviews, build_edge_index  # Import both functions from data_loader
import yaml
import os

def train_lightgcn(model, optimizer, edge_index, interactions, num_users, num_items, epochs, model_path):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        user_embeddings, item_embeddings = model(edge_index)
        loss = bpr_loss(user_embeddings, item_embeddings, interactions, num_users)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # Save the model after training
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    # Load configuration
    with open("backend/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Load data
    interactions, num_users, num_items = load_amazon_reviews(config['file_path'])
    edge_index = build_edge_index(interactions, num_users)

    # Initialize the model and train
    model = LightGCNAttention(num_users, num_items, config['embedding_dim'], config['num_layers'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Train and save the model
    train_lightgcn(model, optimizer, edge_index, interactions, num_users, num_items, config['epochs'], config['model_save_path'])
