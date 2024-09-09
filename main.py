import argparse
import yaml
from backend.train import train_lightgcn
from backend.data_loader import load_amazon_reviews, build_edge_index
from backend.eda import perform_eda

def main():
    # Argument parser to choose which action to perform
    parser = argparse.ArgumentParser(description="LightGCN pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['eda', 'train', 'serve'], 
                        help='Mode to run: eda, train, or serve the model.')
    args = parser.parse_args()

    # Load configuration from config.yaml
    with open("backend/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Perform EDA
    if args.mode == 'eda':
        print("Performing EDA on the dataset...")
        perform_eda(config['file_path'])

    # Train the model
    elif args.mode == 'train':
        print("Training the LightGCN model...")
        interactions, num_users, num_items = load_amazon_reviews(config['file_path'])
        edge_index = build_edge_index(interactions, num_users)

        # Initialize the model and train
        from backend.model import LightGCNAttention
        import torch.optim as optim
        model = LightGCNAttention(num_users, num_items, config['embedding_dim'], config['num_layers'])
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Train and save the model
        train_lightgcn(model, optimizer, edge_index, interactions, num_users, num_items, config['epochs'], config['model_save_path'])

    # Serve the model through FastAPI
    elif args.mode == 'serve':
        print("Serving the model with FastAPI...")
        import os
        os.system("uvicorn backend.app:app --reload")

if __name__ == "__main__":
    main()
