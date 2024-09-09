from fastapi import FastAPI
import torch
from model import LightGCNAttention
from data_loader import load_amazon_reviews, build_edge_index
from pydantic import BaseModel

app = FastAPI()

# Load pre-trained model and data
file_path = "../data/amazon_reviews.csv"
interactions, num_users, num_items = load_amazon_reviews(file_path)
edge_index = build_edge_index(interactions, num_users)

# Initialize model and load saved state
embedding_dim = 64
num_layers = 3
model_path = "saved_models/lightgcn_model.pth"
model = LightGCNAttention(num_users, num_items, embedding_dim, num_layers)
model.load_state_dict(torch.load(model_path))
model.eval()

class UserRequest(BaseModel):
    user_id: int
    top_k: int = 10

@app.post("/recommend/")
async def recommend(req: UserRequest):
    user_embeddings, item_embeddings = model(edge_index)
    user_embedding = user_embeddings[req.user_id]
    
    scores = torch.matmul(user_embedding, item_embeddings.T)
    _, recommended_items = torch.topk(scores, req.top_k)
    
    return {"recommended_items": recommended_items.tolist()}
