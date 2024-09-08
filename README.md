
# GNN-Based Recommendation System using LightGCN

This project implements a **Graph Neural Network (GNN)** based recommendation system using **LightGCN** (Light Graph Convolution Network) for personalized product recommendations. The backend is powered by **FastAPI**, while the frontend is a simple HTML/JS interface to query the recommendations.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [How to Run](#how-to-run)
  - [1. Run EDA](#1-run-eda)
  - [2. Train the Model](#2-train-the-model)
  - [3. Serve the API](#3-serve-the-api)
  - [4. Serve the Frontend](#4-serve-the-frontend)
- [Endpoints](#endpoints)
- [Technologies Used](#technologies-used)


---

## Project Overview

This project aims to build an end-to-end recommendation system using GNNs, specifically **LightGCN**, which is tailored for large-scale recommendation tasks. The system uses user-item interactions to predict and recommend products to users.

The **backend** provides API endpoints to serve recommendations using FastAPI, and the **frontend** enables users to input a user ID and retrieve recommendations through a simple interface.

---
## Project Structure

``` bash

gnn_recommendation/
├── backend                     # Backend logic (FastAPI, training, data processing)
│   ├── __init__.py             # Makes backend a module
│   ├── app.py                  # FastAPI main application
│   ├── train.py                # Training script for LightGCN
│   ├── model.py                # LightGCN model definition
│   ├── data_loader.py          # Data loading and preprocessing for Amazon Reviews
│   ├── utils.py                # Utility functions (e.g., BPR loss)
│   ├── config.yaml             # Configuration file (hyperparameters, paths)
│   ├── eda.py                  # Exploratory Data Analysis (EDA)
│   ├── requirements.txt        # Python dependencies for backend
│   ├── saved_models            # Directory to save/load the trained LightGCN models
│   └── tests                   # Unit and integration tests
│       └── test_app.py         # Tests for FastAPI application
├── frontend                    # Frontend (HTML, CSS, JavaScript)
│   ├── index.html              # Main HTML page for recommendations UI
│   ├── static
│   │   ├── css
│   │   │   └── style.css       # Styles for the HTML page
│   │   └── js
│   │       └── script.js       # JavaScript to interact with FastAPI
├── data                        # Dataset folder (contains CSV files for interactions)
│   └── amazon_reviews.csv      # Amazon product review dataset
├── main.py                     # Main entry point for running the pipeline
└── README.md                   # Documentation and instructions for the project
```

---

## Setup and Installation

### Backend Setup

1. **Clone the repository**:
 ```
   git clone https://github.com/yourusername/gnn_recommendation.git
   cd gnn_recommendation

```

2. **Install backend dependencies**:
   Install all necessary Python packages by running:
   
```
   pip install -r backend/requirements.txt
  
```

3. **Download the dataset**:
   Place the Amazon Product Reviews dataset (\`amazon_reviews.csv\`) inside the \`data/\` folder.

### Frontend Setup

No special setup is required for the frontend. It's a simple HTML/CSS/JS static page that will interact with the FastAPI backend.

---
## How to Run

### 1. Run EDA
To explore the dataset and generate visual insights, run the following command:

```
python3 main.py --mode eda

```

This will analyze and visualize basic statistics about the dataset.

### 2. Train the Model
To train the LightGCN model on the Amazon dataset, run:

``` bash
python3 main.py --mode train

```

The model will be saved in \`backend/saved_models/lightgcn_model.pth\` after training is completed.

### 3. Serve the API
Once the model is trained, you can serve the FastAPI backend to provide recommendations:

```
python3 main.py --mode serve

```

This will start a FastAPI server at \`http://localhost:8000/\`.

### 4. Serve the Frontend
There are two ways to serve the frontend:

#### Method 1: Using Python's Simple HTTP Server
1. Navigate to the \`frontend/\` directory:
   
```
   cd frontend

```

2. Run Python's built-in HTTP server:

```
   python3 -m http.server 8001

```

3. Open your browser and visit \`http://localhost:8001/\` to interact with the frontend.

---
#### Method 2: Using Live Server in VS Code
1. Install the **Live Server** extension in **Visual Studio Code**.
2. Right-click on \`index.html\` and choose **Open with Live Server**.
---

### Endpoints

The backend exposes a POST endpoint to fetch recommendations.

- **POST** \`/recommend/\`
  - **Payload**:
    
```
    {
      "user_id": <int>,    # User ID for whom you want recommendations
      "top_k": <int>       # Number of recommendations (top K items)
    }

```

  - **Response**:

```
    {
      "recommended_items": [<list of item IDs>]
    }

```
---
## Technologies Used

- **Graph Neural Networks (GNN)**: LightGCN model implemented in PyTorch Geometric for recommendation tasks.
- **Backend**: FastAPI for building and serving REST API.
- **Frontend**: HTML, CSS, JavaScript (AJAX to call the FastAPI backend).
- **Data Processing**: Pandas for data loading and manipulation.
- **Model Training**: PyTorch for training the LightGCN model.


