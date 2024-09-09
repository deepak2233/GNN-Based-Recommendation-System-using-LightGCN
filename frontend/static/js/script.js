document.getElementById('getRecommendations').addEventListener('click', async () => {
    const userId = document.getElementById('userId').value;
    const topK = document.getElementById('topK').value;
    const recommendationsDiv = document.getElementById('recommendations');
    
    recommendationsDiv.innerHTML = '<p>Loading...</p>';
    
    const response = await fetch('http://localhost:8000/recommend/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            user_id: parseInt(userId),
            top_k: parseInt(topK)
        }),
    });
    
    const data = await response.json();
    const items = data.recommended_items;
    
    recommendationsDiv.innerHTML = '<h3>Recommended Items:</h3>';
    items.forEach((item, index) => {
        recommendationsDiv.innerHTML += `<p>${index + 1}. Item ID: ${item}</p>`;
    });
});
