document.getElementById('reviewForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent default form submission

    const review = document.getElementById('review').value.trim();

    // Prevent empty submissions
    if (!review) {
        document.getElementById('analysisResult').innerText = "Please enter a review.";
        return;
    }

    try {
        // Fetch API response from FastAPI backend
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ review }),
        });

        if (!response.ok) {
            throw new Error('Server error. Try again.');
        }

        const data = await response.json();
        document.getElementById('analysisResult').innerText = `Sentiment: ${data.sentiment}`;
    } catch (error) {
        document.getElementById('analysisResult').innerText = `Error: ${error.message}`;
    }
});
