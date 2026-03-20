async function predictNews() {
    const newsInput = document.getElementById('newsInput');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = predictBtn.querySelector('.btn-text');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('resultContainer');
    const resultLabel = document.getElementById('resultLabel');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const resultMsg = document.getElementById('resultMsg');

    const text = newsInput.value.trim();

    if (!text) {
        alert("Please enter some news text to analyze.");
        return;
    }

    // UI Loading State
    predictBtn.disabled = true;
    btnText.style.opacity = '0';
    loader.style.display = 'block';
    resultContainer.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ news_text: text })
        });

        const data = await response.json();

        if (response.ok) {
            // Update UI with results
            resultLabel.textContent = data.prediction;
            resultLabel.className = 'result-badge ' + (data.prediction === 'REAL' ? 'badge-real' : 'badge-fake');

            confidenceValue.textContent = data.confidence;
            confidenceFill.style.width = data.confidence + '%';

            if (data.prediction === 'REAL') {
                resultMsg.textContent = "This article appears to be credible.";
            } else {
                resultMsg.textContent = "Caution: This article shows signs of being unreliable.";
            }

            resultContainer.classList.remove('hidden');
        } else {
            alert(data.error || "Something went wrong.");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Failed to connect to the server.");
    } finally {
        predictBtn.disabled = false;
        btnText.style.opacity = '1';
        loader.style.display = 'none';
    }
}
