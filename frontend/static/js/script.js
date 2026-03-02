document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((val, key) => data[key] = Number(val));

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        const result = await response.json();
        console.log(result)

        const showResult = document.getElementById("result")
        showResult.style.display = 'block'
        showResult.innerHTML = `
            <h2 class="text-primary mb-3 text-center">Prediction Result</h2>

            <div class="text-center mb-3">
                <h4 class="fw-bold ${result.prediction === 1 ? "text-danger" : "text-success"}">
                    ${result.prediction === 1 ? "Leave" : "Stay"}
                </h4>
            </div>
        `;
    } catch (error) {
        console.error("Prediction error:", error);
        document.getElementById("result").innerHTML =
            `<p class="text-danger">Prediction failed. Please try again.</p>`;
    }
});
