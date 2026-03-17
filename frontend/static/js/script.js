document.getElementById('fetchFeaturesBtn').addEventListener('click', async () => {
    const employeeId = document.getElementById('employeeIdInput').value.trim();
    const statusEl = document.getElementById('fetchStatus');

    if (!employeeId) {
        statusEl.innerHTML = `<span class="text-warning">⚠️ Please enter an Employee ID.</span>`;
        return;
    }

    statusEl.innerHTML = `<span class="text-muted">⏳ Fetching features...</span>`;

    try {
        const res = await fetch(`/features/${employeeId}`);
        const response = await res.json();
        console.log(response)

        // Set hidden employee_id
        document.getElementById('employeeIdHidden').value = employeeId;

        // Auto-fill form fields
        const fieldMap = {
            years_at_company:      '[name="years_at_company"]',
            company_tenure:        '[name="company_tenure"]',
            number_of_promotions:  '[name="number_of_promotions"]',
            number_of_dependents:  '[name="number_of_dependents"]',
            overtime:              '[name="overtime"]',
            remote_work:           '[name="remote_work"]',
            performance_rating:    '[name="performance_rating"]',
            education_level:       '[name="education_level"]',
            job_level:             '[name="job_level"]',
            company_size:          '[name="company_size"]',
            company_reputation:    '[name="company_reputation"]',
            overall_satisfaction:  '[name="overall_satisfaction"]',
            opportunities:         '[name="opportunities"]',
            annual_income:         '[name="annual_income"]',
            age_group:             '[name="age_group"]',
        };
        for (const [key, selector] of Object.entries(fieldMap)) {
            const el = document.querySelector(selector);
            if (el && response['features'][key] !== undefined && response['features'][key] !== null) {
                let value = response['features'][key]
                if(["years_at_company", "company_tenure"].includes(key)) {
                    value = Math.round(Number(value))
                }
                el.value = value;
            }
        }

        statusEl.innerHTML = `<span class="text-success">✅ Features loaded for Employee ID: <strong>${employeeId}</strong></span>`;

    } catch (err) {
        statusEl.innerHTML = `<span class="text-danger">❌ Error: ${err.message}</span>`;
    }
});


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



document.getElementById('clearBtn').addEventListener('click', () => {
    document.getElementById('prediction-form').reset();
    document.getElementById('employeeIdInput').value = '';
    document.getElementById('employeeIdHidden').value = '';
    document.getElementById('fetchStatus').innerHTML = '';
    document.getElementById('result').innerHTML = '';
});