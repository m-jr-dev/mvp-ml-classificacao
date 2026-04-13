
const form = document.getElementById("prediction-form");
const resultCard = document.getElementById("result-card");
const resultLabel = document.getElementById("result-label");
const resultMeta = document.getElementById("result-meta");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    for (const key of Object.keys(payload)) {
        payload[key] = Number(payload[key]);
    }

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
        resultCard.hidden = false;
        resultLabel.textContent = "Falha na predição";
        resultMeta.textContent = data.detail || "Não foi possível processar a requisição.";
        return;
    }

    resultCard.hidden = false;
    resultLabel.textContent = data.label;
    resultMeta.textContent = `Classe numérica prevista: ${data.prediction}. Modelo carregado: ${data.selected_model}.`;
});
