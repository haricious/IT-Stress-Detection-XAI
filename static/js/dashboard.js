(() => {
    const form = document.getElementById("stressForm");
    if (!form) return;

    const root = document.documentElement;
    const submitBtn = document.getElementById("submitBtn");
    const formError = document.getElementById("formError");
    const resultCard = document.getElementById("resultCard");
    const resultBadge = document.getElementById("resultBadge");
    const resultMessage = document.getElementById("resultMessage");
    const resultMeta = document.getElementById("resultMeta");
    const resultMeter = document.getElementById("resultMeter");
    const xaiContainer = document.getElementById("xaiContainer");
    const xaiPlot = document.getElementById("xaiPlot");

    const themeBtn = document.getElementById("themeBtn");
    const fillSampleBtn = document.getElementById("fillSampleBtn");
    const resetBtn = document.getElementById("resetBtn");

    const fields = [
        { id: "work_hours", min: 0, max: 24 },
        { id: "sleep_hours", min: 0, max: 16 },
        { id: "tech_usage", min: 0, max: 24 },
        { id: "physical_activity", min: 0, max: 300 },
        { id: "social_gap", min: 1, max: 10 },
        { id: "deadline_pressure", min: 1, max: 10 },
    ];

    const setBusy = (isBusy) => {
        if (!submitBtn) return;
        submitBtn.disabled = isBusy;
        const label = submitBtn.querySelector(".btn-label");
        if (label) label.textContent = isBusy ? "Neural Engine Computing..." : "Analyze with XAI";
        submitBtn.style.opacity = isBusy ? "0.7" : "1";
        submitBtn.style.cursor = isBusy ? "not-allowed" : "pointer";
    };

    const applyMeter = (status) => {
        if (!resultMeter) return;
        const widths = { high: 86, warning: 55, stable: 34 };
        resultMeter.style.width = '0%';
        setTimeout(() => {
            resultMeter.className = `meter-fill meter-${status}`;
            resultMeter.style.width = `${widths[status] || 0}%`;
        }, 100);
    };

    const updateUI = (data) => {
        // 1. Reveal Result Card
        resultCard.style.display = "block";
        setTimeout(() => resultCard.classList.add("is-visible"), 10);

        // 2. Update Basic Info
        resultBadge.className = `badge badge-${data.status}`;
        resultBadge.textContent = data.status.toUpperCase();
        resultMessage.className = `result-message msg-${data.status}`;
        resultMessage.textContent = data.message;
        resultMeta.textContent = data.detail;

        applyMeter(data.status);

        // 3. Handle XAI Plot (The slow part)
        if (data.xai_plot && xaiPlot) {
            // This event fires only when the browser has finished downloading the plot image
            xaiPlot.onload = () => {
                xaiContainer.classList.remove("hidden");
                xaiContainer.style.display = "block";
                setBusy(false); // Task complete
                resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
            };
            xaiPlot.src = data.xai_plot;
        } else {
            setBusy(false);
        }
    };

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        formError.textContent = "";

        // Simple validation
        const payload = {};
        for (const f of fields) {
            const val = parseFloat(document.getElementById(f.id).value);
            if (isNaN(val)) {
                formError.textContent = "Please fill all fields.";
                return;
            }
            payload[f.id] = val;
        }

        setBusy(true);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Neural Engine Error");
            const data = await response.json();
            updateUI(data);

        } catch (err) {
            formError.textContent = "Server is busy. Try again in 5 seconds.";
            setBusy(false);
        }
    });

    // Utilities
    themeBtn?.addEventListener("click", () => {
        const next = root.dataset.theme === "dark" ? "light" : "dark";
        root.dataset.theme = next;
        localStorage.setItem("theme", next);
    });

    fillSampleBtn?.addEventListener("click", () => {
        const samples = [10.5, 4, 12, 15, 8, 9]; 
        fields.forEach((f, i) => {
            document.getElementById(f.id).value = samples[i];
            const out = document.getElementById(`${f.id}_out`);
            if (out) out.textContent = samples[i];
        });
    });

    resetBtn?.addEventListener("click", () => {
        form.reset();
        resultCard.classList.remove("is-visible");
        xaiContainer.style.display = "none";
        xaiContainer.classList.add("hidden");
    });

    fields.filter(f => f.max === 10).forEach(f => {
        const input = document.getElementById(f.id);
        const out = document.getElementById(`${f.id}_out`);
        input?.addEventListener("input", () => out.textContent = input.value);
    });

    root.dataset.theme = localStorage.getItem("theme") || "light";
})();