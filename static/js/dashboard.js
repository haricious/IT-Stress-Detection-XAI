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

    const setBusy = (isBusy) => {
        if (!submitBtn) return;
        submitBtn.disabled = isBusy;
        const label = submitBtn.querySelector(".btn-label");
        if (label) {
            label.textContent = isBusy ? "Neural Engine Computing..." : "Analyze with XAI";
        }
        submitBtn.style.opacity = isBusy ? "0.6" : "1";
    };

    const applyMeter = (status) => {
        if (!resultMeter) return;
        const widths = { high: 88, warning: 55, stable: 32 };
        resultMeter.style.width = '0%';
        setTimeout(() => {
            resultMeter.className = `meter-fill meter-${status}`;
            resultMeter.style.width = `${widths[status] || 0}%`;
        }, 50);
    };

    const updateUI = (data) => {
        // STEP A: Show text results instantly
        resultCard.style.display = "block";
        resultCard.classList.add("is-visible");

        resultBadge.className = `badge badge-${data.status}`;
        resultBadge.textContent = data.status.toUpperCase();
        resultMessage.className = `result-message msg-${data.status}`;
        resultMessage.textContent = data.message;
        resultMeta.textContent = data.detail;

        applyMeter(data.status);

        // STEP B: Handle XAI Plot (Fades in when ready)
        if (data.xai_plot && xaiPlot) {
            xaiPlot.onload = () => {
                xaiContainer.style.display = "block";
                xaiContainer.classList.remove("hidden");
                setBusy(false); 
                console.log("-> [XAI] Plot Loaded.");
            };
            xaiPlot.src = data.xai_plot;
        } else {
            setBusy(false);
        }

        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    };

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        formError.textContent = "";

        const payload = {};
        const fieldIds = ["work_hours", "sleep_hours", "tech_usage", "physical_activity", "social_gap", "deadline_pressure"];
        
        for (const id of fieldIds) {
            const val = parseFloat(document.getElementById(id).value);
            if (isNaN(val)) {
                formError.textContent = "All inputs are required.";
                return;
            }
            payload[id] = val;
        }

        setBusy(true);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Server Error");
            const data = await response.json();
            updateUI(data);

        } catch (err) {
            formError.textContent = "System busy. Please retry.";
            setBusy(false);
        }
    });

    // Theme & Sliders
    document.getElementById("themeBtn")?.addEventListener("click", () => {
        const next = root.dataset.theme === "dark" ? "light" : "dark";
        root.dataset.theme = next;
    });

    document.getElementById("fillSampleBtn")?.addEventListener("click", () => {
        const vals = [9.5, 4.5, 11, 10, 8, 9];
        fieldIds.forEach((id, i) => {
            document.getElementById(id).value = vals[i];
            const out = document.getElementById(`${id}_out`);
            if (out) out.textContent = vals[i];
        });
    });
})();