(function () {
    const POLL_INTERVAL_MS = 10_000;

    const elements = {
        root: document.getElementById("usage-widget"),
        model: document.getElementById("usage-model"),
        inputTokens: document.getElementById("usage-input-tokens"),
        outputTokens: document.getElementById("usage-output-tokens"),
        totalPrice: document.getElementById("usage-total-price"),
        meta: document.getElementById("usage-meta"),
    };

    if (!elements.root) {
        return;
    }

    let pollHandle = null;
    let requestInFlight = false;

    function formatTokens(value) {
        return new Intl.NumberFormat("en-US").format(Number(value || 0));
    }

    function formatUsd(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "Unavailable";
        }
        return `$${Number(value).toFixed(6)}`;
    }

    function formatSnapshotStatus(payload) {
        const sourceStatus = String(payload?.source_status || "unavailable").toLowerCase();
        const statusLabel = sourceStatus === "live"
            ? "live pricing snapshot"
            : sourceStatus === "fallback"
                ? "fallback pricing snapshot"
                : "pricing unavailable";

        const loadedAt = payload?.pricing_loaded_at
            ? new Date(payload.pricing_loaded_at).toLocaleString()
            : "";

        if (!loadedAt) {
            return `Cumulative since startup. ${statusLabel}. Refresh every 10 seconds.`;
        }

        return `Cumulative since startup. ${statusLabel} at ${loadedAt}. Refresh every 10 seconds.`;
    }

    function renderUsage(payload) {
        elements.model.textContent = payload?.display_name || payload?.model_name || "LLM usage unavailable";
        elements.inputTokens.textContent = formatTokens(payload?.input_tokens);
        elements.outputTokens.textContent = formatTokens(payload?.output_tokens);
        elements.totalPrice.textContent = formatUsd(payload?.total_cost_usd);
        elements.meta.textContent = formatSnapshotStatus(payload);
    }

    function renderError(message) {
        elements.model.textContent = "LLM usage unavailable";
        elements.inputTokens.textContent = "0";
        elements.outputTokens.textContent = "0";
        elements.totalPrice.textContent = "Unavailable";
        elements.meta.textContent = message || "Could not read usage details.";
    }

    async function refreshUsage() {
        if (requestInFlight) {
            return;
        }

        requestInFlight = true;
        try {
            const response = await fetch("/api/usage", { method: "GET" });
            if (!response.ok) {
                throw new Error("Usage endpoint returned an error.");
            }
            const payload = await response.json();
            renderUsage(payload);
        } catch (error) {
            renderError(error.message || "Could not read usage details.");
        } finally {
            requestInFlight = false;
        }
    }

    function scheduleNextPoll() {
        window.clearTimeout(pollHandle);
        pollHandle = window.setTimeout(async () => {
            await refreshUsage();
            scheduleNextPoll();
        }, POLL_INTERVAL_MS);
    }

    window.addEventListener("beforeunload", () => {
        window.clearTimeout(pollHandle);
    });

    refreshUsage().finally(scheduleNextPoll);
})();
