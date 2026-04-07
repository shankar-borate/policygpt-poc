(function () {
    const POLL_INTERVAL_MS = 10_000;

    const elements = {
        root: document.getElementById("usage-widget"),
        model: document.getElementById("usage-model"),
        inputTokens: document.getElementById("usage-input-tokens"),
        outputTokens: document.getElementById("usage-output-tokens"),
        totalPrice: document.getElementById("usage-total-price"),
        meta: document.getElementById("usage-meta"),
        lastRequest: document.getElementById("usage-last-request"),
        lastResponse: document.getElementById("usage-last-response"),
        totalRequest: document.getElementById("usage-total-request"),
        historyToggle: document.getElementById("usage-history-toggle"),
        historyPanel: document.getElementById("usage-history-panel"),
        historyBody: document.getElementById("usage-history-body"),
    };

    if (!elements.root) {
        return;
    }

    let pollHandle = null;
    let requestInFlight = false;
    let historyExpanded = false;

    function formatTokens(value) {
        return new Intl.NumberFormat("en-US").format(Number(value || 0));
    }

    function formatInr(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "Unavailable";
        }
        return `Rs ${Number(value).toFixed(6)}`;
    }

    function formatDuration(value) {
        const durationMs = Number(value || 0);
        if (!Number.isFinite(durationMs) || durationMs <= 0) {
            return "0 ms";
        }
        if (durationMs >= 1000) {
            return `${(durationMs / 1000).toFixed(2)} s`;
        }
        return `${Math.round(durationMs)} ms`;
    }

    function renderHistory(entries) {
        if (!elements.historyBody) {
            return;
        }

        if (!Array.isArray(entries) || !entries.length) {
            elements.historyBody.innerHTML = `
                <tr>
                    <td class="usage-history-empty" colspan="5">No request history yet.</td>
                </tr>
            `;
            return;
        }

        elements.historyBody.innerHTML = entries
            .map((entry) => `
                <tr>
                    <td>${String(entry?.request_id || "Unavailable")}</td>
                    <td>${formatInr(entry?.input_cost_inr)}</td>
                    <td>${formatInr(entry?.output_cost_inr)}</td>
                    <td>${formatInr(entry?.total_cost_inr)}</td>
                    <td>${formatDuration(entry?.duration_ms)}</td>
                </tr>
            `)
            .join("");
    }

    function setHistoryExpanded(nextValue) {
        historyExpanded = Boolean(nextValue);
        if (elements.historyPanel) {
            elements.historyPanel.hidden = !historyExpanded;
        }
        if (elements.historyToggle) {
            elements.historyToggle.setAttribute("aria-expanded", historyExpanded ? "true" : "false");
            elements.historyToggle.textContent = historyExpanded
                ? "Hide request/response cost history"
                : "View request/response cost history";
        }
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
        const exchangeRate = Number(payload?.exchange_rate_usd_to_inr || 0);
        const rateCopy = exchangeRate > 0
            ? `Rate: Rs ${exchangeRate.toFixed(2)} per USD. `
            : "";

        if (!loadedAt) {
            return `Cumulative since startup. ${rateCopy}${statusLabel}. Refresh every 10 seconds.`;
        }

        return `Cumulative since startup. ${rateCopy}${statusLabel} at ${loadedAt}. Refresh every 10 seconds.`;
    }

    function renderUsage(payload) {
        elements.model.textContent = payload?.display_name || payload?.model_name || "LLM usage unavailable";
        elements.inputTokens.textContent = formatTokens(payload?.input_tokens);
        elements.outputTokens.textContent = formatTokens(payload?.output_tokens);
        elements.totalPrice.textContent = formatInr(payload?.total_cost_inr);
        elements.meta.textContent = formatSnapshotStatus(payload);
        elements.lastRequest.textContent = `Last request: ${formatInr(payload?.last_input_cost_inr)}`;
        elements.lastResponse.textContent = `Last response: ${formatInr(payload?.last_output_cost_inr)}`;
        elements.totalRequest.textContent = `Last total cost: ${formatInr(payload?.last_total_cost_inr)}`;
        renderHistory(payload?.history);
    }

    function renderError(message) {
        elements.model.textContent = "LLM usage unavailable";
        elements.inputTokens.textContent = "0";
        elements.outputTokens.textContent = "0";
        elements.totalPrice.textContent = "Unavailable";
        elements.meta.textContent = message || "Could not read usage details.";
        elements.lastRequest.textContent = "Last request: Unavailable";
        elements.lastResponse.textContent = "Last response: Unavailable";
        elements.totalRequest.textContent = "Last total cost: Unavailable";
        renderHistory([]);
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

    if (elements.historyToggle) {
        elements.historyToggle.addEventListener("click", () => {
            setHistoryExpanded(!historyExpanded);
        });
    }

    setHistoryExpanded(false);
    refreshUsage().finally(scheduleNextPoll);
})();
