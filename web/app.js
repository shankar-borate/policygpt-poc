const state = {
    health: null,
    domain: null,
    threads: [],
    activeThread: null,
    pendingPrompt: "",
    loading: false,
    uiError: "",
    pollHandle: null,
    healthRequestInFlight: false,
};

const elements = {
    threadList: document.getElementById("thread-list"),
    chatTitle: document.getElementById("chat-title"),
    assistantLabel: document.getElementById("assistant-label"),
    heroEyebrow: document.getElementById("hero-eyebrow"),
    heroDescription: document.getElementById("hero-description"),
    promptRow: document.getElementById("prompt-row"),
    messages: document.getElementById("messages"),
    hero: document.getElementById("hero"),
    corpusSummary: document.getElementById("corpus-summary"),
    statusPill: document.getElementById("status-pill"),
    statusMeta: document.getElementById("status-meta"),
    progressBar: document.getElementById("progress-bar"),
    progressCopy: document.getElementById("progress-copy"),
    composerForm: document.getElementById("composer-form"),
    composerInput: document.getElementById("composer-input"),
    sendButton: document.getElementById("send-btn"),
    newChatButton: document.getElementById("new-chat-btn"),
    resetChatButton: document.getElementById("reset-chat-btn"),
};

function escapeHtml(text) {
    return String(text || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function renderInlineMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer noopener">$1</a>');
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return html;
}

function renderMarkdownBlock(text) {
    const trimmedBlock = text.trim();
    if (!trimmedBlock) {
        return "";
    }

    const lines = trimmedBlock.split(/\r?\n/);
    const html = [];
    let listType = null;
    let paragraph = [];

    const closeList = () => {
        if (listType) {
            html.push(`</${listType}>`);
            listType = null;
        }
    };

    const flushParagraph = () => {
        if (paragraph.length) {
            html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
            paragraph = [];
        }
    };

    for (const rawLine of lines) {
        const line = rawLine.trim();

        if (!line) {
            flushParagraph();
            closeList();
            continue;
        }

        if (/^#{1,3}\s/.test(line)) {
            flushParagraph();
            closeList();
            const level = Math.min(line.match(/^#+/)[0].length, 3);
            const content = line.replace(/^#{1,3}\s+/, "");
            html.push(`<h${level}>${renderInlineMarkdown(content)}</h${level}>`);
            continue;
        }

        if (/^\d+\.\s/.test(line)) {
            flushParagraph();
            if (listType !== "ol") {
                closeList();
                html.push("<ol>");
                listType = "ol";
            }
            html.push(`<li>${renderInlineMarkdown(line.replace(/^\d+\.\s+/, ""))}</li>`);
            continue;
        }

        if (/^[-*]\s/.test(line)) {
            flushParagraph();
            if (listType !== "ul") {
                closeList();
                html.push("<ul>");
                listType = "ul";
            }
            html.push(`<li>${renderInlineMarkdown(line.replace(/^[-*]\s+/, ""))}</li>`);
            continue;
        }

        if (/^>\s?/.test(line)) {
            flushParagraph();
            closeList();
            html.push(`<blockquote>${renderInlineMarkdown(line.replace(/^>\s?/, ""))}</blockquote>`);
            continue;
        }

        paragraph.push(line);
    }

    flushParagraph();
    closeList();
    return html.join("");
}

function renderMarkdown(text) {
    if (!text) {
        return "";
    }

    const parts = [];
    // Match code fences OR block-level HTML elements — both bypass escaping.
    const blockPattern = /```([\s\S]*?)```|(<(?:table|div|figure|details)[^>]*>[\s\S]*?<\/(?:table|div|figure|details)>)/gi;
    let lastIndex = 0;
    let match;

    while ((match = blockPattern.exec(text)) !== null) {
        parts.push(renderMarkdownBlock(text.slice(lastIndex, match.index)));
        if (match[1] !== undefined) {
            // Code fence
            parts.push(`<pre><code>${escapeHtml(match[1].trim())}</code></pre>`);
        } else {
            // HTML block — pass through as-is so tables render correctly
            parts.push(match[2]);
        }
        lastIndex = match.index + match[0].length;
    }

    parts.push(renderMarkdownBlock(text.slice(lastIndex)));
    return parts.join("");
}

function autosizeComposer() {
    elements.composerInput.style.height = "auto";
    elements.composerInput.style.height = `${Math.min(elements.composerInput.scrollHeight, 180)}px`;
}

function setStatus(health) {
    const status = health?.status || "starting";
    const progress = health?.progress || {};
    const totalFiles = progress.total_files || 0;
    const processedFiles = progress.processed_files || 0;
    const percent = status === "ready"
        ? 100
        : Math.max(0, Math.min(Number(progress.percent || 0), 100));

    elements.statusPill.textContent = status.replaceAll("_", " ");
    elements.statusPill.className = `status-pill status-${status}`;
    elements.progressBar.style.width = `${percent}%`;

    if (status === "ready") {
        elements.statusMeta.textContent = `${health.document_count} documents indexed from ${health.document_folder}`;
        elements.progressCopy.textContent = totalFiles
            ? `${processedFiles}/${totalFiles} policy files completed.`
            : "Policy index is ready.";
        return;
    }

    if (status === "in_progress") {
        elements.statusMeta.textContent = totalFiles
            ? `Indexing ${processedFiles} of ${totalFiles} files`
            : `Scanning ${health.document_folder}`;
        elements.progressCopy.textContent = progress.current_file
            ? `Current file: ${progress.current_file}`
            : "Preparing the next policy file...";
        return;
    }

    if (status === "error") {
        elements.statusMeta.textContent = health.error || "Initialization failed.";
        elements.progressCopy.textContent = totalFiles
            ? `Stopped after ${processedFiles} of ${totalFiles} files.`
            : "Indexing could not start.";
        return;
    }

    elements.statusMeta.textContent = "Preparing the policy index...";
    elements.progressCopy.textContent = "Waiting for indexing updates...";
}

function renderThreadList() {
    const activeThreadId = state.activeThread?.thread_id;

    if (!state.threads.length) {
        const emptyMessage = state.health?.status === "ready"
            ? "No conversations yet. Start with a new chat."
            : "Conversations appear after indexing is ready.";
        elements.threadList.innerHTML = `<p class="thread-empty">${emptyMessage}</p>`;
        return;
    }

    elements.threadList.innerHTML = state.threads
        .map((thread) => {
            const activeClass = thread.thread_id === activeThreadId ? "active" : "";
            const preview = thread.preview || "No messages yet.";
            return `
                <button class="thread-button ${activeClass}" type="button" data-thread-id="${thread.thread_id}">
                    <p class="thread-title">${escapeHtml(thread.title)}</p>
                    <p class="thread-preview">${escapeHtml(preview)}</p>
                </button>
            `;
        })
        .join("");

    elements.threadList.querySelectorAll("[data-thread-id]").forEach((button) => {
        button.addEventListener("click", async () => {
            const threadId = button.getAttribute("data-thread-id");
            if (!threadId || threadId === state.activeThread?.thread_id) {
                return;
            }
            await loadThread(threadId);
        });
    });
}

const RELATED_MARKER = "**You might also ask:**";

function splitRelatedQuestions(text) {
    const markerIdx = text.indexOf(RELATED_MARKER);
    if (markerIdx === -1) return { mainText: text, questions: [] };

    const mainText = text.slice(0, markerIdx).trimEnd();
    const rest = text.slice(markerIdx + RELATED_MARKER.length);
    const questions = rest
        .split(/\r?\n/)
        .map((line) => line.trim().match(/^[-*]\s+(.+)/))
        .filter(Boolean)
        .map((m) => m[1].trim());

    return { mainText, questions };
}

function renderRelatedQuestions(questions) {
    if (!questions || questions.length === 0) return "";
    const chips = questions.map(
        (q) => `<button class="related-chip" type="button" data-question="${escapeHtml(q)}">${escapeHtml(q)}</button>`
    );
    return `<div class="related-refs"><span class="source-refs-label">You might also ask</span><div class="related-chips">${chips.join("")}</div></div>`;
}

function renderSources(sources) {
    if (!sources || sources.length === 0) return "";

    const seen = new Set();
    const unique = sources.filter((s) => {
        if (seen.has(s.document_url)) return false;
        seen.add(s.document_url);
        return true;
    });

    const chips = unique.map((src) => {
        const sectionWords = (src.section_title || "").trim().split(/\s+/).filter(Boolean);
        const isGeneric = sectionWords.length === 0 || src.section_title.toLowerCase() === "introduction";
        const shortLabel = isGeneric
            ? src.file_name.replace(/\.[^/.]+$/, "")
            : sectionWords.slice(0, 4).join(" ") + (sectionWords.length > 4 ? "\u2026" : "");
        const tooltip = src.section_title && !isGeneric
            ? `${src.document_title} \u203a ${src.section_title}`
            : src.document_title;

        return `<a class="source-chip" href="${escapeHtml(src.document_url)}" target="_blank" rel="noreferrer noopener" data-tooltip="${escapeHtml(tooltip)}">${escapeHtml(shortLabel)}</a>`;
    });

    return `<div class="source-refs"><span class="source-refs-label">Sources</span><div class="source-chips">${chips.join("")}</div></div>`;
}

function renderMessages() {
    const savedMessages = state.activeThread?.messages || [];
    const hasPending = Boolean(state.pendingPrompt);
    const hasMessages = savedMessages.length > 0 || hasPending;

    elements.hero.classList.toggle("hidden", hasMessages);

    if (!hasMessages) {
        elements.messages.innerHTML = "";
        return;
    }

    const lastAssistantIdx = savedMessages.reduce((last, msg, idx) =>
        msg.role === "assistant" ? idx : last, -1);

    const parts = savedMessages.map((message, idx) => {
        const label = message.role === "user" ? "You" : "Policy GPT";
        let body, relatedHtml = "";

        if (message.role === "assistant") {
            const { mainText, questions } = splitRelatedQuestions(message.content);
            body = renderMarkdown(mainText);
            relatedHtml = renderRelatedQuestions(questions);
        } else {
            body = `<p>${escapeHtml(message.content)}</p>`;
        }

        const sourcesHtml = (message.role === "assistant" && idx === lastAssistantIdx)
            ? renderSources(state.activeThread?.sources || [])
            : "";

        return `
            <article class="message ${message.role}">
                <div class="message-meta">${label}</div>
                <div class="message-card">
                    <div class="message-body">${body}</div>
                    ${relatedHtml}
                    ${sourcesHtml}
                </div>
            </article>
        `;
    });

    if (hasPending) {
        parts.push(`
            <article class="message user">
                <div class="message-meta">You</div>
                <div class="message-card">
                    <div class="message-body"><p>${escapeHtml(state.pendingPrompt)}</p></div>
                </div>
            </article>
        `);
        parts.push(`
            <article class="message pending">
                <div class="message-meta">Policy GPT</div>
                <div class="message-card">
                    <div class="typing-indicator" aria-label="Generating response">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </article>
        `);
    }

    elements.messages.innerHTML = parts.join("");

    elements.messages.querySelectorAll(".related-chip").forEach((btn) => {
        btn.addEventListener("click", () => {
            const question = btn.getAttribute("data-question");
            if (!question) return;
            elements.composerInput.value = question;
            autosizeComposer();
            sendMessage();
        });
    });

    elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderCorpusSummary() {
    const health = state.health;

    if (!health) {
        elements.corpusSummary.innerHTML = '<p class="thread-empty">Waiting for backend status...</p>';
        return;
    }

    const progress = health.progress || {};
    const progressLine = progress.total_files
        ? `${progress.processed_files}/${progress.total_files} files`
        : "Waiting for file discovery";

    const cards = [
        {
            title: health.status === "ready"
                ? `${health.document_count} documents indexed`
                : `Index status: ${health.status.replaceAll("_", " ")}`,
            body: health.status === "error"
                ? (health.error || "Unknown startup error")
                : `${health.section_count} sections available. ${progressLine}.`,
        },
        {
            title: "Policy folder",
            body: health.document_folder,
        },
        {
            title: `${health.thread_count} threads in memory`,
            body: state.uiError || "Threads are stored in-process and reset when the server restarts.",
        },
    ];

    elements.corpusSummary.innerHTML = cards
        .map((card) => `
            <article class="corpus-card">
                <h4>${escapeHtml(card.title)}</h4>
                <p>${escapeHtml(card.body)}</p>
            </article>
        `)
        .join("");
}

function renderHeader() {
    const ready = state.health?.status === "ready";
    elements.chatTitle.textContent = state.activeThread?.title || "New chat";
    elements.newChatButton.disabled = state.loading || !ready;
    elements.resetChatButton.disabled = !state.activeThread || state.loading || !ready;
    elements.sendButton.disabled = state.loading || !ready;
    elements.sendButton.textContent = state.loading ? "Working..." : "Send";
}

function renderDomainUI() {
    const domain = state.domain;
    if (!domain) {
        return;
    }

    elements.assistantLabel.textContent = domain.assistant_label;
    elements.heroEyebrow.textContent = domain.eyebrow;
    elements.heroDescription.textContent = domain.description;

    elements.promptRow.innerHTML = domain.prompt_chips
        .map((chip) =>
            `<button class="prompt-chip" type="button" data-prompt="${escapeHtml(chip.prompt)}">${escapeHtml(chip.label)}</button>`
        )
        .join("");

    elements.promptRow.querySelectorAll("[data-prompt]").forEach((button) => {
        button.addEventListener("click", () => {
            elements.composerInput.value = button.getAttribute("data-prompt") || "";
            autosizeComposer();
            elements.composerInput.focus();
        });
    });
}

function render() {
    setStatus(state.health);
    renderThreadList();
    renderHeader();
    renderMessages();
    renderCorpusSummary();
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
        },
        ...options,
    });

    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Request failed.");
    }

    return response.json();
}

async function refreshHealth() {
    state.health = await fetchJson("/api/health", { method: "GET" });
}

async function refreshThreads() {
    const data = await fetchJson("/api/threads", { method: "GET" });
    state.threads = data.items;
}

async function loadThread(threadId) {
    try {
        state.uiError = "";
        state.activeThread = await fetchJson(`/api/threads/${threadId}`, { method: "GET" });
    } catch (error) {
        state.uiError = error.message;
    }
    render();
}

async function createThread() {
    try {
        state.uiError = "";
        state.activeThread = await fetchJson("/api/threads", { method: "POST", body: "{}" });
        await refreshHealth();
        await refreshThreads();
    } catch (error) {
        state.uiError = error.message;
    }
    render();
}

async function resetActiveThread() {
    if (!state.activeThread) {
        return;
    }

    try {
        state.uiError = "";
        state.activeThread = await fetchJson(`/api/threads/${state.activeThread.thread_id}/reset`, {
            method: "POST",
            body: "{}",
        });
        await refreshHealth();
        await refreshThreads();
    } catch (error) {
        state.uiError = error.message;
    }
    render();
}

async function sendMessage() {
    const prompt = elements.composerInput.value.trim();
    if (!prompt || state.loading || state.health?.status !== "ready") {
        return;
    }

    state.loading = true;
    state.pendingPrompt = prompt;
    state.uiError = "";
    elements.composerInput.value = "";
    autosizeComposer();
    render();

    try {
        const payload = await fetchJson("/api/chat", {
            method: "POST",
            body: JSON.stringify({
                thread_id: state.activeThread?.thread_id || null,
                message: prompt,
            }),
        });
        state.activeThread = payload.thread;
        await refreshHealth();
        await refreshThreads();
    } catch (error) {
        state.uiError = error.message;
    } finally {
        state.loading = false;
        state.pendingPrompt = "";
        render();
    }
}

function scheduleHealthPoll() {
    window.clearTimeout(state.pollHandle);
    const delay = state.health?.status === "in_progress" || state.health?.status === "starting"
        ? 1500
        : 8000;
    state.pollHandle = window.setTimeout(pollHealth, delay);
}

async function hydrateReadyState(previousStatus) {
    if (state.health?.status !== "ready") {
        if (previousStatus === "ready") {
            state.threads = [];
            state.activeThread = null;
        }
        return;
    }

    if (previousStatus !== "ready") {
        await refreshThreads();
        if (!state.activeThread && state.threads.length) {
            state.activeThread = await fetchJson(`/api/threads/${state.threads[0].thread_id}`, { method: "GET" });
        }
    }
}

async function pollHealth() {
    if (state.healthRequestInFlight) {
        scheduleHealthPoll();
        return;
    }

    state.healthRequestInFlight = true;

    try {
        const previousStatus = state.health?.status || null;
        await refreshHealth();
        await hydrateReadyState(previousStatus);
    } catch (error) {
        state.health = {
            status: "error",
            error: error.message,
            document_folder: "",
            document_count: 0,
            section_count: 0,
            thread_count: 0,
            progress: {
                processed_files: 0,
                total_files: 0,
                current_file: null,
                percent: 0,
            },
        };
        state.uiError = error.message;
    } finally {
        state.healthRequestInFlight = false;
        render();
        scheduleHealthPoll();
    }
}

async function fetchDomain() {
    try {
        state.domain = await fetchJson("/api/domain", { method: "GET" });
        renderDomainUI();
    } catch {
        // Domain UI is non-critical — leave containers empty if it fails.
    }
}

async function bootstrap() {
    render();
    await fetchDomain();
    await pollHealth();
}

elements.composerInput.addEventListener("input", autosizeComposer);
elements.composerInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

elements.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
});

elements.newChatButton.addEventListener("click", createThread);
elements.resetChatButton.addEventListener("click", resetActiveThread);


window.addEventListener("beforeunload", () => {
    window.clearTimeout(state.pollHandle);
});

autosizeComposer();
bootstrap();
