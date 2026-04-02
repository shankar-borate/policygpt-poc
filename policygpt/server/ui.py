from pathlib import Path


class WebUIRenderer:
    INDEX_HEAD_INJECTION = """
<style>
    .reference-panel {
        display: none !important;
    }

    .message-reference-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }

    .message-reference-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border: 1px solid #e5e7eb;
        border-radius: 999px;
        color: #6b7280;
        background: #f3f4f6;
        font-size: 0.74rem;
        font-style: italic;
        line-height: 1.2;
        white-space: nowrap;
        text-decoration: none;
    }

    .message-reference-pill:hover {
        color: #374151;
        background: #e5e7eb;
    }
</style>
"""

    INDEX_BODY_INJECTION = """
<script>
(() => {
    function buildReferenceRow(rawText) {
        const text = String(rawText || "").trim();
        if (!text.toLowerCase().startsWith("reference:")) {
            return null;
        }

        const values = text
            .slice("Reference:".length)
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean);

        if (!values.length) {
            return null;
        }

        const row = document.createElement("div");
        row.className = "message-reference-row";

        values.forEach((value) => {
            const pill = document.createElement("span");
            pill.className = "message-reference-pill";
            pill.textContent = value;
            row.appendChild(pill);
        });

        return row;
    }

    function buildReferenceRowFromParagraph(paragraph) {
        const text = String(paragraph?.textContent || "").trim();
        if (!text.toLowerCase().startsWith("reference:")) {
            return null;
        }

        const links = [...paragraph.querySelectorAll("a[href]")];
        if (!links.length) {
            return buildReferenceRow(text);
        }

        const row = document.createElement("div");
        row.className = "message-reference-row";

        links.forEach((link) => {
            const pill = document.createElement("a");
            pill.className = "message-reference-pill";
            pill.href = link.href;
            pill.target = "_blank";
            pill.rel = "noreferrer noopener";
            pill.textContent = link.textContent.trim();
            row.appendChild(pill);
        });

        return row;
    }

    function enhanceMessageReferences() {
        document.querySelectorAll(".message.assistant .message-body").forEach((body) => {
            if (body.dataset.referenceEnhanced === "true") {
                return;
            }

            const paragraphs = body.querySelectorAll(":scope > p");
            const lastParagraph = paragraphs[paragraphs.length - 1];
            if (!lastParagraph) {
                body.dataset.referenceEnhanced = "true";
                return;
            }

            const referenceRow = buildReferenceRowFromParagraph(lastParagraph);
            if (referenceRow) {
                lastParagraph.replaceWith(referenceRow);
            }

            body.dataset.referenceEnhanced = "true";
        });
    }

    function startReferenceEnhancer() {
        const messages = document.getElementById("messages");
        if (!messages) {
            return;
        }

        const observer = new MutationObserver(() => {
            enhanceMessageReferences();
        });

        observer.observe(messages, { childList: true, subtree: true });
        enhanceMessageReferences();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startReferenceEnhancer, { once: true });
    } else {
        startReferenceEnhancer();
    }
})();
</script>
"""

    def __init__(self, web_dir: Path) -> None:
        self.web_dir = web_dir

    def render_index(self) -> str:
        html = (self.web_dir / "index.html").read_text(encoding="utf-8")
        with_head = html.replace("</head>", f"{self.INDEX_HEAD_INJECTION}</head>")
        return with_head.replace("</body>", f"{self.INDEX_BODY_INJECTION}</body>")
