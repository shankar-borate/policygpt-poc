(() => {
    function activateSection(targetId) {
        if (!targetId) {
            return;
        }

        const target = document.getElementById(targetId);
        if (!target) {
            return;
        }

        document.querySelectorAll(".document-viewer-section.is-current").forEach((section) => {
            section.classList.remove("is-current");
        });
        document.querySelectorAll(".document-viewer-toc-link.is-active").forEach((link) => {
            link.classList.remove("is-active");
        });

        target.classList.add("is-current", "flash-target");
        target.focus({ preventScroll: true });
        target.scrollIntoView({ behavior: "smooth", block: "start" });

        const activeLink = document.querySelector(`[data-section-link-id="${CSS.escape(targetId)}"]`);
        if (activeLink) {
            activeLink.classList.add("is-active");
        }

        window.setTimeout(() => {
            target.classList.remove("flash-target");
        }, 1800);
    }

    function bindTocLinks() {
        document.querySelectorAll(".document-viewer-toc-link[data-section-link-id]").forEach((link) => {
            link.addEventListener("click", () => {
                const targetId = link.getAttribute("data-section-link-id");
                window.setTimeout(() => activateSection(targetId), 0);
            });
        });
    }

    function bootstrapViewer() {
        bindTocLinks();
        const payload = document.body.dataset.target;
        if (!payload) {
            return;
        }

        try {
            const parsed = JSON.parse(payload);
            window.requestAnimationFrame(() => activateSection(parsed.id));
        } catch (_error) {
            // Ignore malformed viewer payload.
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrapViewer, { once: true });
    } else {
        bootstrapViewer();
    }
})();
