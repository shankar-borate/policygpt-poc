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

    function bindTabBar() {
        const tabs = document.querySelectorAll(".document-viewer-tab[data-tab]");
        if (!tabs.length) {
            return;
        }

        tabs.forEach((tab) => {
            tab.addEventListener("click", () => {
                const targetTab = tab.getAttribute("data-tab");

                // Update tab buttons
                tabs.forEach((t) => {
                    const active = t.getAttribute("data-tab") === targetTab;
                    t.classList.toggle("is-active", active);
                    t.setAttribute("aria-selected", String(active));
                });

                // Show / hide panels
                const sectionsPanel = document.getElementById("tab-panel-sections");
                const originalPanel = document.getElementById("tab-panel-original");
                const banner = document.getElementById("viewer-banner");

                if (targetTab === "original") {
                    if (sectionsPanel) sectionsPanel.hidden = true;
                    if (originalPanel) originalPanel.hidden = false;
                    if (banner) banner.hidden = true;
                } else {
                    if (sectionsPanel) sectionsPanel.hidden = false;
                    if (originalPanel) originalPanel.hidden = true;
                    if (banner) banner.hidden = false;
                }
            });
        });
    }

    function bootstrapViewer() {
        bindTocLinks();
        bindTabBar();
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
