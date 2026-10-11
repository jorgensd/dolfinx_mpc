// Read the Docs-style version flyout (bottom right).
// Served from the gh-pages root and injected into every built page by
// .github/workflows/deploy-pages.yml, so it also covers old release builds.
(function () {
  const base = "/dolfinx_mpc/";
  const path = window.location.pathname;
  if (!path.startsWith(base)) return;
  const [current, ...rest] = path.slice(base.length).split("/");
  const page = rest.join("/");
  // Same-origin links: the github.io URL redirects to the custom domain, which
  // breaks the cross-origin existence check below
  const home = (v) => base + v.version + "/";

  const css = `
#mpc-flyout { position: fixed; right: 1rem; bottom: 1rem; z-index: 1000;
  font-size: 0.85rem; font-family: var(--pst-font-family-base, sans-serif); }
#mpc-flyout > button { cursor: pointer; border-radius: 0.35rem; padding: 0.35rem 0.75rem;
  border: 1px solid var(--pst-color-border, #ccc);
  background: var(--pst-color-surface, #f5f5f5); color: var(--pst-color-text-base, #222);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); }
#mpc-flyout > button::before { content: "v: "; color: var(--pst-color-primary, #0a7d91); }
#mpc-flyout ul { display: none; position: absolute; right: 0; bottom: calc(100% + 0.4rem);
  margin: 0; padding: 0.3rem 0; list-style: none; min-width: 10rem; border-radius: 0.35rem;
  border: 1px solid var(--pst-color-border, #ccc);
  background: var(--pst-color-surface, #f5f5f5); box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); }
#mpc-flyout.open ul { display: block; }
#mpc-flyout li a { display: block; padding: 0.25rem 0.9rem; text-decoration: none;
  color: var(--pst-color-text-base, #222); }
#mpc-flyout li a:hover { background: var(--pst-color-background, #fff); }
#mpc-flyout li.current a { font-weight: bold; color: var(--pst-color-primary, #0a7d91); }`;

  fetch(base + "switcher.json")
    .then((r) => r.json())
    .then((versions) => {
      const style = document.createElement("style");
      style.textContent = css;
      document.head.appendChild(style);

      const root = document.createElement("div");
      root.id = "mpc-flyout";
      const button = document.createElement("button");
      const entry = versions.find((v) => v.version === current);
      button.textContent = entry ? entry.name : current;
      button.setAttribute("aria-label", "Choose documentation version");
      const list = document.createElement("ul");
      for (const v of versions) {
        const li = document.createElement("li");
        if (v.version === current) li.className = "current";
        const a = document.createElement("a");
        a.textContent = v.name;
        a.href = home(v) + page + window.location.hash;
        // Same page may not exist in that version: fall back to its front page
        a.addEventListener("click", (e) => {
          e.preventDefault();
          fetch(home(v) + page, { method: "HEAD" })
            .then((r) => (window.location.href = r.ok ? a.href : home(v)))
            .catch(() => (window.location.href = home(v)));
        });
        li.appendChild(a);
        list.appendChild(li);
      }
      button.addEventListener("click", () => root.classList.toggle("open"));
      document.addEventListener("click", (e) => {
        if (!root.contains(e.target)) root.classList.remove("open");
      });
      root.append(list, button);
      document.body.appendChild(root);
    })
    .catch(() => {});
})();
