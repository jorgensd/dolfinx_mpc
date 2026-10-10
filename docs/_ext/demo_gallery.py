# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""A gallery of the demos, for the front page: one card per demo in the table of contents, and one
vtk.js viewer shared by all interactive scenes of the book.

The ``demo-gallery`` directive makes a section for each part of the table of contents that has
demos, titled with its caption, and a card for each demo, titled with the first heading of the
demo and linking to it.

A demo names its figure after itself: ``demo_x.py`` writes ``demo_x.gif`` or ``demo_x.png``, and
may export an interactive scene, ``demo_x.vtksz``, with ``pyvista.Plotter.export_vtksz``. The demos
write them while the book executes them, which is after the front page has been read. The cards
therefore refer to copies under ``_static/gallery``, made here once the build has finished, and
name their picture without an extension, ``demo_x.*``. That is then resolved to the animation if
the demo wrote one, and else to the picture; a demo that wrote neither is warned about. A card has a
"3D" button, which shows the interactive scene in place of the picture, if the demo exported one,
with the script and stylesheet added here.

The scenes share one viewer, the static viewer of ``trame_vtk``, in ``_static/scenes``, which
shows the scene named by ``?scene=``. Each scene is a script next to it, ``<scene>.vtksz.js``, as
browsers load scripts for a book opened from disk, ``file://``, but refuse it the request that the
viewer's own ``?fileURL=`` makes. Besides the scenes of the gallery, these are the plots that
pyvista's ``html`` Jupyter backend shows in the pages: it embeds a copy of the viewer, of about
1 MB, in each plot, which is replaced here by the shared viewer. A plot that is not recognised is
left as it is.
"""

import base64
import hashlib
import json
import re
import shutil
from pathlib import Path

import yaml
from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from trame_vtk.tools.vtksz2html import HTML_VIEWER_PATH

logger = logging.getLogger(__name__)

# Where the demos are, and write their figures, relative to the book's root
DEMOS = "python/demos"
PATTERNS = ("*.gif", "*.png")
# For a figure named without its extension, in order of preference
PREFERENCE = (".gif", ".png")
# The script and stylesheet of the "3D" buttons
STATIC = Path(__file__).parent / "static"
# Under the root of the built book: the pictures of the gallery, and the viewer and its scenes
GALLERY = "_static/gallery"
SCENES = "_static/scenes"
VIEWER = "viewer.html"
# Added to the viewer: load the script of the scene named by ?scene=, and show it
LOADER = """<script>
(function () {
  const scene = new URLSearchParams(window.location.search).get("scene");
  if (scene === null || !/^[\\w-]+$/.test(scene)) {
    return;
  }
  const script = document.createElement("script");
  script.src = `${scene}.vtksz.js`;
  script.onload = () =>
    OfflineLocalView.load(document.querySelector(".content"), { base64Str: window.vtkszBase64 });
  document.body.appendChild(script);
})();
</script>
"""
# The state of the Jupyter widgets of a page, which holds the plots of the html backend
WIDGET_STATE = re.compile(
    r'(<script type="application/vnd\.jupyter\.widget-state\+json">)(.*?)(</script>)',
    re.DOTALL,
)
# A plot of the html backend: the viewer with the scene, in base64, as the source of a frame
EMBEDDED = re.compile(
    r'^<iframe srcdoc="[^"]*"(?P<attributes>[^>]*)></iframe>\s*$', re.DOTALL
)
EMBEDDED_SCENE = re.compile(r"var base64Str = &quot;(?P<scene>[A-Za-z0-9+/=]+)&quot;;")


def _title(demo: Path) -> str:
    """The first top-level heading of a demo, a jupytext script."""
    heading = re.search(r"^# # (.+)$", demo.read_text(), flags=re.MULTILINE)
    return heading.group(1).strip() if heading else demo.stem


def _card(docname: str, title: str, prefix: str) -> str:
    """A card linking to the demo `docname`, with its figure and a "3D" button."""
    stem = Path(docname).name
    return (
        f"````{{grid-item-card}} {title}\n:link: {docname}\n:link-type: doc\n\n"
        "```{raw} html\n"
        '<div class="gallery-figure">\n'
        f'<img src="{prefix}{GALLERY}/{stem}.*" alt="{title}" loading="lazy" style="width: 100%">\n'
        '<button class="gallery-3d" type="button" aria-pressed="false" '
        f'data-scene="{prefix}{SCENES}/{VIEWER}?scene={stem}" '
        'title="Show the interactive scene">3D</button>\n'
        "</div>\n```\n````\n"
    )


class DemoGallery(SphinxDirective):
    """A section per part of the table of contents that has demos, with a card per demo."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        root = Path(self.env.srcdir)
        toc_path = root / getattr(self.config, "external_toc_path", "_toc.yml")
        self.env.note_dependency(str(toc_path))
        toc = yaml.safe_load(toc_path.read_text())
        # The pictures are relative to the root of the built book
        prefix = "../" * self.env.docname.count("/")
        text = []
        for part in toc.get("parts", []):
            demos = [
                c["file"]
                for c in part.get("chapters", [])
                if c["file"].startswith(f"{DEMOS}/")
            ]
            if not demos:
                continue
            cards = []
            for docname in demos:
                docname = str(Path(docname).with_suffix(""))
                source = (root / docname).with_suffix(".py")
                self.env.note_dependency(str(source))
                cards.append(_card(docname, _title(source), prefix))
            text.append(
                f"## {part['caption']}\n\n`````{{grid}} 1 2 2 3\n:gutter: 3\n\n"
                + "\n".join(cards)
                + "`````\n"
            )
        container = nodes.section()
        nested_parse_with_titles(
            self.state, StringList("\n".join(text).splitlines()), container
        )
        return container.children


def _write_scene(scenes: Path, name: str, content: str) -> None:
    """Write the scene `name`, a .vtksz file in base64, as a script for the viewer."""
    (scenes / f"{name}.vtksz.js").write_text(f'window.vtkszBase64 = "{content}";\n')


def _resolve_gallery(html: str, gallery: Path, scenes: Path) -> str:
    """Point each figure of a gallery named without an extension at the animation if there is one,
    and else at the picture, warn about a demo that wrote no figure, and drop the "3D" buttons
    without a scene."""

    def choose(match: re.Match) -> str:
        path, stem = match.group(1), match.group(2)
        for suffix in PREFERENCE:
            if (gallery / f"{stem}{suffix}").exists():
                return match.group(0).replace(
                    f"{path}{stem}.*", f"{path}{stem}{suffix}"
                )
        logger.warning(
            f"The gallery shows {stem}, but it wrote neither {stem}.gif nor {stem}.png; did it run?"
        )
        return match.group(0)

    def keep_button(match: re.Match) -> str:
        return (
            match.group(0) if (scenes / f"{match.group(1)}.vtksz.js").exists() else ""
        )

    html = re.sub(
        rf'<img src="((?:\.\./)*{GALLERY}/)([\w-]+)\.\*"[^>]*>\n?', choose, html
    )
    return re.sub(
        r'<button class="gallery-3d"[^>]*'
        rf'data-scene="(?:\.\./)*{SCENES}/{re.escape(VIEWER)}\?scene=([\w-]+)"'
        r"[^>]*>3D</button>\n?",
        keep_button,
        html,
    )


def _share_viewer(html: str, scenes: Path, prefix: str) -> str:
    """Replace the viewer that each plot of the html backend embeds by the shared viewer."""

    def share(match: re.Match) -> str:
        state = json.loads(match.group(2))
        shared = 0
        for model in state.get("state", {}).values():
            value = model.get("state", {}).get("value")
            if not isinstance(value, str) or not value.startswith("<iframe srcdoc="):
                continue
            embedded, scene = EMBEDDED.match(value), EMBEDDED_SCENE.search(value)
            if embedded is None or scene is None:
                continue
            content = scene.group("scene")
            name = f"plot-{hashlib.sha1(content.encode()).hexdigest()[:16]}"
            _write_scene(scenes, name, content)
            model["state"]["value"] = (
                f'<iframe src="{prefix}{SCENES}/{VIEWER}?scene={name}"'
                f"{embedded.group('attributes')}></iframe>"
            )
            shared += 1
        if shared == 0:
            return match.group(0)
        # As in the original, "</" is escaped so that the state cannot close its script
        return match.group(1) + json.dumps(state).replace("</", "<\\/") + match.group(3)

    return WIDGET_STATE.sub(share, html)


def build_assets(app: Sphinx, exception: Exception | None) -> None:
    """Write the viewer and the scenes of an HTML build, copy the figures of the demos, and point
    the pages at them."""
    if exception is not None or app.builder.format != "html":
        return
    outdir = Path(app.outdir)
    source = Path(app.srcdir) / DEMOS
    gallery, scenes = outdir / GALLERY, outdir / SCENES
    gallery.mkdir(parents=True, exist_ok=True)
    scenes.mkdir(parents=True, exist_ok=True)
    for pattern in PATTERNS:
        for figure in sorted(source.glob(pattern)):
            shutil.copy2(figure, gallery / figure.name)
    for scene in sorted(source.glob("*.vtksz")):
        _write_scene(scenes, scene.stem, base64.b64encode(scene.read_bytes()).decode())
    viewer = HTML_VIEWER_PATH.read_text(encoding="utf-8")
    end = viewer.rindex("</body>")
    (scenes / VIEWER).write_text(viewer[:end] + LOADER + viewer[end:], encoding="utf-8")
    static = outdir / "_static"
    for page in outdir.rglob("*.html"):
        if static in page.parents:
            continue
        html = page.read_text()
        resolved = html
        if f"{GALLERY}/" in resolved:
            resolved = _resolve_gallery(resolved, gallery, scenes)
        if "<iframe srcdoc=" in resolved:
            prefix = "../" * len(page.relative_to(outdir).parent.parts)
            resolved = _share_viewer(resolved, scenes, prefix)
        if resolved != html:
            page.write_text(resolved)


def add_static_path(app: Sphinx) -> None:
    """Serve the script and stylesheet of the "3D" buttons."""
    app.config.html_static_path.append(str(STATIC))


def setup(app: Sphinx) -> dict:
    app.add_directive("demo-gallery", DemoGallery)
    app.connect("builder-inited", add_static_path)
    app.connect("build-finished", build_assets)
    app.add_js_file("demo_gallery.js")
    app.add_css_file("demo_gallery.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
