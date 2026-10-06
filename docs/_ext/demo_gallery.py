# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""A gallery of the demos, for the front page: one card per demo in the table of contents.

The ``demo-gallery`` directive makes a section for each part of the table of contents that has
demos, titled with its caption, and a card for each demo, titled with the first heading of the
demo and linking to it.

A demo names its figure after itself: ``demo_x.py`` writes ``demo_x.gif`` or ``demo_x.png``, and
may export an interactive scene, ``demo_x.html``. The demos write them while the book executes
them, which is after the front page has been read. The cards therefore refer to copies under
``_static/gallery``, made here once the build has finished, and name their picture without an
extension, ``demo_x.*``. That is then resolved to the animation if the demo wrote one, and else to
the picture. A card has a "3D" button, which shows the interactive scene in place of the picture,
if the demo exported one, with the script and stylesheet added here.
"""

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

logger = logging.getLogger(__name__)

# Where the demos are, and write their figures, relative to the book's root
DEMOS = "python/demos"
PATTERNS = ("*.gif", "*.png", "*.html")
# For a figure named without its extension, in order of preference
PREFERENCE = (".gif", ".png")
# The script and stylesheet of the "3D" buttons
STATIC = Path(__file__).parent / "static"


def _title(demo: Path) -> str:
    """The first top-level heading of a demo, a jupytext script."""
    heading = re.search(r"^# # (.+)$", demo.read_text(), flags=re.MULTILINE)
    return heading.group(1).strip() if heading else demo.stem


def _card(docname: str, title: str, prefix: str) -> str:
    """A card linking to the demo `docname`, with its figure and a "3D" button."""
    stem = Path(docname).name
    gallery = f"{prefix}_static/gallery"
    return (
        f"````{{grid-item-card}} {title}\n:link: {docname}\n:link-type: doc\n\n"
        "```{raw} html\n"
        '<div class="gallery-figure">\n'
        f'<img src="{gallery}/{stem}.*" alt="{title}" loading="lazy" style="width: 100%">\n'
        f'<button class="gallery-3d" type="button" aria-pressed="false" data-scene="{gallery}/{stem}.html" '
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


def _resolve(page: Path, target: Path) -> None:
    """Point each figure of a built page named without an extension at the animation if there is
    one, and else at the picture, drop the "3D" buttons without a scene, and warn about a demo that
    wrote no figure."""
    html = page.read_text()

    def choose(match: re.Match) -> str:
        path, stem = match.group(1), match.group(2)
        for suffix in PREFERENCE:
            if (target / f"{stem}{suffix}").exists():
                return f"{path}{stem}{suffix}"
        logger.warning(
            f"The gallery shows {stem}, but it wrote neither {stem}.gif nor {stem}.png; did it run?"
        )
        return match.group(0)

    def keep_button(match: re.Match) -> str:
        return match.group(0) if (target / f"{match.group(1)}.html").exists() else ""

    resolved = re.sub(r"((?:\.\./)*_static/gallery/)([\w-]+)\.\*", choose, html)
    resolved = re.sub(
        r'<button class="gallery-3d"[^>]*data-scene="(?:\.\./)*_static/gallery/([\w-]+)\.html"[^>]*>3D</button>\n?',
        keep_button,
        resolved,
    )
    if resolved != html:
        page.write_text(resolved)


def build_gallery(app: Sphinx, exception: Exception | None) -> None:
    """Copy the figures of the demos to ``_static/gallery`` of an HTML build, and resolve those of
    the pages with a gallery."""
    if exception is not None or app.builder.format != "html":
        return
    source = Path(app.srcdir) / DEMOS
    target = Path(app.outdir) / "_static" / "gallery"
    target.mkdir(parents=True, exist_ok=True)
    for pattern in PATTERNS:
        for figure in sorted(source.glob(pattern)):
            shutil.copy2(figure, target / figure.name)
    for page in Path(app.outdir).rglob("*.html"):
        if target in page.parents:
            continue
        if "_static/gallery/" in page.read_text():
            _resolve(page, target)


def add_static_path(app: Sphinx) -> None:
    """Serve the script and stylesheet of the "3D" buttons."""
    app.config.html_static_path.append(str(STATIC))


def setup(app: Sphinx) -> dict:
    app.add_directive("demo-gallery", DemoGallery)
    app.connect("builder-inited", add_static_path)
    app.connect("build-finished", build_gallery)
    app.add_js_file("demo_gallery.js")
    app.add_css_file("demo_gallery.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
