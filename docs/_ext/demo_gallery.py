# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Copy the pictures the demos write into the built documentation, for the gallery on the front page.

The demos write their animations and pictures while the book executes them, which is after the
front page has been read. The front page therefore refers to copies under ``_static/gallery``,
made here once the build has finished.
"""

import re
import shutil
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Where the demos write their pictures, and the page with the gallery, relative to the book's root
DEMOS = "python/demos"
FRONT_PAGE = "index.md"
PATTERNS = ("*.gif", "*.png")


def copy_pictures(app: Sphinx, exception: Exception | None) -> None:
    """Copy every GIF and PNG of the demos to ``_static/gallery`` of an HTML build, and warn
    about a picture the front page shows that no demo wrote."""
    if exception is not None or app.builder.format != "html":
        return
    source = Path(app.srcdir) / DEMOS
    target = Path(app.outdir) / "_static" / "gallery"
    target.mkdir(parents=True, exist_ok=True)
    for pattern in PATTERNS:
        for picture in sorted(source.glob(pattern)):
            shutil.copy2(picture, target / picture.name)

    front_page = Path(app.srcdir) / FRONT_PAGE
    if front_page.exists():
        shown = set(re.findall(r"_static/gallery/([^\"'\s)]+)", front_page.read_text()))
        for name in sorted(shown):
            if not (target / name).exists():
                logger.warning(
                    f"The gallery shows {name}, which no demo in {DEMOS} wrote; did it run?"
                )


def setup(app: Sphinx) -> dict:
    app.connect("build-finished", copy_pictures)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
