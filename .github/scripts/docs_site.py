# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Helpers for the versioned docs deployment (.github/workflows/deploy-pages.yml).

Versions are listed, newest first, in .github/docs-versions.yml. The built HTML
of each version is kept as ``docs-<tag>.tar.gz`` on the ``docs-archive`` release.
"""

import argparse
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

import yaml

# Root-relative, so links stay on whichever domain serves the site
BASE_URL = "/dolfinx_mpc"
# Release holding the built HTML, set by the workflow
ARCHIVE = os.environ.get("ARCHIVE", "docs-archive")
CONFIG = Path(__file__).parents[1] / "docs-versions.yml"
MAIN = {"label": "main", "name": "main (nightly)", "tag": "main"}
FLYOUT = f'<script defer src="{BASE_URL}/version-flyout.js"></script>'
# The page's own <head> comes first; later ones are inside embedded documents,
# e.g. the escaped viewer page in a pyvista plot's widget state
HEAD = re.compile(r"<head(?:\s[^>]*)?>", re.IGNORECASE)


def versions() -> list[dict[str, str]]:
    """Return all documented versions, newest first, ``main`` included."""
    with CONFIG.open() as f:
        config = yaml.safe_load(f)
    tagged = [
        {
            "label": v["minor"],
            "name": v["minor"],
            "tag": v["mpc_tag"],
            "image": v["dolfinx_image"],
        }
        for v in config["versions"]
    ]
    return [MAIN | {"image": "ghcr.io/fenics/dolfinx/lab:nightly"}, *tagged]


def asset_name(tag: str) -> str:
    return f"docs-{tag}.tar.gz"


def matrix(args: argparse.Namespace) -> None:
    """Write the build matrix to $GITHUB_OUTPUT: main, plus tags not yet archived."""
    archived = set(args.archived.read_text().split())
    jobs = [
        {"label": v["label"], "git_ref": v["tag"], "dolfinx_image": v["image"]}
        for v in versions()
        if v["tag"] == "main" or asset_name(v["tag"]) not in archived
    ]
    print("Jobs to build:", [j["label"] for j in jobs])
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"matrix={json.dumps({'include': jobs})}\n")


def write_redirects(site: Path) -> None:
    """Keep pre-versioning URLs working: ``<page>.html`` -> ``main/<page>.html``."""
    for page in (site / "main").rglob("*.html"):
        rel = page.relative_to(site / "main")
        stub = site / rel
        if stub.exists() or rel.parts[0].startswith("_"):
            continue
        stub.parent.mkdir(parents=True, exist_ok=True)
        target = "../" * (len(rel.parts) - 1) + "main/" + rel.as_posix()
        stub.write_text(
            '<!DOCTYPE html>\n<meta charset="utf-8">\n'
            f'<meta http-equiv="refresh" content="0; url={target}">\n'
            f'<link rel="canonical" href="{target}">\n'
        )


def assemble(args: argparse.Namespace) -> None:
    """Unpack every version into the site, fetching missing ones from the archive."""
    entries = []
    for v in versions():
        tarball = args.fresh / asset_name(v["tag"])
        if not tarball.exists():
            fetched = subprocess.run(
                ["gh", "release", "download", ARCHIVE, "--pattern", tarball.name]
                + ["--dir", str(args.fresh)],
                check=False,
            )
            if fetched.returncode != 0:
                print(f"::warning::No build of {v['tag']}, leaving {v['label']} out")
                continue
        with tarfile.open(tarball) as t:
            t.extractall(args.site / v["label"], filter="data")
        entries.append(
            {
                "name": v["name"],
                "version": v["label"],
                "url": f"{BASE_URL}/{v['label']}/",
            }
        )

    if not any(e["version"] == "main" for e in entries):
        raise SystemExit("main docs are neither built nor archived")
    (args.site / "switcher.json").write_text(json.dumps(entries, indent=2))
    print("switcher.json:", entries)
    write_redirects(args.site)


def inject(args: argparse.Namespace) -> None:
    """Load the version flyout in every page of the deployed versions."""
    entries = json.loads((args.site / "switcher.json").read_text())
    for e in entries:
        for page in (args.site / e["version"]).rglob("*.html"):
            html = page.read_text(encoding="utf-8")
            head = HEAD.search(html)
            if head is None:
                continue
            page.write_text(
                html[: head.end()] + FLYOUT + html[head.end() :], encoding="utf-8"
            )


def assets(args: argparse.Namespace) -> None:
    """Print the archive assets still in use."""
    print(*(asset_name(v["tag"]) for v in versions()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("matrix", help=matrix.__doc__)
    p.add_argument("archived", type=Path, help="file listing archived asset names")
    p.set_defaults(func=matrix)

    p = sub.add_parser("assemble", help=assemble.__doc__)
    p.add_argument("fresh", type=Path, help="directory with freshly built tarballs")
    p.add_argument("site", type=Path, help="output directory")
    p.set_defaults(func=assemble)

    p = sub.add_parser("inject", help=inject.__doc__)
    p.add_argument("site", type=Path)
    p.set_defaults(func=inject)

    p = sub.add_parser("assets", help=assets.__doc__)
    p.set_defaults(func=assets)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
