"""A small accessibility audit for the dashboard markup.

Not a replacement for a real screen reader, but it catches the mistakes that
creep back in silently: unlabelled controls, headings that skip a level, tables
without captions, colour used as the only signal, images without text.

    python tools/a11y_audit.py            # audits the shipped file
    python tools/a11y_audit.py --url http://127.0.0.1:4000/dashboard
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTML = ROOT / "cachellm" / "server" / "static" / "index.html"
DEFAULT_JS = ROOT / "cachellm" / "server" / "static" / "dashboard.js"

problems: list[str] = []
passes: list[str] = []


def check(ok: bool, description: str, detail: str = "") -> None:
    if ok:
        passes.append(description)
    else:
        problems.append(description + (f" ({detail})" if detail else ""))


def audit(html: str, js: str) -> None:
    # --- document basics -------------------------------------------------
    check(bool(re.search(r"<html[^>]+lang=", html)), "the page declares its language")
    check("<title>" in html, "the page has a title")
    check('name="viewport"' in html, "the page scales on small screens and when zoomed")

    # --- landmarks and skip link ----------------------------------------
    check("<header" in html, "there is a banner landmark")
    check("<main" in html, "there is a main landmark")
    check('href="#main"' in html and 'id="main"' in html,
          "a skip link jumps past the header to the content")
    check(".skip-link:focus" in html, "the skip link becomes visible when focused")

    # --- headings --------------------------------------------------------
    levels = [int(m) for m in re.findall(r"<h([1-6])\b", html)]
    check(levels.count(1) == 1, "exactly one h1", f"found {levels.count(1)}")
    skipped = [
        (a, b) for a, b in zip(levels, levels[1:]) if b > a + 1
    ]
    check(not skipped, "heading levels never skip a level", str(skipped))

    # --- controls --------------------------------------------------------
    control_ids = set(re.findall(r"<(?:input|select|textarea)[^>]*\bid=\"([^\"]+)\"", html))
    labelled = set(re.findall(r"<label[^>]*\bfor=\"([^\"]+)\"", html))
    self_labelled = set(
        re.findall(r"<(?:input|select)[^>]*\bid=\"([^\"]+)\"[^>]*aria-label=", html)
    )
    orphans = control_ids - labelled - self_labelled
    check(not orphans, "every form control has a label", str(sorted(orphans)))

    buttons = re.findall(r"<button[^>]*>", html)
    untyped = [b for b in buttons if 'type="' not in b]
    check(not untyped, "every button declares type", f"{len(untyped)} without type")

    empty_buttons = re.findall(r"<button[^>]*>\s*</button>", html)
    check(not empty_buttons, "no button is empty of text")

    check("onclick=" not in html and "onchange=" not in html,
          "no inline event handlers (keyboard support comes from real elements)")

    # --- tabs ------------------------------------------------------------
    if 'role="tablist"' in html:
        tabs = re.findall(r'role="tab"[^>]*aria-controls="([^"]+)"', html)
        check(len(tabs) >= 2, "tabs declare which panel they control")
        check(all(f'id="{panel}"' in html for panel in tabs),
              "every tab points at a panel that exists")
        check(html.count('aria-selected') >= len(tabs),
              "every tab reports whether it is selected")
        check(all(key in js for key in ("ArrowRight", "ArrowLeft", "Home", "End")),
              "tabs are navigable with the arrow keys")
        check("tabIndex" in js or "tabindex" in js,
              "only the selected tab is in the tab order")

    # --- tables ----------------------------------------------------------
    tables = html.count("<table")
    captions = html.count("<caption")
    check(captions >= tables, "every table has a caption",
          f"{tables} tables, {captions} captions")
    check('scope="col"' in html, "column headers are scoped")
    check("th.scope = 'row'" in js or 'scope="row"' in html, "row headers are scoped")

    # --- live regions ----------------------------------------------------
    check('aria-live="polite"' in html, "there is a polite live region for confirmations")
    check('role="alert"' in html, "there is an assertive region for errors")

    # --- colour and motion ----------------------------------------------
    check("prefers-reduced-motion" in html, "reduced-motion preference is honoured")
    check("prefers-contrast" in html, "increased-contrast preference is honoured")
    check("prefers-color-scheme" in html, "light and dark themes both supported")
    check(":focus-visible" in html, "focus is always visible")
    check("outline" in html, "the focus style uses an outline, not just colour")

    # Text label alongside colour for every state.
    for label in ("hit (exact)", "miss", "not cacheable", "error"):
        check(label in js, f"state {label!r} has a text label, not just a colour")

    # --- injection / announcement hygiene -------------------------------
    check("innerHTML" not in js, "content is built with textContent, never innerHTML")
    check("aria-label" in js or "setAttribute('aria-label'" in js,
          "generated buttons get accessible names")

    # --- images ----------------------------------------------------------
    images = re.findall(r"<img[^>]*>", html)
    check(all("alt=" in image for image in images), "every image has alt text")
    decorative = re.findall(r'role="img"[^>]*>', html)
    for element in decorative:
        check("aria-label" in element, "graphical elements carry a text alternative")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the CacheLLM dashboard markup")
    parser.add_argument("--url", help="audit a running dashboard instead of the file")
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--js", type=Path, default=DEFAULT_JS)
    args = parser.parse_args()

    if args.url:
        import httpx

        html = httpx.get(args.url, timeout=15).text
        base = args.url.rsplit("/", 1)[0]
        js = httpx.get(f"{base}/static/dashboard.js", timeout=15).text
    else:
        html = args.html.read_text(encoding="utf-8")
        js = args.js.read_text(encoding="utf-8")

    audit(html, js)

    for line in passes:
        print(f"  ok    {line}")
    for line in problems:
        print(f"  FAIL  {line}")
    print(f"\n{len(passes)} checks passed, {len(problems)} failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
