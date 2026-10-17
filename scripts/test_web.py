#!/usr/bin/env python3
"""
End-to-end test of the built browser demo in headless Chromium.

    python scripts/build_web.py --pyodide ... --out site
    python scripts/test_web.py site [--screenshots docs]

Serves the folder on localhost, waits for the engine to boot, then runs
Identify, Verify (genuine and impostor), a threshold change and Analyse, and
checks the results shown on the page. Needs `pip install playwright` and a
Chromium (set CHROMIUM=/path/to/chrome to use a specific one).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import sys
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BOOT_TIMEOUT_MS = 240_000
RUN_TIMEOUT_MS = 60_000


def serve(folder: Path) -> http.server.ThreadingHTTPServer:
    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm",
                          ".mjs": "text/javascript", ".js": "text/javascript"}

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(folder)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def run(page, mode: str, print_name: str, claim: str = None):
    page.click(f".tab[data-mode={mode}]")
    if claim:
        page.select_option("#claim", claim)
    page.click(f'.print[data-name="{print_name}"]')
    page.click("#run")
    expect(page.locator("#result")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    expect(page.locator("#busy")).to_be_hidden(timeout=RUN_TIMEOUT_MS)
    return page.locator("#result")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", type=Path)
    ap.add_argument("--screenshots", type=Path, help="save screenshots of the demo to this folder")
    args = ap.parse_args(argv)

    httpd = serve(args.site.resolve())
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    errors = []
    with sync_playwright() as pw:
        exe = os.environ.get("CHROMIUM")
        browser = pw.chromium.launch(executable_path=exe) if exe else pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1180, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        external = []
        page.on("request", lambda r: external.append(r.url) if not r.url.startswith((url, "data:", "blob:")) else None)

        page.goto(url)
        expect(page.locator("#app")).to_be_visible(timeout=BOOT_TIMEOUT_MS)
        expect(page.locator(".person")).to_have_count(4)
        print("booted:", page.locator("#version").inner_text())
        if args.screenshots:
            args.screenshots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.screenshots / "web_demo_home.png"), full_page=False)

        res = run(page, "identify", "demo-03_2")
        expect(res.locator(".verdict")).to_contain_text("Candidate: demo-03")
        expect(res.locator(".truth")).to_contain_text("Correct")
        expect(res.locator("tbody tr")).to_have_count(4)
        expect(res.locator(".pic img")).to_be_visible()
        print("identify demo-03_2 ->", res.locator(".verdict").inner_text())
        if args.screenshots:
            res.screenshot(path=str(args.screenshots / "web_demo_identify.png"))

        res = run(page, "verify", "demo-01_3", claim="demo-01")
        expect(res.locator(".verdict")).to_contain_text("Match: consistent with demo-01")
        res = run(page, "verify", "demo-01_3", claim="demo-02")
        expect(res.locator(".verdict")).to_contain_text("No match")
        expect(res.locator(".truth")).to_contain_text("Correct")
        print("verify genuine and impostor: ok")

        # A threshold above every score turns the genuine identification into "no candidate".
        page.click(".tab[data-mode=identify]")
        page.locator("#threshold").fill("0.6")
        page.locator("#threshold").dispatch_event("change")
        res = run(page, "identify", "demo-03_2")
        expect(res.locator(".verdict")).to_contain_text("No candidate above the threshold")
        page.locator("#threshold").fill("0.18")
        page.locator("#threshold").dispatch_event("change")
        print("threshold change: ok")

        res = run(page, "analyse", "demo-02_1")
        expect(res.locator(".stages figure")).to_have_count(7)
        expect(res.locator(".stat").first).to_contain_text("Quality")
        print("analyse: ok")
        if args.screenshots:
            res.screenshot(path=str(args.screenshots / "web_demo_analyse.png"))

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(url)
        expect(mobile.locator("#app")).to_be_visible(timeout=BOOT_TIMEOUT_MS)
        overflow = mobile.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        print("mobile horizontal overflow:", overflow)
        browser.close()
    httpd.shutdown()

    if errors:
        print("page errors:", *errors, sep="\n  ")
    if external:
        print("unexpected external requests:", *external, sep="\n  ")
    if overflow > 0:
        print("page scrolls horizontally on a phone-sized screen")
    ok = not errors and not external and overflow <= 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
