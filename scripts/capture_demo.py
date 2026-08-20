"""Capture screenshots of the demo page for the README.

Starts the FastAPI app on a throwaway port, drives the page with a real browser, uploads
a validation frame, and captures each layer view. Uses the system Chrome via Playwright's
``channel="chrome"`` so no separate browser binary is downloaded.

Playwright is a development-only dependency and is deliberately absent from
``requirements.txt``: nothing at training, evaluation or serving time needs it, and the
README images it produces are committed, so a reader reproducing the project never has to
install a browser stack.

Run::

    pip install playwright
    PYTHONPATH=. python scripts/capture_demo.py
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
#: Layer tabs to capture, and the filename each is written to.
LAYERS = ("overlay", "mask", "confidence")


def _free_port() -> int:
    """Pick an unused port, so a capture never collides with a dev server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(url: str, timeout: float = 60.0) -> bool:
    """Block until the app answers, or give up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def capture(
    image: Path, out_dir: Path, width: int = 1280, height: int = 900, scale: int = 2
) -> list[Path]:
    """Serve the app, drive the page, and write one screenshot per layer."""
    from playwright.sync_api import sync_playwright

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.api:app", "--port", str(port),
         "--log-level", "warning"],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": __import__("os").environ["PATH"]},
    )
    written: list[Path] = []
    try:
        if not _wait_for_health(url):
            raise RuntimeError(f"app did not come up on {url}")
        out_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as playwright:
            # Use the installed Chrome rather than a downloaded build.
            browser = playwright.chromium.launch(channel="chrome")
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=scale,
                # The page ships a single light theme; pin it so a reviewer running with
                # a dark OS preference still regenerates identical screenshots.
                color_scheme="light",
            )
            page.goto(url, wait_until="networkidle")
            page.set_input_files("#file", str(image))
            # The result is base64 PNGs decoded by the browser; wait for the <img> to
            # actually have pixels rather than for the network call to return.
            page.wait_for_selector("#compare:not(.hidden)", timeout=30_000)
            page.wait_for_function(
                "() => ['view-input', 'view'].every(id => { const i ="
                " document.getElementById(id); return i && i.complete && i.naturalWidth > 0; })",
                timeout=30_000,
            )
            for layer in LAYERS:
                page.click(f'button[data-layer="{layer}"]')
                page.wait_for_function(
                    "() => { const i = document.getElementById('view');"
                    " return i.complete && i.naturalWidth > 0; }"
                )
                page.wait_for_timeout(150)  # let the tab transition settle
                path = out_dir / f"demo_{layer}.png"
                page.screenshot(path=str(path))
                written.append(path)
                print(f"  {path.relative_to(REPO_ROOT)}", flush=True)
            page.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=None,
        help="Frame to segment; defaults to a validation frame from the prepared dataset",
    )
    parser.add_argument("--out-dir", default="docs/images")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    if args.image:
        image = Path(args.image)
    else:
        candidates = sorted(
            (REPO_ROOT / "data/processed/level1_official/images/val").glob("*.jpg")
        )
        if not candidates:
            raise SystemExit(
                "No validation frames found. Run: python -m data.prepare_masks"
            )
        # A mid-split frame rather than the first, which tends to be an outlier scene.
        image = candidates[len(candidates) // 3]

    print(f"capturing {image.name}")
    written = capture(image, REPO_ROOT / args.out_dir, args.width, args.height)
    print(f"\n{len(written)} screenshots -> {args.out_dir}/")


if __name__ == "__main__":
    main()
