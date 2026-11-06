"""Render docs/proposal.md -> docs/proposal.pdf (US Letter, compact 1-page layout) via headless Edge/Chrome."""
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC, HTML, PDF = ROOT / "docs" / "proposal.md", ROOT / "docs" / "proposal.html", ROOT / "docs" / "proposal.pdf"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # Chrome first: headless Edge can silently write nothing
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]

CSS = """
@page { size: Letter; margin: 0.45in 0.5in; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 9.3pt; line-height: 1.26; color: #111; }
h1 { font-size: 15pt; margin: 0 0 4px; }
h2 { font-size: 11pt; margin: 9px 0 3px; border-bottom: 1px solid #bbb; padding-bottom: 1px; }
p, ul, ol { margin: 3px 0; } ul, ol { padding-left: 18px; } li { margin: 1px 0; }
table { border-collapse: collapse; width: 100%; margin: 3px 0; }
th, td { border: 1px solid #ccc; padding: 2px 6px; vertical-align: top; text-align: left; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f4f4f4; padding: 0 2px; }
"""


def main() -> None:
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables"])
    HTML.write_text(f"<!doctype html><meta charset='utf-8'><title>Proposal</title><style>{CSS}</style>{body}", encoding="utf-8")
    exe = next((b for b in BROWSERS if Path(b).exists()), None) or sys.exit("No Edge/Chrome found")
    PDF.unlink(missing_ok=True)  # fails here if the PDF is open in a viewer that locks it
    # A separate profile stops headless mode from handing off to an already-open browser window.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                        f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=120)
    HTML.unlink()
    if not PDF.exists():
        sys.exit("PDF was not written")
    from pypdf import PdfReader
    print(f"{PDF.name}: {len(PdfReader(PDF).pages)} page(s)")


if __name__ == "__main__":
    main()
