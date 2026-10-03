"""
Convert the supervisor's annotated manuscript (thesis_paper/source/*.docx) to thesis_paper/paper/paper.md.

The source colours text: black = thesis, blue = rewritten during conversion, red = issues to fix, yellow
highlight = text under revision. Pandoc drops colours, so this script works on a COPY of the docx (the original is
never touched): it tags red runs with a character style, converts with `pandoc -f docx+styles`, and turns
  - red text      into an HTML comment at the same place:  <!-- RED: ... -->
  - yellow text   into <mark>...</mark> (pandoc reads highlights itself)
Blue text is kept as ordinary text; its provenance is recorded per claim in reports/claim_inventory.csv.
The 12 embedded images are extracted to thesis_paper/figures/source_media/.

Usage (from the repository root):
    python thesis_paper/scripts/convert_docx_to_md.py            # write paper/paper.md
    python thesis_paper/scripts/convert_docx_to_md.py --check    # exit 1 if paper.md differs from a fresh conversion
Needs pandoc on PATH (or $PANDOC).
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "thesis_paper" / "source" / "Perovskite_Thermoelectric_Manuscript.docx"
SRC_SHA256 = "f12d66404e126de1cb7cc32d8843058988f43dc3777ac5669b17c319223fa2d4"
OUT = REPO / "thesis_paper" / "paper" / "paper.md"
MEDIA = REPO / "thesis_paper" / "figures" / "source_media"

BACKSLASH = chr(92)
STYLE_XML = (
    '<w:style w:type="character" w:customStyle="1" w:styleId="RedNote"><w:name w:val="RedNote"/>'
    '<w:basedOn w:val="DefaultParagraphFont"/></w:style>'
)
LIST_PARAGRAPH = re.compile(r'(?m)^(\d+\.|-)(\s+)::: \{custom-style="List Paragraph"\}\n\s+(.*?)\n\s+:::', re.S)


def tag_runs(document_xml):
    """Add the RedNote character style to every red run (highlighted or not); pandoc reads highlights itself."""
    counts = {"RedNote": 0}

    def fix(m):
        run = m.group(0)
        if 'w:color w:val="FF0000"' not in run or "<w:rPr>" not in run:
            return run
        counts["RedNote"] += 1
        run = re.sub(r'<w:rStyle w:val="[^"]*"\s*/>', "", run, count=1)
        return run.replace("<w:rPr>", '<w:rPr><w:rStyle w:val="RedNote"/>', 1)

    return re.sub(r"<w:r(?:\s[^>]*)?>.*?</w:r>", fix, document_xml, flags=re.S), counts


def pandoc_exe():
    """Path of the pandoc binary."""
    exe = os.environ.get("PANDOC") or shutil.which("pandoc")
    if not exe:
        raise SystemExit("pandoc not found: set $PANDOC or put it on PATH")
    return exe


def unescape(s):
    """Plain text from pandoc-markdown inside a comment: drop the backslash before an escaped punctuation mark."""
    punct = "[]*_`<>#~^$@|'\"" + BACKSLASH
    out, i = [], 0
    while i < len(s):
        if s[i] == BACKSLASH and i + 1 < len(s) and s[i + 1] in punct:
            i += 1
        out.append(s[i])
        i += 1
    return "".join(out).replace("---", "—").replace("--", "–")


def replace_spans(md, attr, fn):
    """Replace every pandoc span `[text]{attr}` (text may hold balanced brackets) by fn(text)."""
    suffix = "]" + attr
    out, pos = [], 0
    while True:
        end = md.find(suffix, pos)
        if end < 0:
            out.append(md[pos:])
            break
        depth, i = 0, end
        while i >= pos:
            if md[i] == "]" and (i == 0 or md[i - 1] != BACKSLASH):
                depth += 1
            elif md[i] == "[" and (i == 0 or md[i - 1] != BACKSLASH):
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        assert i >= pos and depth == 0, md[max(0, end - 80):end + 40]
        out.append(md[pos:i])
        out.append(fn(md[i + 1:end]))
        pos = end + len(suffix)
    return "".join(out)


def convert():
    """Return (markdown text, run counts)."""
    assert hashlib.sha256(SRC.read_bytes()).hexdigest() == SRC_SHA256, "source docx changed"
    tmp = Path(tempfile.mkdtemp())
    try:
        work = tmp / "tagged.docx"
        counts = None
        with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(work, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    xml, counts = tag_runs(data.decode("utf-8"))
                    data = xml.encode("utf-8")
                elif item.filename == "word/styles.xml":
                    data = data.decode("utf-8").replace("</w:styles>", STYLE_XML + "</w:styles>").encode("utf-8")
                zout.writestr(item, data)
        media_tmp = tmp / "media"
        cmd = [pandoc_exe(), str(work), "-f", "docx+styles", "-t", "markdown+pipe_tables-simple_tables-multiline_tables-grid_tables",
               "--wrap=none", f"--extract-media={media_tmp}", "--markdown-headings=atx"]
        md = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True).stdout
        if MEDIA.exists():
            shutil.rmtree(MEDIA)
        shutil.copytree(media_tmp / "media", MEDIA)
        for prefix in (str(media_tmp).replace(BACKSLASH, "/"), str(media_tmp)):
            md = md.replace(prefix + "/media/", "../figures/source_media/").replace(prefix + BACKSLASH + "media" + BACKSLASH, "../figures/source_media/")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    md = replace_spans(md, "{.mark}", lambda t: "<mark>" + t + "</mark>")
    md = replace_spans(md, '{custom-style="Hyperlink"}', lambda t: t)
    md = replace_spans(md, '{custom-style="RedNote"}', lambda t: "<!-- RED: " + unescape(re.sub(r"</?mark>", "", t)).strip() + " -->")
    md = LIST_PARAGRAPH.sub(lambda m: m.group(1) + m.group(2) + m.group(3), md)
    # pandoc repeats each caption as image alt text (cutting the red notes short); the caption paragraph that follows is complete
    md = re.sub(r'!\[.*?\]\((\.\./figures/source_media/image\d+\.png) "(Figure \d+)"\)', lambda m: "![" + m.group(2) + "](" + m.group(1) + ")", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    return md, counts


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    md, counts = convert()
    leftovers = re.findall(r'custom-style="[^"]*"', md)
    assert not leftovers, leftovers[:3]
    if args.check:
        cur = OUT.read_bytes().decode("utf-8").replace("\r\n", "\n") if OUT.exists() else ""
        if cur != md:
            print("paper.md differs from a fresh conversion (it has been edited, or the conversion changed)")
            return 1
        print("paper.md matches a fresh conversion")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(md.encode("utf-8"))
    print(f"wrote {OUT}: {len(md):,} chars; red runs tagged {counts['RedNote']}; red comments {md.count('<!-- RED')}; "
          f"highlighted spans {md.count('<mark>')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
