"""
Minimal PDF reader for vector figures (no third-party library): objects and object streams, page content, and a content-stream interpreter that returns the filled and
stroked paths (CTM applied, with colour) and the text strings with positions. Written for the NA12 reading of the plotted markers of Sebastia-Luna et al. 2022
(docs/decisions.md, 2026-10-13); it handles the files of that paper, not arbitrary PDFs.
"""

import re
import sys
import zlib


OBJ = re.compile(rb"(\d+)\s+(\d+)\s+obj\b(.*?)endobj", re.S)


def decode_stream(d, raw):
    if b"/FlateDecode" in d or b"/Fl" in d:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            try:
                return zlib.decompressobj().decompress(raw)
            except zlib.error:
                return b""
    if b"/Filter" not in d:
        return raw
    return None  # other filters (images)


def load(path):
    b = open(path, "rb").read()
    objs = {}
    for m in OBJ.finditer(b):
        num = int(m.group(1))
        body = m.group(3)
        i = body.find(b"stream")
        if i >= 0:
            d = body[:i]
            s = body[i + 6:]
            s = s[2:] if s.startswith(b"\r\n") else s[1:] if s[:1] in (b"\n", b"\r") else s
            j = s.rfind(b"endstream")
            raw = s[:j] if j >= 0 else s
            objs[num] = (d, decode_stream(d, raw), raw)
        else:
            objs[num] = (body, None, None)
    # object streams
    for num, (d, data, raw) in list(objs.items()):
        if b"/ObjStm" in d and data:
            n = int(re.search(rb"/N\s+(\d+)", d).group(1))
            first = int(re.search(rb"/First\s+(\d+)", d).group(1))
            head = data[:first].split()
            ids = [(int(head[2 * k]), int(head[2 * k + 1])) for k in range(n)]
            for k, (oid, off) in enumerate(ids):
                end = ids[k + 1][1] if k + 1 < n else len(data) - first
                objs[oid] = (data[first + off:first + end], None, None)
    return objs


def ref(s, key):
    m = re.search(key + rb"\s+(\d+)\s+0\s+R", s)
    return int(m.group(1)) if m else None


def xobjects(objs, res):
    """{name: objnum} of an /XObject dictionary found in a resources dict (inline or by reference)."""
    out = {}
    m = re.search(rb"/XObject\s*(<<.*?>>|\d+\s+0\s+R)", res, re.S)
    if not m:
        return out
    t = m.group(1)
    if re.fullmatch(rb"\d+\s+0\s+R", t.strip()):
        t = objs.get(int(t.split()[0]), (b"", None, None))[0]
    for n, o in re.findall(rb"/([^\s/<>\[\]]+)\s+(\d+)\s+0\s+R", t):
        out[n] = int(o)
    return out


def scan(objs, content, xo, depth=0):
    """(path operators, text operators, [images (w, h)]) of one content stream, following form XObjects."""
    paths = len(re.findall(rb"(?<![A-Za-z0-9/.\-])(?:re|m|l|c|v|y)(?=[\s\r\n]|$)", content))
    texts = len(re.findall(rb"(?:Tj|TJ)(?=[\s\r\n]|$)", content))
    images = []
    for name in re.findall(rb"/([^\s/<>\[\]()]+)\s+Do\b", content):
        o = xo.get(name)
        if o is None or o not in objs:
            continue
        d, data, raw = objs[o]
        if b"/Subtype" in d and re.search(rb"/Subtype\s*/Image", d):
            w = re.search(rb"/Width\s+(\d+)", d)
            h = re.search(rb"/Height\s+(\d+)", d)
            images.append((int(w.group(1)) if w else 0, int(h.group(1)) if h else 0))
        elif re.search(rb"/Subtype\s*/Form", d) and data and depth < 4:
            sub_res = re.search(rb"/Resources\s*(<<.*?>>|\d+\s+0\s+R)", d, re.S)
            sub_xo = dict(xo)
            if sub_res:
                r = sub_res.group(1)
                if re.fullmatch(rb"\d+\s+0\s+R", r.strip()):
                    r = objs.get(int(r.split()[0]), (b"", None, None))[0]
                sub_xo.update(xobjects(objs, r))
            p, t, im = scan(objs, data, sub_xo, depth + 1)
            paths += p
            texts += t
            images += im
    return paths, texts, images


def pages(objs):
    out = []
    for num, (d, _, _) in objs.items():
        if re.search(rb"/Type\s*/Page(?![s\w])", d):
            out.append(num)
    return out


def page_order(objs):
    """Page object numbers in document order via the /Pages tree (falls back to object number order)."""
    root = None
    for num, (d, _, _) in objs.items():
        if re.search(rb"/Type\s*/Pages\b", d) and not re.search(rb"/Parent\s+\d+\s+0\s+R", d):
            root = num
    order = []

    def walk(n):
        d = objs[n][0]
        if re.search(rb"/Type\s*/Pages\b", d):
            kids = re.search(rb"/Kids\s*\[(.*?)\]", d, re.S)
            for k in re.findall(rb"(\d+)\s+0\s+R", kids.group(1)) if kids else []:
                if int(k) in objs:
                    walk(int(k))
        else:
            order.append(n)
    if root is not None:
        try:
            walk(root)
        except RecursionError:
            order = []
    return order or sorted(pages(objs))


def resources_of(objs, num):
    while num is not None and num in objs:
        d = objs[num][0]
        m = re.search(rb"/Resources\s*(<<.*>>|\d+\s+0\s+R)", d, re.S)
        if m:
            r = m.group(1)
            if re.fullmatch(rb"\d+\s+0\s+R", r.strip()):
                return objs.get(int(r.split()[0]), (b"", None, None))[0]
            return r
        num = ref(d, rb"/Parent")
    return b""






TOK = re.compile(rb"""
  (?P<ws>\s+)|(?P<cm>%[^\r\n]*)|(?P<hex><[0-9A-Fa-f\s]*>)|(?P<dstart><<)|(?P<dend>>>)|
  (?P<str>\((?:\\.|[^\\()]|\((?:\\.|[^\\()])*\))*\))|(?P<arr>[\[\]])|(?P<name>/[^\s/<>\[\]()%]*)|
  (?P<num>[+-]?(?:\d+\.?\d*|\.\d+))|(?P<op>[A-Za-z'"*]+)""", re.X)


def unescape(s):
    out = bytearray()
    i = 0
    s = s[1:-1]
    while i < len(s):
        c = s[i]
        if c == 0x5C and i + 1 < len(s):
            n = s[i + 1]
            m = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12, ord("("): 40, ord(")"): 41, 0x5C: 92}
            if n in m:
                out.append(m[n])
                i += 2
            elif 48 <= n <= 55:
                j = i + 1
                v = 0
                k = 0
                while j < len(s) and k < 3 and 48 <= s[j] <= 55:
                    v = v * 8 + s[j] - 48
                    j += 1
                    k += 1
                out.append(v & 255)
                i = j
            else:
                i += 2
        else:
            out.append(c)
            i += 1
    return bytes(out)


def tokens(data):
    pos = 0
    while pos < len(data):
        m = TOK.match(data, pos)
        if not m:
            pos += 1
            continue
        pos = m.end()
        k = m.lastgroup
        if k in ("ws", "cm"):
            continue
        v = m.group(k)
        if k == "num":
            yield ("num", float(v))
        elif k == "name":
            yield ("name", v[1:].decode("latin1"))
        elif k == "str":
            yield ("str", unescape(v))
        elif k == "hex":
            h = re.sub(rb"\s", b"", v[1:-1])
            if len(h) % 2:
                h += b"0"
            yield ("str", bytes.fromhex(h.decode()))
        elif k == "arr":
            yield ("arr", v.decode())
        elif k in ("dstart", "dend"):
            yield (k, None)
        else:
            op = v.decode("latin1")
            if op in ("BI",):
                e = data.find(b"EI", pos)
                pos = e + 2 if e >= 0 else len(data)
                continue
            yield ("op", op)


def mat_mul(a, b):
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5])


def apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def parse_cmap(data):
    out = {}
    for blk in re.findall(rb"beginbfchar(.*?)endbfchar", data, re.S):
        for a, b in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            out[int(a, 16)] = bytes.fromhex(b.decode()).decode("utf-16-be", "replace")
    for blk in re.findall(rb"beginbfrange(.*?)endbfrange", data, re.S):
        for a, b, c in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            lo, hi = int(a, 16), int(b, 16)
            base = int(c, 16)
            for k in range(lo, hi + 1):
                out[k] = chr(base + k - lo)
    return out


class Font:
    def __init__(self, objs, num):
        d = objs[num][0]
        self.two = b"/Type0" in d
        self.map = {}
        r = ref(d, rb"/ToUnicode")
        if r and r in objs and objs[r][1]:
            self.map = parse_cmap(objs[r][1])

    def decode(self, s):
        if self.two:
            return [(int.from_bytes(s[i:i + 2], "big"), self.map.get(int.from_bytes(s[i:i + 2], "big"), "?")) for i in range(0, len(s) - 1, 2)]
        return [(c, self.map.get(c, chr(c) if 32 <= c < 127 else "?")) for c in s]


def fonts_of(objs, res):
    out = {}
    m = re.search(rb"/Font\s*(<<.*?>>|\d+\s+0\s+R)", res, re.S)
    if not m:
        return out
    t = m.group(1)
    if re.fullmatch(rb"\d+\s+0\s+R", t.strip()):
        t = objs.get(int(t.split()[0]), (b"", None, None))[0]
    for n, o in re.findall(rb"/([^\s/<>\[\]]+)\s+(\d+)\s+0\s+R", t):
        if int(o) in objs:
            out[n.decode("latin1")] = Font(objs, int(o))
    return out


def interpret(objs, content, res, ctm=(1, 0, 0, 1, 0, 0), depth=0, state=None):
    """Return (paths, texts). paths: dicts {kind, color, width, pts(list of subpaths)}; texts: dicts {x, y, size, text}."""
    xo = xobjects(objs, res)
    fonts = fonts_of(objs, res)
    paths, texts = [], []
    st = state or {}
    gs = {"ctm": ctm, "stroke": (0, 0, 0), "fill": (0, 0, 0), "w": 1.0}
    stack = []
    cur = []  # current path: list of subpaths, each list of points (user space)
    sub = None
    ncurve = {}  # id(subpath) -> number of curve segments (a circle has four, a diamond none)
    ops = []
    tm = tlm = (1, 0, 0, 1, 0, 0)
    font, fsize, tl, tc, tw, th = None, 1.0, 0.0, 0.0, 0.0, 1.0
    clip_pending = False

    def finish(kind):
        nonlocal cur, sub
        if sub:
            cur.append(sub)
        if cur and kind != "n":
            pts = [[apply(gs["ctm"], x, y) for (x, y) in sp] for sp in cur]
            paths.append({"kind": kind, "color": gs["stroke"] if kind.startswith("S") else gs["fill"], "stroke": gs["stroke"], "fill": gs["fill"],
                          "width": gs["w"] * (abs(gs["ctm"][0]) + abs(gs["ctm"][3])) / 2, "pts": pts, "ncurve": [ncurve.get(id(sp), 0) for sp in cur]})
        for sp in cur:
            ncurve.pop(id(sp), None)
        cur = []
        sub = None

    for kind, val in tokens(content):
        if kind in ("num", "name", "str", "arr", "dstart", "dend"):
            ops.append((kind, val))
            continue
        op = val
        nums = [v for (k, v) in ops if k == "num"]
        try:
            if op == "q":
                stack.append((dict(gs), font, fsize))
            elif op == "Q":
                if stack:
                    gs, font, fsize = stack.pop()
            elif op == "cm" and len(nums) >= 6:
                gs["ctm"] = mat_mul(tuple(nums[-6:]), gs["ctm"])
            elif op == "w" and nums:
                gs["w"] = nums[-1]
            elif op in ("RG",) and len(nums) >= 3:
                gs["stroke"] = tuple(nums[-3:])
            elif op in ("rg",) and len(nums) >= 3:
                gs["fill"] = tuple(nums[-3:])
            elif op == "G" and nums:
                gs["stroke"] = (nums[-1],) * 3
            elif op == "g" and nums:
                gs["fill"] = (nums[-1],) * 3
            elif op in ("SC", "SCN") and len(nums) in (1, 3):
                gs["stroke"] = tuple(nums) if len(nums) == 3 else (nums[0],) * 3
            elif op in ("sc", "scn") and len(nums) in (1, 3):
                gs["fill"] = tuple(nums) if len(nums) == 3 else (nums[0],) * 3
            elif op == "m" and len(nums) >= 2:
                if sub:
                    cur.append(sub)
                sub = [(nums[-2], nums[-1])]
            elif op == "l" and len(nums) >= 2 and sub is not None:
                sub.append((nums[-2], nums[-1]))
            elif op == "c" and len(nums) >= 6 and sub is not None:
                sub.append((nums[-2], nums[-1]))
                ncurve[id(sub)] = ncurve.get(id(sub), 0) + 1
            elif op in ("v", "y") and len(nums) >= 4 and sub is not None:
                sub.append((nums[-2], nums[-1]))
                ncurve[id(sub)] = ncurve.get(id(sub), 0) + 1
            elif op == "h" and sub:
                sub.append(sub[0])
            elif op == "re" and len(nums) >= 4:
                x, y, w, h = nums[-4:]
                if sub:
                    cur.append(sub)
                cur.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
                sub = None
            elif op in ("S", "s"):
                if op == "s" and sub:
                    sub.append(sub[0])
                finish("S")
            elif op in ("f", "F", "f*"):
                finish("f")
            elif op in ("B", "B*", "b", "b*"):
                finish("S")
            elif op == "n":
                finish("n")
            elif op == "BT":
                tm = tlm = (1, 0, 0, 1, 0, 0)
            elif op == "Tf" and len(ops) >= 2:
                nm = [v for (k, v) in ops if k == "name"]
                if nm:
                    font = fonts.get(nm[-1])
                if nums:
                    fsize = nums[-1]
            elif op == "Td" and len(nums) >= 2:
                tlm = mat_mul((1, 0, 0, 1, nums[-2], nums[-1]), tlm)
                tm = tlm
            elif op == "TD" and len(nums) >= 2:
                tl = -nums[-1]
                tlm = mat_mul((1, 0, 0, 1, nums[-2], nums[-1]), tlm)
                tm = tlm
            elif op == "Tm" and len(nums) >= 6:
                tm = tlm = tuple(nums[-6:])
            elif op == "T*":
                tlm = mat_mul((1, 0, 0, 1, 0, -tl), tlm)
                tm = tlm
            elif op == "TL" and nums:
                tl = nums[-1]
            elif op in ("Tj", "TJ", "'", '"'):
                strs = [v for (k, v) in ops if k == "str"]
                items = []
                for k, v in ops:
                    if k == "str":
                        items.append(v)
                txt = ""
                for s in items:
                    if font:
                        txt += "".join(ch for (_, ch) in font.decode(s))
                    else:
                        txt += s.decode("latin1")
                full = mat_mul(tm, gs["ctm"])
                x, y = apply(full, 0, 0)
                texts.append({"x": x, "y": y, "size": fsize * (abs(full[3]) or 1), "text": txt, "font": bool(font)})
                if op == "'":
                    tlm = mat_mul((1, 0, 0, 1, 0, -tl), tlm)
                    tm = tlm
            elif op == "Do":
                nm = [v for (k, v) in ops if k == "name"]
                if nm and nm[-1].encode() in xo and depth < 5:
                    o = xo[nm[-1].encode()]
                    if o in objs:
                        d, data, raw = objs[o]
                        if re.search(rb"/Subtype\s*/Form", d) and data:
                            mm = re.search(rb"/Matrix\s*\[([^\]]*)\]", d)
                            fm = tuple(float(x) for x in mm.group(1).split()) if mm else (1, 0, 0, 1, 0, 0)
                            r2 = re.search(rb"/Resources\s*(<<.*>>|\d+\s+0\s+R)", d, re.S)
                            rr = res
                            if r2:
                                rr = r2.group(1)
                                if re.fullmatch(rb"\d+\s+0\s+R", rr.strip()):
                                    rr = objs.get(int(rr.split()[0]), (b"", None, None))[0]
                            p2, t2 = interpret(objs, data, rr, mat_mul(fm, gs["ctm"]), depth + 1)
                            paths += p2
                            texts += t2
        except Exception as e:  # keep going on odd operands
            st["errors"] = st.get("errors", 0) + 1
        ops = []
    return paths, texts


def page(path, n):
    objs = load(path)
    order = page_order(objs)
    pn = order[n - 1]
    d = objs[pn][0]
    cs = []
    m = re.search(rb"/Contents\s*\[(.*?)\]", d, re.S)
    if m:
        cs = [int(x) for x in re.findall(rb"(\d+)\s+0\s+R", m.group(1))]
    else:
        r = ref(d, rb"/Contents")
        if r:
            cs = [r]
    content = b"\n".join((objs[c][1] or b"") for c in cs if c in objs)
    res = resources_of(objs, pn)
    mb = re.search(rb"/MediaBox\s*\[([^\]]*)\]", d)
    return objs, content, res, (mb.group(1).decode() if mb else None)
