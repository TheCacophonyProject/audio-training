"""Summarise summit-road track tags + spectv10 predictions to CSV and ODS.

Rows are sorted by NZ recording time, with marker rows inserted at each
sunrise/sunset (fixed Christchurch times for 2026-11-19 used for every day).
"""

import argparse, csv, glob, json, os, subprocess, tempfile
from collections import defaultdict
from datetime import datetime, time, timedelta
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

NZ = ZoneInfo("Pacific/Auckland")
MODEL_KEY = "spectv10-fine-val_pr_auc"
TAG_MODELS = ["efficientnetv2bm-kokako", "efficientnetv2b3-rms", "Master", "human"]
SUNRISE = time(5, 52)  # Christchurch 2026-11-19 NZDT
SUNSET = time(20, 38)
TEXT_COLS = {"rec_id", "track_id"}

p = argparse.ArgumentParser()
p.add_argument("--dir", default="summit-road/metadata")
p.add_argument("--out", default="summit-road/summit-road-tracks.csv")
p.add_argument("--threshold", type=float, default=0.5)
p.add_argument(
    "--no-ods", action="store_true", help="skip writing the .ods spreadsheet"
)
a = p.parse_args()


def track_rows(rec, rec_dt):
    for t in rec.get("Tracks") or rec.get("tracks", []):
        tags = defaultdict(list)
        for g in t.get("tags", []):
            m = g.get("model") or ("human" if not g.get("automatic") else "unknown")
            conf = g.get("confidence")
            tags[m].append(g["what"] + (f" ({conf})" if conf is not None else ""))
        pred = t.get(MODEL_KEY)
        above, top, nwin = "", "", 0
        if pred and pred["windows"]:
            wins = pred["windows"]
            nwin = len(wins)
            sums = defaultdict(float)
            for w in wins:
                for k, v in w["scores"].items():
                    sums[k] += v
            ranked = sorted(
                ((k, v / nwin) for k, v in sums.items()), key=lambda kv: -kv[1]
            )
            above = ";".join(f"{k} ({v:.2f})" for k, v in ranked if v > a.threshold)
            top = f"{ranked[0][0]} ({ranked[0][1]:.2f})"
        row = {
            "rec_id": rec["id"],
            "recording_datetime_nz": rec_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": t["id"],
            "start": round(t["start"], 2),
            "end": round(t["end"], 2),
        }
        for m in TAG_MODELS + sorted(set(tags) - set(TAG_MODELS)):
            row[m] = ";".join(tags.get(m, []))
        row["spectv10_windows"] = nwin
        row[f"spectv10_mean>{a.threshold}"] = above
        row["spectv10_top"] = top
        yield (rec_dt + timedelta(seconds=t["start"]), t["id"]), row


recs = []
for f in glob.glob(os.path.join(a.dir, "*.txt")):
    rec = json.load(open(f))
    dt = datetime.fromisoformat(
        rec["recordingDateTime"].replace("Z", "+00:00")
    ).astimezone(NZ)
    recs.append((dt, rec))
recs.sort(key=lambda r: (r[0], r[1]["id"]))

items = [ir for dt, rec in recs for ir in track_rows(rec, dt)]
items.sort(key=lambda kv: kv[0])

# sun events spanning the data
events = []
if items:
    day = items[0][0][0].date() - timedelta(days=1)
    last = items[-1][0][0].date() + timedelta(days=1)
    while day <= last:
        for name, tm in (("SUNRISE", SUNRISE), ("SUNSET", SUNSET)):
            events.append((datetime.combine(day, tm, tzinfo=NZ), name))
        day += timedelta(days=1)
    events.sort()

rows, ei = [], 0
for (ts, _), row in items:
    passed = None
    while ei < len(events) and events[ei][0] <= ts:
        passed = events[ei]
        ei += 1
    if passed and rows:  # only mark boundaries between data, not before the first row
        ev_dt, name = passed
        rows.append(
            {
                "rec_id": f"---- {name} {ev_dt:%Y-%m-%d %H:%M} ----",
                "recording_datetime_nz": ev_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "_sep": True,
            }
        )
    rows.append(row)

fields = []
for r in rows:
    for k in r:
        if k not in fields and k != "_sep":
            fields.append(k)

with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, restval="", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
n_tracks = sum(1 for r in rows if not r.get("_sep"))
print(f"wrote {n_tracks} tracks + {len(rows) - n_tracks} sun markers to {a.out}")


def cell(k, v, bold=False):
    style = ' table:style-name="sep"' if bold else ""
    if v == "" or v is None:
        return "<table:table-cell/>"
    if isinstance(v, (int, float)) and k not in TEXT_COLS:
        return f'<table:table-cell{style} office:value-type="float" office:value="{v}"><text:p>{v}</text:p></table:table-cell>'
    return f'<table:table-cell{style} office:value-type="string"><text:p>{escape(str(v))}</text:p></table:table-cell>'


def write_ods(path):
    widths = {
        "rec_id": 2.2,
        "recording_datetime_nz": 4.0,
        "track_id": 2.2,
        "start": 1.6,
        "end": 1.6,
        "spectv10_windows": 3.2,
        "spectv10_top": 3.5,
    }
    cols = "".join(
        f'<style:style style:name="c{i}" style:family="table-column"><style:table-column-properties style:column-width="{widths.get(k, 4.5)}cm"/></style:style>'
        for i, k in enumerate(fields)
    )
    body = ['<table:table table:name="tracks">']
    body += [
        f'<table:table-column table:style-name="c{i}"/>' for i in range(len(fields))
    ]
    body.append(
        "<table:table-row>"
        + "".join(cell(k, k, True) for k in fields)
        + "</table:table-row>"
    )
    for r in rows:
        sep = r.get("_sep", False)
        if sep:
            body.append(
                "<table:table-row>"
                + cell("", r["rec_id"], True)
                + cell("", r["recording_datetime_nz"], True)
                + "</table:table-row>"
            )
        else:
            body.append(
                "<table:table-row>"
                + "".join(cell(k, r.get(k, "")) for k in fields)
                + "</table:table-row>"
            )
    body.append("</table:table>")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 office:version="1.2" office:mimetype="application/vnd.oasis.opendocument.spreadsheet">
<office:automatic-styles>{cols}
<style:style style:name="sep" style:family="table-cell"><style:table-cell-properties fo:background-color="#ffe599"/><style:text-properties fo:font-weight="bold"/></style:style>
</office:automatic-styles>
<office:body><office:spreadsheet>{"".join(body)}</office:spreadsheet></office:body></office:document>"""
    with tempfile.TemporaryDirectory() as td:
        fods = os.path.join(td, os.path.splitext(os.path.basename(path))[0] + ".fods")
        open(fods, "w").write(xml)
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "ods",
                "--outdir",
                os.path.dirname(os.path.abspath(path)),
                fods,
            ],
            check=True,
            capture_output=True,
        )
    print(f"wrote {path}")


if not a.no_ods:
    write_ods(os.path.splitext(a.out)[0] + ".ods")
