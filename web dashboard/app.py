# -*- coding: utf-8 -*-
import os, sys, io, hashlib, threading, webbrowser
from functools import lru_cache

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from skimage import filters, measure
from flask import Flask, Response, jsonify, request, send_file

DICOM_DIR  = r"E:/misp/all/new/1003056/DICOM/20250612/164412"
OUTPUT_DIR = r"E:/misp/all/new/dashboard_output"
EXCEL_NAME = "angles_records.xlsx"
HOST, PORT = "127.0.0.1", 5000

DEFAULT_SIGMA   = 1.0
DEFAULT_RATIO   = 0.4
MIN_CONTOUR_LEN = 20

DEMO = "--demo" in sys.argv
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXCEL_PATH = os.path.join(OUTPUT_DIR, EXCEL_NAME)

COLUMNS = ["User_ID", "Slice", "Label",
           "Apex_row", "Apex_col", "Pt1_row", "Pt1_col", "Pt2_row", "Pt2_col",
           "Angle_deg", "Turning_deg", "Best"]

LCOLUMNS = ["User_ID", "Slice", "Label", "Pts", "N_pts",
            "Apparent_px", "Real_mm", "Chord_mm", "RL"]


def make_demo_volume(n=20, h=400, w=400):
    vol = np.zeros((n, h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(n):
        cx = 200 + int(25 * np.sin(k / n * 2 * np.pi))
        stem = (abs(xx - cx) < 14) & (yy > 220)
        b1 = (abs((xx - cx) + (yy - 220) * 0.5) < 9) & (yy <= 220) & (yy > 90)
        b2 = (abs((xx - cx) - (yy - 220) * 0.5) < 7) & (yy <= 220) & (yy > 110)
        vol[k][stem | b1 | b2] = 800
    vol += np.random.RandomState(0).normal(0, 15, vol.shape).astype(np.float32)
    return vol


def segment_like_pipeline(img):
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorMagnitude(img)
    img = sitk.Cast(img, sitk.sitkFloat32)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    binary = otsu.Execute(img)
    binary = sitk.BinaryMorphologicalClosing(binary, [2] * 3)
    cc = sitk.ConnectedComponentImageFilter().Execute(binary)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = stats.GetLabels()
    sizes = [stats.GetNumberOfPixels(l) for l in labels]
    largest = labels[int(np.argmax(sizes))]
    return sitk.GetArrayFromImage(cc == largest).astype(bool)


def build_volume():
    if DEMO:
        arr = make_demo_volume()
        return arr, (1.0, 1.0, 1.0), arr > 400

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(DICOM_DIR)
    if not names:
        raise FileNotFoundError(f"DICOM پیدا نشد: {DICOM_DIR}")
    reader.SetFileNames(names)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)

    key = hashlib.md5((DICOM_DIR + str(len(names)) + str(arr.shape)).encode()).hexdigest()[:10]
    cache = os.path.join(OUTPUT_DIR, f"mask_{key}.npy")
    if os.path.exists(cache):
        refined = np.load(cache)
        print("mask از cache خوانده شد.")
    else:
        print("در حال سگمنت (فقط بار اول طول می‌کشد)...")
        refined = segment_like_pipeline(img)
        np.save(cache, refined)
    return arr, img.GetSpacing(), refined


print("در حال بارگذاری داده...")
ARR, SPACING, REFINED = build_volume()
N = ARR.shape[0]
H, W = ARR.shape[1], ARR.shape[2]
IS_RGB = ARR.ndim == 4
SP_X, SP_Y = float(SPACING[0]), float(SPACING[1])
if not IS_RGB:
    _sub = ARR[::2, ::4, ::4].astype(np.float32)
    VMIN, VMAX = np.percentile(_sub, 1), np.percentile(_sub, 99.5)
print(f"آماده: {N} اسلایس، اندازه {W}x{H}، spacing=({SP_X:.3f},{SP_Y:.3f}) mm، RGB={IS_RGB}")


def gray_slice(i):
    s = ARR[i]
    if s.ndim == 3:
        s = s[..., :3].mean(axis=-1)
    return s.astype(np.float32)


@lru_cache(maxsize=256)
def slice_png(i):
    s = ARR[i]
    if IS_RGB:
        im = Image.fromarray(np.clip(s[..., :3], 0, 255).astype(np.uint8), "RGB")
    else:
        g = np.clip((s.astype(np.float32) - VMIN) / (VMAX - VMIN + 1e-9), 0, 1)
        im = Image.fromarray((g * 255).astype(np.uint8), "L")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


@lru_cache(maxsize=512)
def contours_for(i, sigma, ratio):
    mask = REFINED[i].astype(bool)
    if not mask.any():
        return []
    g = gray_slice(i)
    masked = np.where(mask, g, 0.0).astype(np.float32)
    sm = filters.gaussian(masked, sigma=sigma) if sigma > 0 else masked
    thresh = float(sm[mask].mean() * ratio)
    binary = (sm > thresh) & mask
    cs = measure.find_contours(binary.astype(float), 0.5)
    cs = [c for c in cs if len(c) >= MIN_CONTOUR_LEN]
    cs.sort(key=len, reverse=True)
    return [np.round(c, 1).tolist() for c in cs]


def calc_angle(apex, p1, p2):
    a = np.array([apex[0] * SP_Y, apex[1] * SP_X], float)
    v1 = np.array([p1[0] * SP_Y, p1[1] * SP_X], float) - a
    v2 = np.array([p2[0] * SP_Y, p2[1] * SP_X], float) - a
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None, None
    ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))
    return round(ang, 2), round(180 - ang, 2)


def calc_length(pts):
    if len(pts) < 2:
        return None
    sp = np.array([SP_Y, SP_X], float)
    P = np.array(pts, float)
    seg = np.diff(P, axis=0)
    path_px = float(np.linalg.norm(seg, axis=1).sum())
    path_mm = float(np.linalg.norm(seg * sp, axis=1).sum())
    chord = P[-1] - P[0]
    chord_px = float(np.linalg.norm(chord))
    chord_mm = float(np.linalg.norm(chord * sp))
    if chord_mm < 1e-9:
        return None
    return {"path_px": round(path_px, 2), "path_mm": round(path_mm, 2),
            "chord_px": round(chord_px, 2), "chord_mm": round(chord_mm, 2),
            "rl": round(path_mm / chord_mm, 3)}


LOCK = threading.Lock()


def load_records():
    if os.path.exists(EXCEL_PATH):
        try:
            df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
            for c in COLUMNS:
                if c not in df.columns:
                    df[c] = None
            df = df[COLUMNS].copy()
            df["User_ID"] = df["User_ID"].astype(str)
            df["Label"] = df["Label"].astype(str)
            return df
        except Exception as e:
            print("خواندن Excel قبلی ناموفق بود:", e)
    return pd.DataFrame(columns=COLUMNS)


def load_length_records():
    if os.path.exists(EXCEL_PATH):
        try:
            df = pd.read_excel(EXCEL_PATH, sheet_name="Lengths", engine="openpyxl")
            if "Pts" not in df.columns and "P1_row" in df.columns:
                df["Pts"] = [f"{int(q.P1_row)},{int(q.P1_col)};{int(q.P2_row)},{int(q.P2_col)}"
                             for q in df.itertuples()]
                df["N_pts"] = 2
                df["Chord_mm"] = df["Real_mm"]
                df["RL"] = 1.0
            for c in LCOLUMNS:
                if c not in df.columns:
                    df[c] = None
            df = df[LCOLUMNS].copy()
            df["User_ID"] = df["User_ID"].astype(str)
            df["Label"] = df["Label"].astype(str)
            return df
        except ValueError:
            pass
        except Exception as e:
            print("خواندن شیت Lengths ناموفق بود:", e)
    return pd.DataFrame(columns=LCOLUMNS)


DF = load_records()
DF_L = load_length_records()


def save_records():
    try:
        with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
            DF.to_excel(writer, sheet_name="Angles", index=False)
            DF_L.to_excel(writer, sheet_name="Lengths", index=False)
        return True, ""
    except PermissionError:
        return False, "فایل Excel باز است؛ آن را ببندید (تغییر در حافظه حفظ شد و در ثبت بعدی ذخیره می‌شود)"
    except Exception as e:
        return False, str(e)


def parse_payload(d):
    uid = str(d.get("user_id", "")).strip()
    label = str(d.get("label", "")).strip()
    if not uid or not label:
        raise ValueError("User ID و Label باید پر باشند")
    sl = int(d["slice"])
    if not (0 <= sl < N):
        raise ValueError("شماره اسلایس نامعتبر است")
    pts = []
    for k in ("apex", "pt1", "pt2"):
        v = d.get(k)
        if not v or len(v) != 2:
            raise ValueError("هر سه نقطه باید انتخاب شوند")
        r, c = int(round(float(v[0]))), int(round(float(v[1])))
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError("نقطه خارج از تصویر است")
        pts.append((r, c))
    ang, turn = calc_angle(*pts)
    if ang is None:
        raise ValueError("دو نقطه روی هم افتاده‌اند؛ زاویه قابل محاسبه نیست")
    return {"User_ID": uid, "Slice": sl, "Label": label,
            "Apex_row": pts[0][0], "Apex_col": pts[0][1],
            "Pt1_row": pts[1][0], "Pt1_col": pts[1][1],
            "Pt2_row": pts[2][0], "Pt2_col": pts[2][1],
            "Angle_deg": ang, "Turning_deg": turn,
            "Best": 1 if int(d.get("best", 0)) else 0}


def parse_length_payload(d):
    uid = str(d.get("user_id", "")).strip()
    label = str(d.get("label", "")).strip()
    if not uid or not label:
        raise ValueError("User ID و Label باید پر باشند")
    sl = int(d["slice"])
    if not (0 <= sl < N):
        raise ValueError("شماره اسلایس نامعتبر است")
    raw = d.get("pts")
    if not raw or len(raw) < 2:
        raise ValueError("حداقل دو نقطه باید انتخاب شود")
    pts = []
    for v in raw:
        if not v or len(v) != 2:
            raise ValueError("مختصات نقطه نامعتبر است")
        r, c = int(round(float(v[0]))), int(round(float(v[1])))
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError("نقطه خارج از تصویر است")
        pts.append((r, c))
    res = calc_length(pts)
    if res is None:
        raise ValueError("نقطه ابتدا و انتها روی هم افتاده‌اند؛ طول نسبی قابل محاسبه نیست")
    return {"User_ID": uid, "Slice": sl, "Label": label,
            "Pts": ";".join(f"{r},{c}" for r, c in pts), "N_pts": len(pts),
            "Apparent_px": res["path_px"], "Real_mm": res["path_mm"],
            "Chord_mm": res["chord_mm"], "RL": res["rl"]}


app = Flask(__name__)


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/meta")
def meta():
    return jsonify(n_slices=N, width=W, height=H, sp_x=SP_X, sp_y=SP_Y, is_rgb=IS_RGB,
                   default_sigma=DEFAULT_SIGMA, default_ratio=DEFAULT_RATIO,
                   excel_path=os.path.abspath(EXCEL_PATH))


@app.route("/api/slice/<int:i>.png")
def api_slice(i):
    if not (0 <= i < N):
        return jsonify(error="slice out of range"), 404
    return Response(slice_png(i), mimetype="image/png")


@app.route("/api/contours/<int:i>")
def api_contours(i):
    if not (0 <= i < N):
        return jsonify(error="slice out of range"), 404
    sigma = round(float(request.args.get("sigma", DEFAULT_SIGMA)), 2)
    ratio = round(float(request.args.get("ratio", DEFAULT_RATIO)), 3)
    return jsonify(contours_for(i, sigma, ratio))


@app.route("/api/records", methods=["GET"])
def get_records():
    uid = request.args.get("user_id", "").strip()
    sub = DF[DF["User_ID"] == uid] if uid else DF
    out = sub.reset_index().rename(columns={"index": "idx"}).to_json(orient="records")
    return Response(out, mimetype="application/json")


@app.route("/api/records", methods=["POST"])
def add_record():
    global DF
    try:
        row = parse_payload(request.get_json(force=True))
    except Exception as e:
        return jsonify(error=str(e)), 400
    with LOCK:
        DF = pd.concat([DF, pd.DataFrame([row])], ignore_index=True)
        ok, msg = save_records()
    return jsonify(ok=True, saved=ok, warning=msg, angle=row["Angle_deg"], turning=row["Turning_deg"])


@app.route("/api/records/<int:idx>", methods=["PUT"])
def update_record(idx):
    try:
        row = parse_payload(request.get_json(force=True))
    except Exception as e:
        return jsonify(error=str(e)), 400
    with LOCK:
        if idx not in DF.index:
            return jsonify(error="رکورد پیدا نشد"), 404
        for k, v in row.items():
            DF.at[idx, k] = v
        ok, msg = save_records()
    return jsonify(ok=True, saved=ok, warning=msg, angle=row["Angle_deg"], turning=row["Turning_deg"])


@app.route("/api/records/<int:idx>/best", methods=["PATCH"])
def set_best(idx):
    with LOCK:
        if idx not in DF.index:
            return jsonify(error="رکورد پیدا نشد"), 404
        DF.at[idx, "Best"] = 1 if int(request.get_json(force=True).get("best", 0)) else 0
        ok, msg = save_records()
    return jsonify(ok=True, saved=ok, warning=msg)


@app.route("/api/records/<int:idx>", methods=["DELETE"])
def delete_record(idx):
    global DF
    with LOCK:
        if idx not in DF.index:
            return jsonify(error="رکورد پیدا نشد"), 404
        DF = DF.drop(idx).reset_index(drop=True)
        ok, msg = save_records()
    return jsonify(ok=True, saved=ok, warning=msg)


@app.route("/api/lengths", methods=["GET"])
def get_lengths():
    uid = request.args.get("user_id", "").strip()
    sub = DF_L[DF_L["User_ID"] == uid] if uid else DF_L
    out = sub.reset_index().rename(columns={"index": "idx"}).to_json(orient="records")
    return Response(out, mimetype="application/json")


@app.route("/api/lengths", methods=["POST"])
def add_length():
    global DF_L
    try:
        row = parse_length_payload(request.get_json(force=True))
    except Exception as e:
        return jsonify(error=str(e)), 400
    with LOCK:
        DF_L = pd.concat([DF_L, pd.DataFrame([row])], ignore_index=True)
        ok, msg = save_records()
    return jsonify(ok=True, saved=ok, warning=msg, apparent=row["Apparent_px"], real=row["Real_mm"], rl=row["RL"])


@app.route("/api/lengths/<int:idx>", methods=["PUT"])
def update_length(idx):
    try:
        row = parse_length_payload(request.get_json(force=True))
    except Exception as e:
        return jsonify(error=str(e)), 400
    with LOCK:
        if idx not in DF_L.index:
            return jsonify(error="رکورد پیدا نشد"), 404
        for k, v in row.items():
            DF_L.at[idx, k] = v
        ok, msg = save_records()
    return jsonify(ok=True, saved=ok, warning=msg, apparent=row["Apparent_px"], real=row["Real_mm"], rl=row["RL"])


@app.route("/api/lengths/<int:idx>", methods=["DELETE"])
def delete_length(idx):
    global DF_L
    with LOCK:
        if idx not in DF_L.index:
            return jsonify(error="رکورد پیدا نشد"), 404
        DF_L = DF_L.drop(idx).reset_index(drop=True)
        ok, msg = save_records()
    return jsonify(ok=True, saved=ok, warning=msg)


@app.route("/api/download")
def download():
    with LOCK:
        save_records()
    return send_file(os.path.abspath(EXCEL_PATH), as_attachment=True, download_name=EXCEL_NAME)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="fa">
<head>
<meta charset="utf-8">
<title>Angle Annotation Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  color-scheme:light;
  --bg:#F4F6F8;--panel:#FFFFFF;--panel2:#F8FAFB;--panel3:#EDF2F5;
  --line:#E2E8ED;--line2:#CFD9E0;
  --txt:#253443;--mut:#82909C;--dim:#A3AEB8;
  --acc:#167D8D;--acc2:#547C9D;--acc-soft:rgba(22,125,141,.12);
  --ok:#2E9E6B;--bad:#D64B54;--warn:#B7791F;
  --r-lg:16px;--r-md:11px;--r-sm:8px;
  --shadow:0 1px 2px rgba(37,52,67,.04),0 10px 30px -14px rgba(37,52,67,.16);
  --mono:"JetBrains Mono",Consolas,monospace;
  --sans:"Vazirmatn",Tahoma,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;color:var(--txt);font-family:var(--sans);font-size:13px;line-height:1.55;background:var(--bg)}
::selection{background:rgba(22,125,141,.22)}
*::-webkit-scrollbar{width:9px;height:9px}
*::-webkit-scrollbar-track{background:transparent}
*::-webkit-scrollbar-thumb{background:var(--line2);border-radius:99px;border:2px solid transparent;background-clip:padding-box}
*::-webkit-scrollbar-thumb:hover{background:#B7C4CE;background-clip:padding-box;border:2px solid transparent}
button:focus-visible,input:focus-visible,.nav:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.hid{display:none!important}
.grp{display:contents}

header{display:flex;align-items:center;gap:10px;padding:10px 22px;direction:rtl;flex-wrap:wrap;
  background:rgba(255,255,255,.88);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid var(--line);position:sticky;top:0;z-index:20}
header h1{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:700;margin:0 0 0 14px;letter-spacing:.2px;direction:ltr;color:var(--txt)}
header h1 small{font-weight:400;color:var(--mut);font-size:11.5px;margin-left:2px}
.chip{display:inline-flex;align-items:center;gap:7px;background:var(--panel2);border:1px solid var(--line);
  border-radius:999px;padding:4px 14px;color:var(--mut);font-size:12px}
.chip b{color:var(--acc);font-family:var(--mono);font-weight:600;font-size:12px;direction:ltr}
#btnSession{margin-right:auto}

#app{display:grid;grid-template-columns:minmax(520px,1.12fr) minmax(480px,1fr);gap:16px;padding:16px 22px 26px;direction:ltr;align-items:start}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-lg);padding:16px;direction:rtl;box-shadow:var(--shadow)}

.seg{display:flex;gap:4px;padding:4px;background:var(--panel3);border-radius:var(--r-md);margin-bottom:10px;direction:rtl}
.seg button{flex:1;border:0;background:transparent;padding:7px;border-radius:8px;font-family:inherit;font-size:13px;font-weight:600;color:var(--mut);cursor:pointer;transition:background .15s,color .15s}
.seg button.on{background:var(--panel);color:var(--acc);box-shadow:0 1px 3px rgba(37,52,67,.12)}

#banner{display:flex;align-items:center;justify-content:center;gap:10px;font-size:14.5px;font-weight:600;text-align:center;
  padding:11px 14px;border-radius:var(--r-md);background:var(--panel2);border:1px solid var(--line);
  color:var(--mut);margin-bottom:10px;min-height:46px;transition:background .25s,border-color .25s,color .25s}
#banner::before{content:"";flex:none;width:9px;height:9px;border-radius:50%;background:var(--dim)}
#banner.pick{background:rgba(22,125,141,.09);border-color:rgba(22,125,141,.45);color:#0F5F6C}
#banner.pick::before{background:var(--acc);animation:pulse 1.6s infinite}
#banner.ready{background:rgba(46,158,107,.09);border-color:rgba(46,158,107,.45);color:#1F7650}
#banner.ready::before{background:var(--ok)}
#banner.ask{background:rgba(183,121,31,.09);border-color:rgba(183,121,31,.45);color:#8A5A12}
#banner.ask::before{background:var(--warn)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(22,125,141,.55)}70%{box-shadow:0 0 0 9px rgba(22,125,141,0)}100%{box-shadow:0 0 0 0 rgba(22,125,141,0)}}
@media (prefers-reduced-motion:reduce){#banner.pick::before{animation:none}*{transition:none!important}}

#sliceTitle{text-align:center;font-family:var(--mono);font-size:19px;font-weight:700;letter-spacing:.4px;margin:2px 0 10px;direction:ltr;color:var(--txt)}

#stage{display:flex;gap:10px;align-items:stretch;direction:ltr}
.nav{width:46px;font-size:16px;background:var(--panel);color:var(--acc2);border:1px solid var(--line2);border-radius:var(--r-md);cursor:pointer;transition:all .15s}
.nav:hover{background:var(--acc-soft);color:var(--acc);border-color:var(--acc)}
.nav:active{transform:scale(.97)}
#wrap{flex:1;position:relative;height:calc(100vh - 360px);min-height:420px;background:#000;
  border:1px solid var(--line2);border-radius:var(--r-md);overflow:hidden;
  box-shadow:0 12px 30px -16px rgba(37,52,67,.45)}
#cv{position:absolute;left:0;top:0;display:block}
#cv.picking{cursor:crosshair}

.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;align-items:center;direction:rtl}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;background:var(--panel);color:var(--txt);
  border:1px solid var(--line2);border-radius:var(--r-sm);padding:7px 15px;cursor:pointer;font-family:inherit;font-size:13px;font-weight:500;
  transition:background .15s,border-color .15s,transform .08s,box-shadow .15s}
.btn:hover{background:var(--panel3);border-color:#B7C4CE}
.btn:active{transform:translateY(1px)}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none}
.btn.primary{background:linear-gradient(135deg,#167D8D,#547C9D);border-color:transparent;color:#fff;box-shadow:0 8px 18px -10px rgba(22,125,141,.8)}
.btn.primary:hover{filter:brightness(1.08);background:linear-gradient(135deg,#167D8D,#547C9D)}
.btn.ok{background:var(--ok);border-color:transparent;color:#fff;font-weight:600;box-shadow:0 8px 18px -12px rgba(46,158,107,.9)}
.btn.ok:hover{filter:brightness(1.08);background:var(--ok)}
.btn.bad{background:rgba(214,75,84,.08);border-color:rgba(214,75,84,.35);color:var(--bad)}
.btn.bad:hover{background:rgba(214,75,84,.16)}
.btn.sm{padding:3px 10px;font-size:12px;border-radius:7px}
label.opt{display:flex;align-items:center;gap:6px;color:var(--mut);font-size:12.5px;padding:3px 10px;border:1px solid var(--line);border-radius:999px;background:var(--panel2)}
label.opt input[type=checkbox]{accent-color:var(--acc);width:14px;height:14px;margin:0}
input[type=number],input[type=text]{background:var(--panel);color:var(--txt);border:1px solid var(--line2);border-radius:7px;padding:5px 8px;font-family:var(--mono);font-size:12.5px;transition:border-color .15s,box-shadow .15s}
input[type=number]:focus,input[type=text]:focus{outline:none;border-color:var(--acc);box-shadow:0 0 0 3px var(--acc-soft)}
#status{margin-top:10px;color:var(--mut);direction:ltr;font-family:var(--mono);font-size:11.5px;padding:7px 12px;background:var(--panel2);border:1px solid var(--line);border-radius:var(--r-sm)}
.hint{color:var(--mut);font-size:11.5px;margin-top:8px}

h3{margin:2px 0 10px;font-size:13.5px;font-weight:700;color:var(--txt);display:flex;align-items:center;gap:8px}
h3::before{content:"";width:3px;height:14px;border-radius:2px;background:linear-gradient(var(--acc),var(--acc2))}
table{border-collapse:separate;border-spacing:0;width:100%;direction:ltr}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:center;white-space:nowrap}
th{background:var(--panel2);position:sticky;top:0;color:var(--mut);font-weight:600;font-size:11.5px;z-index:1}
#ptsTable,#lptsTable{border:1px solid var(--line);border-radius:var(--r-md);overflow:hidden}
#ptsTable tbody tr:last-child td,#lptsTable tbody tr:last-child td{border-bottom:none}
#ptsTable input,#lptsTable input{width:74px;text-align:center}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;box-shadow:0 0 0 3px rgba(37,52,67,.08)}
#angleBox,#lenBox{margin-top:12px;font-size:14px;direction:ltr;text-align:center;padding:12px;border-radius:var(--r-md);
  background:rgba(22,125,141,.07);border:1px solid rgba(22,125,141,.28);color:var(--mut)}
#angleBox b,#lenBox b{font-family:var(--mono);font-size:20px;font-weight:700;color:var(--acc);margin:0 3px}
#bestBox{display:none;margin-top:12px;padding:14px;border:1px solid rgba(183,121,31,.45);border-radius:var(--r-md);
  background:rgba(183,121,31,.07);text-align:center}
#bestText{font-weight:600;color:#8A5A12}
#bestBox .row{display:flex;gap:10px;justify-content:center;margin-top:10px}
#recWrap,#lrecWrap{max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:var(--r-md);background:var(--panel)}
#recTable td,#lrecTable td{font-family:var(--mono);font-size:12px;color:var(--txt)}
#recTable tbody tr,#lrecTable tbody tr{cursor:pointer;transition:background .12s}
#recTable tbody tr:nth-child(even),#lrecTable tbody tr:nth-child(even){background:var(--panel2)}
#recTable tbody tr:hover,#lrecTable tbody tr:hover{background:rgba(22,125,141,.08)}
#recTable tbody tr:last-child td,#lrecTable tbody tr:last-child td{border-bottom:none}
.bestbtn{min-width:34px;padding:2px 8px;border-radius:999px;border:1px solid var(--line2);background:var(--panel);color:var(--mut);cursor:pointer;font-family:var(--mono);font-weight:700;transition:all .15s}
.bestbtn:hover{border-color:var(--acc);color:var(--acc)}
.bestbtn.on{background:rgba(46,158,107,.12);color:var(--ok);border-color:rgba(46,158,107,.5)}

#modalBg{position:fixed;inset:0;background:rgba(37,52,67,.45);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;z-index:50}
#modal{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:26px;width:410px;max-width:92vw;direction:rtl;
  box-shadow:0 30px 70px -20px rgba(37,52,67,.45)}
#modal h3{font-size:17px;margin-bottom:16px}
#modal label{display:block;color:var(--mut);font-size:12.5px}
#modal input{width:100%;margin:6px 0 14px;padding:10px 12px;font-size:13.5px}
#modal .btn{padding:10px;font-size:14px}

#toasts{position:fixed;bottom:18px;left:18px;display:flex;flex-direction:column;gap:8px;z-index:60;direction:rtl}
.toast{background:var(--panel);color:var(--txt);border:1px solid var(--line2);border-right:3px solid var(--acc);padding:10px 16px;border-radius:var(--r-sm);box-shadow:0 14px 30px -14px rgba(37,52,67,.35);animation:tin .22s ease-out}
.toast.err{border-right-color:var(--bad);background:#FFF5F5}
.toast.warn{border-right-color:var(--warn);background:#FFFAF0}
@keyframes tin{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

@media (max-width:1100px){#app{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none"><path d="M5 21 L13 5 L21 21" stroke="#167D8D" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M8.5 14.5 A6 6 0 0 0 17.5 14.5" stroke="#547C9D" stroke-width="1.8" stroke-linecap="round"/><circle cx="13" cy="5" r="2.2" fill="#E5484D"/></svg>
    Angle Annotation <small>Dashboard</small>
  </h1>
  <span class="chip">User ID <b id="hUser">-</b></span>
  <span class="chip">Label <b id="hLabel">-</b></span>
  <button class="btn sm" id="btnSession">تغییر لیبل / کاربر</button>
</header>

<div id="app">
  <section class="panel" id="viewer">
    <div class="seg">
      <button id="modeAngle" class="on">اندازه‌گیری زاویه</button>
      <button id="modeLen">اندازه‌گیری طول</button>
    </div>
    <div id="banner">...</div>
    <div id="sliceTitle">Slice -</div>
    <div id="stage">
      <button class="nav" id="prev" title="اسلایس قبلی (←)">◀</button>
      <div id="wrap"><canvas id="cv"></canvas></div>
      <button class="nav" id="next" title="اسلایس بعدی (→)">▶</button>
    </div>
    <div class="toolbar">
      <span class="grp" id="angTools">
        <button class="btn primary" id="btnPick">انتخاب نقطه</button>
        <button class="btn ok" id="btnConfirm" disabled>تایید نقاط</button>
        <button class="btn" id="btnClear">پاک کردن نقاط</button>
      </span>
      <span class="grp hid" id="lenTools">
        <button class="btn primary" id="btnLPick">انتخاب نقاط</button>
        <button class="btn" id="btnLFinish" disabled>پایان مسیر</button>
        <button class="btn ok" id="btnLSave" disabled>ثبت طول</button>
        <button class="btn" id="btnLClear">پاک کردن نقاط</button>
        <label class="opt"><input type="checkbox" id="chkPath"> مسیر منحنی (نقاط میانی)</label>
      </span>
      <button class="btn" id="btnFit">Fit</button>
      <label class="opt"><input type="checkbox" id="chkContour" checked> Contour</label>
      <label class="opt"><input type="checkbox" id="chkSaved" checked> نقاط ثبت‌شده</label>
      <label class="opt">sigma <input type="number" id="sigma" step="0.1" min="0" style="width:62px"></label>
      <label class="opt">thresh <input type="number" id="ratio" step="0.05" min="0.01" style="width:62px"></label>
    </div>
    <div id="status">row: -, col: -  |  zoom: -</div>
    <div class="hint">چرخ ماوس = زوم | کشیدن روی تصویر = جابجایی | کشیدن نقطه = اصلاح | ←/→ = اسلایس | Esc = لغو</div>
  </section>

  <aside class="panel" id="side">
    <div id="angSide">
      <h3>نقاط اسلایس جاری</h3>
      <table id="ptsTable">
        <thead><tr><th>نقطه</th><th>row</th><th>col</th><th></th></tr></thead>
        <tbody></tbody>
      </table>
      <div id="angleBox">-</div>
      <div id="bestBox">
        <div id="bestText">آیا این زاویه بهترین است؟</div>
        <div class="row">
          <button class="btn ok" id="bestYes">Yes (1)</button>
          <button class="btn bad" id="bestNo">No (0)</button>
          <button class="btn" id="bestCancel">لغو</button>
        </div>
      </div>

      <h3 style="margin-top:20px;justify-content:space-between">
        <span style="display:flex;align-items:center;gap:8px">جدول ثبت‌ها</span>
        <button class="btn sm" id="btnDownload">دانلود Excel</button>
      </h3>
      <div id="recWrap">
        <table id="recTable">
          <thead><tr>
            <th>User_ID</th><th>Slice</th><th>Label</th>
            <th>Apex_row</th><th>Apex_col</th><th>Pt1_row</th><th>Pt1_col</th><th>Pt2_row</th><th>Pt2_col</th>
            <th>Angle</th><th>Turning</th><th>Best</th><th></th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div id="lenSide" class="hid">
      <h3>نقاط اسلایس جاری</h3>
      <table id="lptsTable">
        <thead><tr><th>نقطه</th><th>row</th><th>col</th><th></th></tr></thead>
        <tbody></tbody>
      </table>
      <div id="lenBox">-</div>

      <h3 style="margin-top:20px;justify-content:space-between">
        <span style="display:flex;align-items:center;gap:8px">جدول ثبت‌های طول</span>
        <button class="btn sm" id="btnLDownload">دانلود Excel</button>
      </h3>
      <div id="lrecWrap">
        <table id="lrecTable">
          <thead><tr>
            <th>User_ID</th><th>Slice</th><th>Label</th>
            <th>N_pts</th><th>Apparent (px)</th><th>Real (mm)</th><th>Chord (mm)</th><th>RL</th><th></th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    <div class="hint" id="excelPath"></div>
  </aside>
</div>

<div id="modalBg">
  <div id="modal">
    <h3>شروع کار</h3>
    <label>User ID</label>
    <input type="text" id="inUser" placeholder="مثلاً P001">
    <label>Label (تا زمانی که تغییر ندهید روی همه ثبت‌ها می‌افتد)</label>
    <input type="text" id="inLabel" placeholder="مثلاً ICA-ECA bifurcation">
    <button class="btn primary" id="btnStart" style="width:100%">شروع</button>
  </div>
</div>
<div id="toasts"></div>

<script>
const S={meta:null,slice:0,userId:'',label:'',images:{},contourCache:{},curImg:null,curContours:[],
 view:{scale:1,tx:0,ty:0},cw:0,ch:0,dpr:1,pending:{},picking:false,repickIdx:null,editingIdx:null,
 awaitBest:false,records:[],sigma:1,ratio:0.4,showContour:true,showSaved:true,mouse:null,
 mode:'angle',lpending:{},lpicking:false,lrepickIdx:null,leditingIdx:null,lrecords:[],pathMode:false};
const $=id=>document.getElementById(id);
const cv=$('cv'),ctx=cv.getContext('2d');
const NAMES=['راس (Apex)','نقطه دوم (Pt1)','نقطه سوم (Pt2)'];
const COLORS=['#ff3b3b','#27e1ff','#ffa927'];
const LORANGE='#F5A524';
function lname(k,n,done){if(k===0)return'مبدا (Vessel origin)';if(k===n-1&&n>=2&&done)return'انتها (Distal end)';return'نقطه مسیر '+k;}
const JH={'Content-Type':'application/json'};

async function api(url,opt){
  const r=await fetch(url,opt);let j=null;
  try{j=await r.json()}catch(e){}
  if(!r.ok)throw new Error((j&&j.error)||('HTTP '+r.status));
  return j;
}
function toast(msg,kind){
  const d=document.createElement('div');d.className='toast '+(kind||'');d.textContent=msg;
  $('toasts').appendChild(d);setTimeout(()=>d.remove(),kind?5500:2600);
}

function calcAngle(a,p1,p2){
  const sy=S.meta.sp_y,sx=S.meta.sp_x;
  const A=[a[0]*sy,a[1]*sx],v1=[p1[0]*sy-A[0],p1[1]*sx-A[1]],v2=[p2[0]*sy-A[0],p2[1]*sx-A[1]];
  const n1=Math.hypot(...v1),n2=Math.hypot(...v2);if(n1<1e-9||n2<1e-9)return null;
  const c=Math.max(-1,Math.min(1,(v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)));
  const ang=Math.acos(c)*180/Math.PI;return{angle:ang,turning:180-ang};
}
function calcLen(pts){
  if(!pts||pts.length<2||pts.some(p=>!p))return null;
  const sy=S.meta.sp_y,sx=S.meta.sp_x;
  let px=0,mm=0;
  for(let i=1;i<pts.length;i++){
    const dr=pts[i][0]-pts[i-1][0],dc=pts[i][1]-pts[i-1][1];
    px+=Math.hypot(dr,dc);mm+=Math.hypot(dr*sy,dc*sx);
  }
  const a=pts[0],b=pts[pts.length-1];
  const cr=b[0]-a[0],cc=b[1]-a[1];
  const chordPx=Math.hypot(cr,cc),chordMm=Math.hypot(cr*sy,cc*sx);
  if(chordMm<1e-9)return null;
  return{px:px,mm:mm,chordPx:chordPx,chordMm:chordMm,rl:mm/chordMm};
}
function parsePts(str){return String(str||'').split(';').filter(Boolean).map(t=>t.split(',').map(Number));}
function currentPts(){if(!S.pending[S.slice])S.pending[S.slice]=[null,null,null];return S.pending[S.slice];}
function allSet(){return currentPts().every(p=>p);}
function currentLen(){if(!S.lpending[S.slice])S.lpending[S.slice]=[];return S.lpending[S.slice];}
function allLenSet(){return currentLen().length>=2&&!S.lpicking;}
function isPicking(){return S.mode==='length'?(S.lpicking||S.lrepickIdx!==null):(S.picking||S.repickIdx!==null);}
function toScr(r,c){const v=S.view;return[(c+.5)*v.scale+v.tx,(r+.5)*v.scale+v.ty];}
function toImg(x,y){const v=S.view;return[(y-v.ty)/v.scale-.5,(x-v.tx)/v.scale-.5];}
function inBounds(r,c){return r>=0&&c>=0&&r<=S.meta.height-1&&c<=S.meta.width-1;}

function loadImage(i){
  if(S.images[i])return Promise.resolve(S.images[i]);
  return new Promise((res,rej)=>{const im=new Image();im.onload=()=>{S.images[i]=im;res(im)};im.onerror=rej;im.src='/api/slice/'+i+'.png';});
}
async function loadContours(i){
  const key=i+'|'+S.sigma+'|'+S.ratio;
  if(!S.contourCache[key])S.contourCache[key]=await api(`/api/contours/${i}?sigma=${S.sigma}&ratio=${S.ratio}`);
  return S.contourCache[key];
}
async function gotoSlice(i,keepAwait){
  i=Math.max(0,Math.min(S.meta.n_slices-1,i));
  S.slice=i;if(!keepAwait)S.awaitBest=false;
  S.picking=false;S.repickIdx=null;S.lpicking=false;S.lrepickIdx=null;
  updateUI();
  try{
    const [im,cs]=await Promise.all([loadImage(i),loadContours(i)]);
    if(S.slice!==i)return;
    S.curImg=im;S.curContours=cs;draw();updateUI();
  }catch(e){toast('خطا در بارگذاری اسلایس: '+e.message,'err');}
}
async function loadRecords(){
  S.records=await api('/api/records?user_id='+encodeURIComponent(S.userId));
  S.lrecords=await api('/api/lengths?user_id='+encodeURIComponent(S.userId));
  renderRecords();renderLenRecords();draw();
}

function resizeCanvas(){
  const w=$('wrap');S.dpr=window.devicePixelRatio||1;S.cw=w.clientWidth;S.ch=w.clientHeight;
  cv.style.width=S.cw+'px';cv.style.height=S.ch+'px';
  cv.width=Math.round(S.cw*S.dpr);cv.height=Math.round(S.ch*S.dpr);
}
function fit(){
  const W=S.meta.width,H=S.meta.height,s=Math.min(S.cw/W,S.ch/H);
  S.view={scale:s,tx:(S.cw-W*s)/2,ty:(S.ch-H*s)/2};
}
function drawArms(pts,alpha,rad,dashed,showText){
  const [a,p1,p2]=pts;ctx.save();ctx.globalAlpha=alpha;ctx.lineWidth=2;
  if(dashed)ctx.setLineDash([6,4]);
  if(a&&p1){const A=toScr(...a),B=toScr(...p1);ctx.strokeStyle=COLORS[1];ctx.beginPath();ctx.moveTo(...A);ctx.lineTo(...B);ctx.stroke();}
  if(a&&p2){const A=toScr(...a),B=toScr(...p2);ctx.strokeStyle=COLORS[2];ctx.beginPath();ctx.moveTo(...A);ctx.lineTo(...B);ctx.stroke();}
  ctx.setLineDash([]);
  pts.forEach((p,k)=>{if(!p)return;const [x,y]=toScr(...p);
    ctx.fillStyle=COLORS[k];ctx.strokeStyle='#000';ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(x,y,rad,0,7);ctx.fill();ctx.stroke();});
  if(showText&&a&&p1&&p2){const r=calcAngle(a,p1,p2);if(r){const [x,y]=toScr(...a);
    ctx.font='bold 14px Tahoma';ctx.fillStyle='#fff';ctx.strokeStyle='#000';ctx.lineWidth=3;
    const t=r.angle.toFixed(1)+'°';ctx.strokeText(t,x+10,y-10);ctx.fillText(t,x+10,y-10);}}
  ctx.restore();
}
function lenTag(text,x,y){
  ctx.font='600 12px Tahoma';const w=ctx.measureText(text).width+14;
  ctx.fillStyle='rgba(0,0,0,.88)';ctx.beginPath();
  if(ctx.roundRect)ctx.roundRect(x,y-11,w,22,4);else ctx.rect(x,y-11,w,22);
  ctx.fill();ctx.fillStyle='#fff';ctx.textBaseline='middle';ctx.fillText(text,x+7,y);ctx.textBaseline='alphabetic';
}
function drawLenSeg(pts,alpha,rad,dashed,tags,done){
  ctx.save();ctx.globalAlpha=alpha;ctx.lineJoin='round';
  const sp=pts.map(p=>toScr(...p));
  if(sp.length>=2){
    if(sp.length>2){
      ctx.strokeStyle='#ffffff';ctx.lineWidth=2;if(dashed)ctx.setLineDash([6,4]);
      ctx.beginPath();sp.forEach((q,k)=>k?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.stroke();ctx.setLineDash([]);
    }
    ctx.strokeStyle=LORANGE;ctx.lineWidth=2.5;if(dashed)ctx.setLineDash([6,4]);
    const e=sp[sp.length-1];ctx.beginPath();ctx.moveTo(sp[0][0],sp[0][1]);ctx.lineTo(e[0],e[1]);ctx.stroke();ctx.setLineDash([]);
  }
  sp.forEach((q,k)=>{const end=k===0||k===sp.length-1;
    ctx.fillStyle=LORANGE;ctx.strokeStyle='#000';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.arc(q[0],q[1],end?rad:rad-1.5,0,7);ctx.fill();ctx.stroke();});
  if(tags){
    if(sp.length>=1)lenTag('Vessel origin',sp[0][0]+12,sp[0][1]);
    if(sp.length>=2&&done)lenTag('Distal end',sp[sp.length-1][0]+12,sp[sp.length-1][1]);
  }
  const r=calcLen(pts);
  if(r){const a=sp[0],b=sp[sp.length-1];
    const lines=[`Apparent length: ${r.px.toFixed(2)} px`,`Real length: ${r.mm.toFixed(2)} mm`,`RL: ${r.rl.toFixed(2)}`];
    const mx=(a[0]+b[0])/2+12,my=(a[1]+b[1])/2-(lines.length-1)*12;
    lines.forEach((t,k)=>lenTag(t,mx,my+k*24));}
  ctx.restore();
}
function drawLenLayer(){
  if(S.showSaved){
    S.lrecords.filter(r=>r.Slice===S.slice&&r.idx!==S.leditingIdx).forEach(r=>{
      drawLenSeg(parsePts(r.Pts),.6,4,true,false,true);});
  }
  drawLenSeg(currentLen(),1,6,false,true,!S.lpicking);
}
function draw(){
  if(!S.meta)return;
  ctx.setTransform(S.dpr,0,0,S.dpr,0,0);ctx.fillStyle='#000';ctx.fillRect(0,0,S.cw,S.ch);
  if(!S.curImg)return;
  const v=S.view,W=S.meta.width,H=S.meta.height;
  ctx.imageSmoothingEnabled=v.scale<1;
  ctx.drawImage(S.curImg,v.tx,v.ty,W*v.scale,H*v.scale);
  if(S.showContour){
    ctx.strokeStyle='#ff2a2a';ctx.lineWidth=1;ctx.lineJoin='round';
    for(const c of S.curContours){ctx.beginPath();
      for(let k=0;k<c.length;k++){const [x,y]=toScr(c[k][0],c[k][1]);k?ctx.lineTo(x,y):ctx.moveTo(x,y);}
      ctx.stroke();}
  }
  if(S.mode==='length'){drawLenLayer();$('status').textContent=statusText();return;}
  if(S.showSaved){
    S.records.filter(r=>r.Slice===S.slice&&r.idx!==S.editingIdx).forEach(r=>{
      drawArms([[r.Apex_row,r.Apex_col],[r.Pt1_row,r.Pt1_col],[r.Pt2_row,r.Pt2_col]],.6,3,true,true);});
  }
  drawArms(currentPts(),1,5,false,true);
  $('status').textContent=statusText();
}
function statusText(){
  const m=S.mouse&&S.mouse.cur;
  return `row: ${m?m[0]:'-'}, col: ${m?m[1]:'-'}  |  zoom: ×${S.view.scale.toFixed(2)}  |  contour sigma=${S.sigma}, thresh=${S.ratio}`;
}

function bannerInfo(){
  const pts=currentPts();
  if(S.awaitBest)return['آیا این زاویه «بهترین» است؟ Yes یا No را بزنید','ask'];
  if(S.repickIdx!==null)return[`نقطه «${NAMES[S.repickIdx]}» را دوباره انتخاب کنید`,'pick'];
  if(S.picking){const i=pts.findIndex(p=>!p);
    return[['ابتدا راس را انتخاب کنید','نقطه دوم را انتخاب کنید','نقطه سوم را انتخاب کنید'][i],'pick'];}
  if(allSet())return['نقاط آماده است — در صورت نیاز اصلاح کنید، سپس «تایید نقاط» را بزنید','ready'];
  return['برای شروع «انتخاب نقطه» را بزنید',''];
}
function updateUI(){
  if(!S.meta)return;
  $('hUser').textContent=S.userId||'-';$('hLabel').textContent=S.label||'-';
  $('sliceTitle').textContent=`Slice ${S.slice}  /  ${S.meta.n_slices-1}`;
  if(S.mode==='length'){updateLenUI();return;}
  const [t,k]=bannerInfo();const b=$('banner');b.textContent=t+(S.editingIdx!==null?'   (حالت ویرایش رکورد)':'');b.className=k;
  cv.classList.toggle('picking',S.picking||S.repickIdx!==null);
  const pts=currentPts();
  $('btnConfirm').disabled=!allSet()||S.awaitBest;
  const tb=$('ptsTable').querySelector('tbody');tb.innerHTML='';
  pts.forEach((p,k)=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td style="text-align:right"><span class="dot" style="background:${COLORS[k]}"></span>${NAMES[k]}</td>
      <td><input type="number" data-k="${k}" data-f="0" value="${p?p[0]:''}"></td>
      <td><input type="number" data-k="${k}" data-f="1" value="${p?p[1]:''}"></td>
      <td><button class="btn sm" data-act="repick" data-k="${k}">انتخاب مجدد</button>
          <button class="btn sm bad" data-act="del" data-k="${k}" ${p?'':'disabled'}>حذف</button></td>`;
    tb.appendChild(tr);
  });
  if(allSet()){const r=calcAngle(...pts);
    $('angleBox').innerHTML=r?`Angle = <b>${r.angle.toFixed(2)}°</b> &nbsp;|&nbsp; Turning = <b>${r.turning.toFixed(2)}°</b>`:'دو نقطه روی هم افتاده‌اند';
  }else $('angleBox').textContent='با انتخاب هر سه نقطه، زاویه محاسبه می‌شود';
  $('bestBox').style.display=S.awaitBest?'block':'none';
  if(S.awaitBest&&allSet()){const r=calcAngle(...pts);
    $('bestText').textContent=`زاویه ${r.angle.toFixed(2)}° — آیا این بهترین است؟`;}
}
function updateLenUI(){
  const pts=currentLen();let t,k='';const done=!S.lpicking;
  if(S.lrepickIdx!==null){t=`نقطه «${lname(S.lrepickIdx,pts.length,done)}» را دوباره انتخاب کنید`;k='pick';}
  else if(S.lpicking){
    if(pts.length===0)t='نقطه مبدا (Vessel origin) را انتخاب کنید';
    else if(S.pathMode)t=`نقطه بعدی مسیر را انتخاب کنید (${pts.length} نقطه) — برای اتمام «پایان مسیر» یا Enter`;
    else t='نقطه انتها (Distal end) را انتخاب کنید';
    k='pick';
  }
  else if(allLenSet()){t='نقاط آماده است — در صورت نیاز اصلاح کنید، سپس «ثبت طول» را بزنید';k='ready';}
  else t='برای شروع «انتخاب نقاط» را بزنید';
  const b=$('banner');b.textContent=t+(S.leditingIdx!==null?'   (حالت ویرایش رکورد)':'');b.className=k;
  cv.classList.toggle('picking',S.lpicking||S.lrepickIdx!==null);
  $('btnLSave').disabled=!allLenSet();
  $('btnLFinish').disabled=!(S.lpicking&&pts.length>=2);
  const tb=$('lptsTable').querySelector('tbody');tb.innerHTML='';
  pts.forEach((p,i)=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td style="text-align:right"><span class="dot" style="background:${LORANGE}"></span>${lname(i,pts.length,done)}</td>
      <td><input type="number" data-k="${i}" data-f="0" value="${p[0]}"></td>
      <td><input type="number" data-k="${i}" data-f="1" value="${p[1]}"></td>
      <td><button class="btn sm" data-act="repick" data-k="${i}">انتخاب مجدد</button>
          <button class="btn sm bad" data-act="del" data-k="${i}">حذف</button></td>`;
    tb.appendChild(tr);
  });
  if(allLenSet()){const r=calcLen(pts);
    const sp1=S.meta.sp_x===1&&S.meta.sp_y===1;
    $('lenBox').innerHTML=r?`Apparent length = <b>${r.px.toFixed(2)}</b> px &nbsp;|&nbsp; Real length = <b>${r.mm.toFixed(2)}</b> mm<br>Straight distance = <b>${r.chordMm.toFixed(2)}</b> mm &nbsp;|&nbsp; RL = <b>${r.rl.toFixed(3)}</b>`+(sp1?'<div style="font-size:11.5px;color:var(--warn);margin-top:6px">spacing تصویر ۱٫۰ است؛ پس طول واقعی با ظاهری برابر شده</div>':''):'نقطه ابتدا و انتها روی هم افتاده‌اند';
  }else $('lenBox').textContent='پس از انتخاب نقاط، طول و نسبت RL محاسبه می‌شود';
}
function renderRecords(){
  const tb=$('recTable').querySelector('tbody');tb.innerHTML='';
  S.records.forEach(r=>{
    const tr=document.createElement('tr');tr.dataset.idx=r.idx;tr.dataset.slice=r.Slice;
    const f=v=>(v===null||v===undefined)?'':v;
    tr.innerHTML=`<td>${f(r.User_ID)}</td><td>${f(r.Slice)}</td><td>${f(r.Label)}</td>
      <td>${f(r.Apex_row)}</td><td>${f(r.Apex_col)}</td><td>${f(r.Pt1_row)}</td><td>${f(r.Pt1_col)}</td>
      <td>${f(r.Pt2_row)}</td><td>${f(r.Pt2_col)}</td>
      <td>${Number(r.Angle_deg).toFixed(2)}</td><td>${Number(r.Turning_deg).toFixed(2)}</td>
      <td><button class="bestbtn ${r.Best?'on':''}" data-act="best" data-idx="${r.idx}">${r.Best?1:0}</button></td>
      <td><button class="btn sm" data-act="edit" data-idx="${r.idx}">ویرایش</button>
          <button class="btn sm bad" data-act="rdel" data-idx="${r.idx}">حذف</button></td>`;
    tb.appendChild(tr);
  });
  $('recWrap').scrollTop=$('recWrap').scrollHeight;
}
function renderLenRecords(){
  const tb=$('lrecTable').querySelector('tbody');tb.innerHTML='';
  S.lrecords.forEach(r=>{
    const tr=document.createElement('tr');tr.dataset.idx=r.idx;tr.dataset.slice=r.Slice;
    const f=v=>(v===null||v===undefined)?'':v;
    tr.innerHTML=`<td>${f(r.User_ID)}</td><td>${f(r.Slice)}</td><td>${f(r.Label)}</td><td>${f(r.N_pts)}</td>
      <td>${Number(r.Apparent_px).toFixed(2)}</td><td>${Number(r.Real_mm).toFixed(2)}</td>
      <td>${Number(r.Chord_mm).toFixed(2)}</td><td>${Number(r.RL).toFixed(3)}</td>
      <td><button class="btn sm" data-act="edit" data-idx="${r.idx}">ویرایش</button>
          <button class="btn sm bad" data-act="rdel" data-idx="${r.idx}">حذف</button></td>`;
    tb.appendChild(tr);
  });
  $('lrecWrap').scrollTop=$('lrecWrap').scrollHeight;
}

function setMode(m){
  S.mode=m;
  S.picking=false;S.repickIdx=null;S.awaitBest=false;S.lpicking=false;S.lrepickIdx=null;
  $('modeAngle').classList.toggle('on',m==='angle');
  $('modeLen').classList.toggle('on',m==='length');
  $('angTools').classList.toggle('hid',m==='length');
  $('lenTools').classList.toggle('hid',m!=='length');
  $('angSide').classList.toggle('hid',m==='length');
  $('lenSide').classList.toggle('hid',m!=='length');
  draw();updateUI();
}

function startPicking(){
  S.pending[S.slice]=[null,null,null];S.picking=true;S.repickIdx=null;S.awaitBest=false;draw();updateUI();
}
function placePoint(r,c){
  const pts=currentPts();let idx;
  if(S.repickIdx!==null){idx=S.repickIdx;S.repickIdx=null;}
  else if(S.picking){idx=pts.findIndex(p=>!p);if(idx<0){S.picking=false;return;}}
  else return;
  pts[idx]=[r,c];
  if(allSet())S.picking=false;
  draw();updateUI();
}
async function saveRecord(best){
  const p=currentPts();
  const body=JSON.stringify({user_id:S.userId,slice:S.slice,label:S.label,apex:p[0],pt1:p[1],pt2:p[2],best});
  try{
    const res=S.editingIdx!==null
      ?await api('/api/records/'+S.editingIdx,{method:'PUT',headers:JH,body})
      :await api('/api/records',{method:'POST',headers:JH,body});
    toast(`ثبت شد — زاویه ${res.angle}° | Best=${best}`);
    if(res.warning)toast(res.warning,'warn');
    delete S.pending[S.slice];S.editingIdx=null;S.awaitBest=false;S.picking=false;S.repickIdx=null;
    await loadRecords();updateUI();draw();
  }catch(e){toast(e.message,'err');}
}
function cancelAll(){S.picking=false;S.repickIdx=null;S.awaitBest=false;draw();updateUI();}

function startLenPicking(){
  S.lpending[S.slice]=[];S.lpicking=true;S.lrepickIdx=null;draw();updateUI();
}
function finishLen(){
  if(currentLen().length<2){toast('حداقل دو نقطه لازم است','err');return;}
  S.lpicking=false;draw();updateUI();
}
function placeLenPoint(r,c){
  const pts=currentLen();
  if(S.lrepickIdx!==null){pts[S.lrepickIdx]=[r,c];S.lrepickIdx=null;}
  else if(S.lpicking){pts.push([r,c]);if(!S.pathMode&&pts.length===2)S.lpicking=false;}
  else return;
  draw();updateUI();
}
async function saveLength(){
  const p=currentLen();
  if(!calcLen(p)){toast('نقطه ابتدا و انتها روی هم افتاده‌اند','err');return;}
  const body=JSON.stringify({user_id:S.userId,slice:S.slice,label:S.label,pts:p});
  try{
    const res=S.leditingIdx!==null
      ?await api('/api/lengths/'+S.leditingIdx,{method:'PUT',headers:JH,body})
      :await api('/api/lengths',{method:'POST',headers:JH,body});
    toast(`ثبت شد — طول ظاهری ${res.apparent} px | طول واقعی ${res.real} mm | RL ${res.rl}`);
    if(res.warning)toast(res.warning,'warn');
    delete S.lpending[S.slice];S.leditingIdx=null;S.lpicking=false;S.lrepickIdx=null;
    await loadRecords();updateUI();draw();
  }catch(e){toast(e.message,'err');}
}
function cancelLen(){S.lpicking=false;S.lrepickIdx=null;draw();updateUI();}

function mpos(e){const b=cv.getBoundingClientRect();return{x:e.clientX-b.left,y:e.clientY-b.top};}
function hitTest(x,y){
  const pts=S.mode==='length'?currentLen():currentPts();let best=null,bd=10;
  pts.forEach((p,k)=>{if(!p)return;const [sx,sy]=toScr(...p);const d=Math.hypot(sx-x,sy-y);if(d<=bd){bd=d;best=k;}});
  return best;
}
function bind(){
  $('prev').onclick=()=>gotoSlice(S.slice-1);
  $('next').onclick=()=>gotoSlice(S.slice+1);
  $('modeAngle').onclick=()=>setMode('angle');
  $('modeLen').onclick=()=>setMode('length');
  $('btnPick').onclick=startPicking;
  $('btnConfirm').onclick=()=>{if(!allSet())return;if(!calcAngle(...currentPts())){toast('دو نقطه روی هم افتاده‌اند','err');return;}S.awaitBest=true;updateUI();};
  $('bestYes').onclick=()=>saveRecord(1);
  $('bestNo').onclick=()=>saveRecord(0);
  $('bestCancel').onclick=cancelAll;
  $('btnClear').onclick=()=>{S.pending[S.slice]=[null,null,null];S.picking=false;S.repickIdx=null;S.awaitBest=false;S.editingIdx=null;draw();updateUI();};
  $('btnLPick').onclick=startLenPicking;
  $('btnLSave').onclick=saveLength;
  $('btnLClear').onclick=()=>{S.lpending[S.slice]=[];S.lpicking=false;S.lrepickIdx=null;S.leditingIdx=null;draw();updateUI();};
  $('btnLFinish').onclick=finishLen;
  $('chkPath').onchange=e=>{S.pathMode=e.target.checked;updateUI();};
  $('btnFit').onclick=()=>{fit();draw();};
  $('btnDownload').onclick=()=>{window.location='/api/download';};
  $('btnLDownload').onclick=()=>{window.location='/api/download';};
  $('btnSession').onclick=()=>openModal();
  $('chkContour').onchange=e=>{S.showContour=e.target.checked;draw();};
  $('chkSaved').onchange=e=>{S.showSaved=e.target.checked;draw();};
  const reC=async()=>{S.sigma=parseFloat($('sigma').value)||0;S.ratio=parseFloat($('ratio').value)||0.4;
    try{S.curContours=await loadContours(S.slice);draw();}catch(e){toast(e.message,'err');}};
  $('sigma').onchange=reC;$('ratio').onchange=reC;

  $('ptsTable').addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;const k=+b.dataset.k;
    if(b.dataset.act==='repick'){S.repickIdx=k;S.awaitBest=false;draw();updateUI();}
    if(b.dataset.act==='del'){currentPts()[k]=null;S.picking=true;S.awaitBest=false;draw();updateUI();}
  });
  $('ptsTable').addEventListener('change',e=>{
    const i=e.target;if(i.tagName!=='INPUT')return;const k=+i.dataset.k,f=+i.dataset.f;
    const pts=currentPts();const cur=pts[k]||[null,null];
    const v=parseInt(i.value);if(isNaN(v))return;cur[f]=v;
    if(cur[0]===null||cur[1]===null){pts[k]=[cur[0]??0,cur[1]??0];} else pts[k]=cur;
    const p=pts[k];p[0]=Math.max(0,Math.min(S.meta.height-1,p[0]));p[1]=Math.max(0,Math.min(S.meta.width-1,p[1]));
    if(allSet())S.picking=false;
    draw();updateUI();
  });
  $('lptsTable').addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;const k=+b.dataset.k;
    if(b.dataset.act==='repick'){S.lrepickIdx=k;draw();updateUI();}
    if(b.dataset.act==='del'){const pts=currentLen();pts.splice(k,1);if(pts.length<2)S.lpicking=true;draw();updateUI();}
  });
  $('lptsTable').addEventListener('change',e=>{
    const i=e.target;if(i.tagName!=='INPUT')return;const k=+i.dataset.k,f=+i.dataset.f;
    const p=currentLen()[k];if(!p)return;
    const v=parseInt(i.value);if(isNaN(v))return;p[f]=v;
    p[0]=Math.max(0,Math.min(S.meta.height-1,p[0]));p[1]=Math.max(0,Math.min(S.meta.width-1,p[1]));
    draw();updateUI();
  });
  $('recTable').addEventListener('click',async e=>{
    const b=e.target.closest('button');
    if(!b){const tr=e.target.closest('tr');if(tr&&tr.dataset.slice!==undefined)gotoSlice(+tr.dataset.slice);return;}
    const idx=+b.dataset.idx;
    try{
      if(b.dataset.act==='best'){
        const rec=S.records.find(r=>r.idx===idx);
        const res=await api(`/api/records/${idx}/best`,{method:'PATCH',headers:JH,body:JSON.stringify({best:rec.Best?0:1})});
        if(res.warning)toast(res.warning,'warn');await loadRecords();
      }else if(b.dataset.act==='rdel'){
        if(!confirm('این ردیف حذف شود؟'))return;
        const res=await api('/api/records/'+idx,{method:'DELETE'});
        if(res.warning)toast(res.warning,'warn');await loadRecords();
      }else if(b.dataset.act==='edit'){
        const r=S.records.find(x=>x.idx===idx);
        S.pending[r.Slice]=[[r.Apex_row,r.Apex_col],[r.Pt1_row,r.Pt1_col],[r.Pt2_row,r.Pt2_col]];
        S.editingIdx=idx;S.picking=false;S.repickIdx=null;
        await gotoSlice(r.Slice);toast('نقاط بارگذاری شد؛ اصلاح کنید و دوباره «تایید نقاط» را بزنید');
      }
    }catch(err){toast(err.message,'err');}
  });
  $('lrecTable').addEventListener('click',async e=>{
    const b=e.target.closest('button');
    if(!b){const tr=e.target.closest('tr');if(tr&&tr.dataset.slice!==undefined)gotoSlice(+tr.dataset.slice);return;}
    const idx=+b.dataset.idx;
    try{
      if(b.dataset.act==='rdel'){
        if(!confirm('این ردیف حذف شود؟'))return;
        const res=await api('/api/lengths/'+idx,{method:'DELETE'});
        if(res.warning)toast(res.warning,'warn');await loadRecords();
      }else if(b.dataset.act==='edit'){
        const r=S.lrecords.find(x=>x.idx===idx);
        S.lpending[r.Slice]=parsePts(r.Pts);
        S.leditingIdx=idx;S.lpicking=false;S.lrepickIdx=null;
        await gotoSlice(r.Slice);toast('نقاط بارگذاری شد؛ اصلاح کنید و دوباره «ثبت طول» را بزنید');
      }
    }catch(err){toast(err.message,'err');}
  });

  cv.addEventListener('mousedown',e=>{
    const {x,y}=mpos(e);
    S.mouse={down:true,x0:x,y0:y,moved:false,hit:hitTest(x,y),tx0:S.view.tx,ty0:S.view.ty,cur:S.mouse&&S.mouse.cur};
  });
  cv.addEventListener('mousemove',e=>{
    const {x,y}=mpos(e);const [r,c]=toImg(x,y);
    const m=S.mouse||(S.mouse={});m.cur=[Math.round(r),Math.round(c)];
    if(m.down){
      if(Math.hypot(x-m.x0,y-m.y0)>3)m.moved=true;
      if(m.moved){
        if(m.hit!==null){
          const rr=Math.max(0,Math.min(S.meta.height-1,Math.round(r))),cc=Math.max(0,Math.min(S.meta.width-1,Math.round(c)));
          if(S.mode==='length'){currentLen()[m.hit]=[rr,cc];}
          else{currentPts()[m.hit]=[rr,cc];S.awaitBest=false;}
          updateUI();
        }else{S.view.tx=m.tx0+(x-m.x0);S.view.ty=m.ty0+(y-m.y0);}
      }
    }
    draw();
  });
  window.addEventListener('mouseup',e=>{
    const m=S.mouse;if(!m||!m.down)return;m.down=false;
    if(!m.moved&&isPicking()){
      const b=cv.getBoundingClientRect();const x=e.clientX-b.left,y=e.clientY-b.top;
      if(x>=0&&y>=0&&x<=b.width&&y<=b.height){
        const [r,c]=toImg(x,y);const rr=Math.round(r),cc=Math.round(c);
        if(inBounds(rr,cc)){if(S.mode==='length')placeLenPoint(rr,cc);else placePoint(rr,cc);}
      }
    }
  });
  cv.addEventListener('wheel',e=>{
    e.preventDefault();const {x,y}=mpos(e);const f=e.deltaY<0?1.2:1/1.2;
    const v=S.view;const ns=Math.max(0.2,Math.min(80,v.scale*f));const k=ns/v.scale;
    v.tx=x-(x-v.tx)*k;v.ty=y-(y-v.ty)*k;v.scale=ns;draw();
  },{passive:false});
  window.addEventListener('keydown',e=>{
    if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;
    if($('modalBg').style.display!=='none')return;
    if(e.key==='ArrowLeft')gotoSlice(S.slice-1);
    else if(e.key==='ArrowRight')gotoSlice(S.slice+1);
    else if(e.key==='Enter'&&S.mode==='length'&&S.lpicking&&currentLen().length>=2)finishLen();
    else if(e.key==='Escape'){if(S.mode==='length')cancelLen();else cancelAll();}
  });
  window.addEventListener('resize',()=>{resizeCanvas();draw();});
}

function openModal(){
  $('inUser').value=S.userId||localStorage.getItem('uid')||'';$('inLabel').value=S.label||'';
  $('modalBg').style.display='flex';setTimeout(()=>($('inUser').value?$('inLabel'):$('inUser')).focus(),50);
}
$('btnStart').onclick=async()=>{
  const u=$('inUser').value.trim(),l=$('inLabel').value.trim();
  if(!u||!l){toast('User ID و Label را پر کنید','err');return;}
  const changedUser=u!==S.userId;
  S.userId=u;S.label=l;localStorage.setItem('uid',u);
  $('modalBg').style.display='none';
  if(changedUser){S.pending={};S.editingIdx=null;S.lpending={};S.leditingIdx=null;}
  await loadRecords();updateUI();draw();
};
['inUser','inLabel'].forEach(id=>$(id).addEventListener('keydown',e=>{if(e.key==='Enter')$('btnStart').click();}));

(async function init(){
  try{
    S.meta=await api('/api/meta');
    S.sigma=S.meta.default_sigma;S.ratio=S.meta.default_ratio;
    $('sigma').value=S.sigma;$('ratio').value=S.ratio;
    $('excelPath').textContent='فایل Excel: '+S.meta.excel_path;
    resizeCanvas();fit();bind();openModal();
    await gotoSlice(0);
  }catch(e){toast('خطا در شروع: '+e.message,'err');}
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\nداشبورد: http://{HOST}:{PORT}   (برای توقف: Ctrl+C)\n")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)