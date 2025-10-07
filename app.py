# app.py
# Talep Tahminleme + Kapasite Planlama
# Modeller: Lineer Regresyon (LR) + Ağırlıklı Hareketli Ortalama (WMA)
# Girdi: talep.xlsx (name=file, zorunlu) + kapasite.xlsx (name=capfile, opsiyonel)
# UI: templates/index.html (Chart.js frontend)

import os
import io
import json
import warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template, abort
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB dosya limiti

_LAST_CSV_BYTES = None  # son CSV cache

# ----------------- IO -----------------
def read_excel_file(file_storage, col_date, col_demand):
    try:
        df = pd.read_excel(file_storage)
    except Exception as e:
        abort(400, f"XLSX okunamadı: {e}")

    df.columns = df.columns.str.strip().str.lower()
    col_date = col_date.lower()
    col_demand = col_demand.lower()

    for c in [col_date, col_demand]:
        if c not in df.columns:
            abort(400, f"Gerekli sütun yok: {c}")

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df[col_demand] = pd.to_numeric(df[col_demand], errors="coerce")
    df = df.dropna(subset=[col_date, col_demand])
    return df


def read_capacity_excel(cap_storage):
    if not cap_storage:
        return pd.DataFrame()
    try:
        cap = pd.read_excel(cap_storage)
    except Exception as e:
        abort(400, f"Kapasite XLSX okunamadı: {e}")

    cap.columns = cap.columns.str.strip().str.lower()
    cap_depo = next((c for c in ["depo", "warehouse", "site"] if c in cap.columns), None)
    cap_kap = next((c for c in ["kapasite", "capacity"] if c in cap.columns), None)
    cap_urun = "urun" if "urun" in cap.columns else None
    if not cap_depo or not cap_kap:
        abort(400, "Kapasite dosyasında gerekli kolonlar yok (depo ve kapasite).")

    cols = [cap_depo, cap_kap] + ([cap_urun] if cap_urun else [])
    cap = cap[cols].copy()
    rename = {cap_depo: "depo", cap_kap: "kapasite"}
    if cap_urun: rename[cap_urun] = "urun"
    return cap.rename(columns=rename)

# ----------------- Zaman Serisi Yardımcıları -----------------
def infer_freq(dates: pd.Series, fallback="M"):
    try:
        f = pd.infer_freq(dates.sort_values())
        return f or fallback
    except Exception:
        return fallback

def next_index(last_date: pd.Timestamp, periods: int, base_dates: pd.Series):
    freq = infer_freq(base_dates)
    step = pd.tseries.frequencies.to_offset(freq)
    start = last_date + step
    return pd.date_range(start, periods=periods, freq=freq)

# ----------------- Modeller -----------------
def fit_predict_lr(y: np.ndarray, h: int):
    n = len(y)
    X = np.arange(n).reshape(-1, 1)
    m = LinearRegression().fit(X, y.astype(float))
    # Çok adımlı ileri tahmin
    X_fut = np.arange(n, n + h).reshape(-1, 1)
    return m.predict(X_fut).tolist()

def predict_wma(y: np.ndarray, h: int, window: int = 3, weights: np.ndarray | None = None):
    if window < 1:
        window = 3
    y_list = list(y.astype(float))
    if weights is None:
        # lineer ağırlıklar: 1..window
        weights = np.arange(1, window + 1, dtype=float)
    else:
        weights = np.array(weights, dtype=float)
        if len(weights) != window:
            window = len(weights)
    w = weights / weights.sum()

    preds = []
    for _ in range(h):
        src = y_list[-window:] if len(y_list) >= window else ( [y_list[-1]] * (window - len(y_list)) + y_list )
        val = float(np.dot(src[-window:], w))
        preds.append(val)
        y_list.append(val)
    return preds

# ----------------- Pipeline -----------------
def run_pipeline(xlsx_file, capfile, h, col_date, col_demand, col_depo, col_urun, wma_window):
    df = read_excel_file(xlsx_file, col_date, col_demand)
    # grup kolonlarını normalize et
    group_cols = [c for c in [col_depo, col_urun] if c and c in df.columns]
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [(("serie",), df.assign(_grp="serie"))]

    cap_df = read_capacity_excel(capfile) if capfile else pd.DataFrame()

    outs = []
    for keys, g in grouped if isinstance(grouped, list) else grouped:
        # tarih sırası
        g = g.sort_values(col_date)
        y = g[col_demand].to_numpy(dtype=float)
        if len(y) < 2:
            continue

        # Tahminler
        lr = fit_predict_lr(y, h)
        wma = predict_wma(y, h, window=wma_window)
        # tarihleri oluştur
        fut_idx = next_index(g[col_date].iloc[-1], h, g[col_date])

        out = pd.DataFrame({
            "tarih": fut_idx,
            "linreg": lr,
            "wma": wma,
        })
        out["tahmin_mean"] = out[["linreg", "wma"]].mean(axis=1)

        # grup anahtarlarını yaz
        if group_cols:
            if isinstance(keys, tuple):
                for name, val in zip(group_cols, keys):
                    out[name] = val
            else:
                out[group_cols[0]] = keys
        else:
            out["_grp"] = "serie"

        outs.append(out)

    if not outs:
        abort(400, "Yeterli veri yok.")

    res = pd.concat(outs, ignore_index=True)

    # kapasite merge
    if not cap_df.empty:
        keys = []
        if col_depo and col_depo in res.columns: keys.append((col_depo, "depo"))
        if col_urun and col_urun in res.columns and "urun" in cap_df.columns: keys.append((col_urun, "urun"))
        if keys:
            left_on = [k[0] for k in keys]; right_on = [k[1] for k in keys]
            res = res.merge(cap_df, how="left", left_on=left_on, right_on=right_on)
            res["kullanim_orani"] = res["tahmin_mean"] / res["kapasite"]

    # sütun sırası
    lead = [c for c in ["tarih", col_depo, col_urun] if c and c in res.columns] or ["tarih"]
    ordered = lead + [c for c in ["linreg", "wma", "tahmin_mean", "kapasite", "kullanim_orani"] if c in res.columns]
    other = [c for c in res.columns if c not in ordered]
    res = res[ordered + other]

    return res, cap_df

# ----------------- Routes -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/forecast", methods=["POST"])
def forecast_endpoint():
    global _LAST_CSV_BYTES

    if "file" not in request.files:
        abort(400, "Talep dosyası yüklenmedi.")
    file = request.files["file"]
    capfile = request.files.get("capfile")  # opsiyonel

    h = int(request.form.get("h", "12"))
    col_date = (request.form.get("col_date", "tarih") or "tarih").strip().lower()
    col_demand = (request.form.get("col_demand", "talep") or "talep").strip().lower()
    col_depo = (request.form.get("col_depo", "depo") or "").strip().lower() or None
    col_urun = (request.form.get("col_urun", "urun") or "").strip().lower() or None
    wma_window = int(request.form.get("wma_window", "3"))

    res_df, cap_df = run_pipeline(file, capfile, h, col_date, col_demand, col_depo, col_urun, wma_window)

    # CSV cache
    csv_buf = io.StringIO()
    res_df.to_csv(csv_buf, index=False)
    _LAST_CSV_BYTES = csv_buf.getvalue().encode("utf-8")

    # JSON-safe
    out_df = res_df.copy()
    if "tarih" in out_df.columns:
        out_df["tarih"] = out_df["tarih"].astype(str)
    out_df = out_df.replace([np.inf, -np.inf], np.nan)
    data = json.loads(out_df.to_json(orient="records"))

    resp = {
        "h": h,
        "wma_window": wma_window,
        "n_kayit": len(out_df),
        "columns": list(out_df.columns),
        "data": data,
        "capacity_source": "xlsx" if (capfile and getattr(capfile, "filename", "")) else "none",
    }
    return jsonify(resp)

@app.route("/download_last.csv")
def download_last():
    global _LAST_CSV_BYTES
    if not _LAST_CSV_BYTES:
        abort(404, "Önce /forecast çalıştırılmalı.")
    return send_file(
        io.BytesIO(_LAST_CSV_BYTES),
        mimetype="text/csv",
        as_attachment=True,
        download_name="forecast_capacity.csv",
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
