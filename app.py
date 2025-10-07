# app.py
# Talep Tahminleme + Kapasite Planlama (Lineer Regresyon + Chart.js için JSON)
# Girdi: talep.xlsx (name=file, zorunlu) + kapasite.xlsx (name=capfile, opsiyonel)

import os
import io
import json
import warnings

import numpy as np
import pandas as pd

from flask import (
    Flask, request, jsonify, send_file, render_template, abort
)
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
app = Flask(__name__)

_LAST_CSV_BYTES = None  # son CSV cache

# ---------- IO ----------
def read_excel_file(file_storage, col_date, col_demand):
    try:
        df = pd.read_excel(file_storage)
    except Exception as e:
        abort(400, f"XLSX okunamadı: {e}")

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

    cap.columns = [str(c).strip().lower() for c in cap.columns]
    cap_depo = next((c for c in ["depo", "warehouse", "site"] if c in cap.columns), None)
    cap_kap = next((c for c in ["kapasite", "capacity"] if c in cap.columns), None)
    cap_urun = "urun" if "urun" in cap.columns else None
    if not cap_depo or not cap_kap:
        abort(400, "Kapasite dosyasında gerekli kolonlar yok (depo ve kapasite).")

    cols = [cap_depo, cap_kap] + ([cap_urun] if cap_urun else [])
    cap = cap[cols].copy()
    return cap.rename(columns={cap_depo: "depo", cap_kap: "kapasite", (cap_urun or "urun"): "urun"})


# ---------- Basit Model ----------
def fit_predict_lr(y, h, s):
    n = len(y)
    X = np.arange(n).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    preds = []
    for k in range(1, h + 1):
        x_next = np.array([[n + k - 1]])
        y_hat = model.predict(x_next)[0]
        preds.append(y_hat)

    return preds


# ---------- Pipeline ----------
def run_pipeline(xlsx_file, capfile, h, col_date, col_demand, col_depo, col_urun):
    df = read_excel_file(xlsx_file, col_date, col_demand)

    group_cols = [c for c in [col_depo, col_urun] if c and c in df.columns]
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        df["_grp"] = "serie"
        group_cols = ["_grp"]
        grouped = df.groupby(group_cols)

    cap_df = read_capacity_excel(capfile) if capfile else pd.DataFrame()

    outs = []
    for keys, g in grouped:
        g = g.sort_values(col_date)
        ts = g[col_demand].values.astype(float)
        preds = fit_predict_lr(ts, h, s=12)

        dates = pd.date_range(g[col_date].iloc[-1] + pd.Timedelta(days=1), periods=h, freq="M")
        out = pd.DataFrame({"tarih": dates, "linreg": preds})

        if isinstance(keys, tuple):
            for name, val in zip(group_cols, keys):
                out[name] = val
        else:
            out[group_cols[0]] = keys

        outs.append(out)

    res = pd.concat(outs, ignore_index=True)

    # kapasite merge
    if not cap_df.empty:
        keys = []
        if col_depo and col_depo in res.columns:
            keys.append((col_depo, "depo"))
        if col_urun and col_urun in res.columns and "urun" in cap_df.columns:
            keys.append((col_urun, "urun"))

        if keys:
            left_on = [k[0] for k in keys]
            right_on = [k[1] for k in keys]
            res = res.merge(cap_df, how="left", left_on=left_on, right_on=right_on)
            res["kullanim_orani"] = res["linreg"] / res["kapasite"]

    return res, cap_df


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/forecast", methods=["POST"])
def forecast_endpoint():
    global _LAST_CSV_BYTES
    if "file" not in request.files:
        abort(400, "Talep dosyası yüklenmedi.")
    file = request.files["file"]
    capfile = request.files.get("capfile")

    h = int(request.form.get("h", "12"))
    col_date = request.form.get("col_date", "tarih").strip()
    col_demand = request.form.get("col_demand", "talep").strip()
    col_depo = (request.form.get("col_depo", "depo") or "").strip() or None
    col_urun = (request.form.get("col_urun", "urun") or "").strip() or None

    res_df, cap_df = run_pipeline(file, capfile, h, col_date, col_demand, col_depo, col_urun)

    # CSV cache
    csv_buf = io.StringIO()
    res_df.to_csv(csv_buf, index=False)
    _LAST_CSV_BYTES = csv_buf.getvalue().encode("utf-8")

    out_df = res_df.copy()
    out_df["tarih"] = out_df["tarih"].astype(str)
    out_df = out_df.replace([np.inf, -np.inf], np.nan)
    data = json.loads(out_df.to_json(orient="records"))

    resp = {
        "h": h,
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
        abort(404, "Önce /forecast çağrısı çalıştırılmalı.")
    return send_file(
        io.BytesIO(_LAST_CSV_BYTES),
        mimetype="text/csv",
        as_attachment=True,
        download_name="forecast_capacity.csv",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
