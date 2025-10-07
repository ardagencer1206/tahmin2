# app.py
# Talep Tahminleme + Kapasite Planlama (SARIMA + Lineer Regresyon)
# Girdi: talep.xlsx (zorunlu, name=file) + kapasite.xlsx (opsiyonel, name=capfile)
# UI: templates/index.html  (JS tarafı /forecast dönen charts[]'ı <img src="..."> olarak gösterebilir)
import json
import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
import os
import io
import warnings
import base64
import numpy as np
import pandas as pd


from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    render_template,
    abort,
)

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
app = Flask(__name__)

_LAST_CSV_BYTES = None  # son çıktı cache

# ---------- Yardımcı ----------
def _img_to_dataurl(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

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
    """Esnek kolon adlarıyla kapasite xlsx oku. Beklenen: depo|warehouse|site, urun?, kapasite|capacity|..."""
    if not cap_storage:
        return pd.DataFrame()
    try:
        cap = pd.read_excel(cap_storage)
    except Exception as e:
        abort(400, f"Kapasite XLSX okunamadı: {e}")
    cap.columns = [str(c).strip().lower() for c in cap.columns]
    # kolon eşle
    cap_depo = next((c for c in ["depo", "warehouse", "site"] if c in cap.columns), None)
    cap_kap = next((c for c in ["kapasite", "capacity", "kapasite_birim", "kapasite_toplam"] if c in cap.columns), None)
    cap_urun = "urun" if "urun" in cap.columns else None
    if not cap_depo or not cap_kap:
        abort(400, "Kapasite dosyasında gerekli kolonlar yok (depo ve kapasite).")
    cols = [cap_depo, cap_kap] + ([cap_urun] if cap_urun else [])
    cap = cap[cols].copy()
    return cap.rename(columns={cap_depo: "depo", cap_kap: "kapasite", (cap_urun or "urun"): "urun"})

def infer_freq_or_use(df, date_col, freq):
    if freq:
        return freq.strip()
    try:
        return pd.infer_freq(df.sort_values(date_col)[date_col])
    except Exception:
        return None

def ensure_continuous_index(s, freq_hint):
    s = s.sort_index()
    if freq_hint is None:
        diffs = s.index.to_series().diff().dropna()
        if len(diffs) == 0:
            freq_hint = "D"
        else:
            median_days = diffs.median() / np.timedelta64(1, "D")
            if 27 <= median_days <= 31:
                freq_hint = "M"
            elif 6 <= median_days <= 8:
                freq_hint = "W"
            else:
                freq_hint = "D"
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq=freq_hint)
    s = s.reindex(full_idx).interpolate(limit_direction="both")
    return s, freq_hint

# ---------- Modeller ----------
def fit_predict_sarima(y, h, s):
    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, int(s)),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=h).predicted_mean
    fc.index = pd.date_range(
        y.index[-1] + (y.index[-1] - y.index[-2] if len(y) > 1 else pd.Timedelta(days=1)),
        periods=h,
        freq=y.index.freq or y.index.inferred_freq,
    )
    return fc

def build_lr_features(y, s):
    n = len(y)
    t = np.arange(n).reshape(-1, 1)
    s = int(s) if s and int(s) > 0 else 12
    sin1 = np.sin(2 * np.pi * np.arange(n) / s).reshape(-1, 1)
    cos1 = np.cos(2 * np.pi * np.arange(n) / s).reshape(-1, 1)
    lag1 = np.roll(y.values, 1).reshape(-1, 1)
    lag1[0] = y.values[0]
    return np.hstack([t, sin1, cos1, lag1])

def fit_predict_lr(y, h, s):
    y_vals = y.values.astype(float)
    X = build_lr_features(y, s)
    model = LinearRegression()
    model.fit(X, y_vals)

    preds = []
    last_val = y_vals[-1]
    n = len(y_vals)
    s = int(s) if s and int(s) > 0 else 12
    for k in range(1, h + 1):
        t_next = n + k - 1
        x_next = np.array([[t_next, np.sin(2 * np.pi * t_next / s), np.cos(2 * np.pi * t_next / s), last_val]])
        y_hat = model.predict(x_next)[0]
        preds.append(y_hat)
        last_val = y_hat

    next_index = pd.date_range(
        y.index[-1] + (y.index[-1] - y.index[-2] if len(y) > 1 else pd.Timedelta(days=1)),
        periods=h,
        freq=y.index.freq or y.index.inferred_freq,
    )
    return pd.Series(preds, index=next_index)

# ---------- Kapasite birleşimi ----------
def merge_capacity(forecasts_df, capacity_df, depo_col, urun_col):
    if capacity_df is None or capacity_df.empty:
        forecasts_df["kapasite"] = np.nan
        forecasts_df["kullanim_orani"] = np.nan
        return forecasts_df
    cap = capacity_df.copy()
    # eşleşme anahtarları
    keys = []
    if depo_col and depo_col in forecasts_df.columns:
        keys.append((depo_col, "depo"))
    if urun_col and urun_col in forecasts_df.columns and "urun" in cap.columns:
        keys.append((urun_col, "urun"))
    if not keys:
        forecasts_df["kapasite"] = np.nan
        forecasts_df["kullanim_orani"] = np.nan
        return forecasts_df
    left_on = [k[0] for k in keys]
    right_on = [k[1] for k in keys]
    merged = forecasts_df.merge(cap.drop_duplicates(), how="left", left_on=left_on, right_on=right_on)
    merged = merged.drop(columns=[c for c in right_on if c in merged.columns], errors="ignore")
    merged["kullanim_orani"] = merged["tahmin_mean"] / merged["kapasite"]
    return merged

# ---------- İş akışı ----------
def forecast_group(df_group, h, s, date_col, demand_col, freq_hint):
    ts = df_group.set_index(date_col)[demand_col].sort_index()
    ts, _ = ensure_continuous_index(ts, freq_hint)
    sar = fit_predict_sarima(ts, h, s)
    lr = fit_predict_lr(ts, h, s)
    out = pd.DataFrame({"tarih": sar.index, "sarima": sar.values, "linreg": lr.values})
    out["tahmin_mean"] = out[["sarima", "linreg"]].mean(axis=1)
    return out

def run_pipeline(xlsx_file, capfile, h, s, col_date, col_demand, col_depo, col_urun, freq):
    df = read_excel_file(xlsx_file, col_date, col_demand)
    freq_hint = infer_freq_or_use(df, col_date, freq)

    group_cols = [c for c in [col_depo, col_urun] if c and c in df.columns]
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        df["_grp"] = "serie"
        group_cols = ["_grp"]
        grouped = df.groupby(group_cols)

    # kapasite dosyası
    cap_df = read_capacity_excel(capfile) if capfile else pd.DataFrame()

    outs = []
    for keys, g in grouped:
        fc = forecast_group(g, h, s, col_date, col_demand, freq_hint)
        if isinstance(keys, tuple):
            for name, val in zip(group_cols, keys):
                fc[name] = val
        else:
            fc[group_cols[0]] = keys
        outs.append(fc)

    res = pd.concat(outs, ignore_index=True)

    depo_col = col_depo if col_depo in res.columns else None
    urun_col = col_urun if col_urun in res.columns else None
    res2 = merge_capacity(res, cap_df, depo_col, urun_col)

    # sütun sırası
    lead = []
    if "tarih" in res2.columns: lead.append("tarih")
    if depo_col: lead.append(depo_col)
    if urun_col: lead.append(urun_col)
    ordered = lead + [c for c in ["sarima", "linreg", "tahmin_mean", "kapasite", "kullanim_orani"] if c in res2.columns]
    other = [c for c in res2.columns if c not in ordered]
    res2 = res2[ordered + other]
    return res2, cap_df

# ---------- Grafikler ----------
def build_charts(result_df, cap_df, depo_col, urun_col):
    charts = []

    # 1) Toplam tahmin eğrisi (tahmin_mean sum)
    if "tarih" in result_df.columns and "tahmin_mean" in result_df.columns:
        agg = result_df.groupby("tarih", as_index=True)["tahmin_mean"].sum()
        fig = plt.figure()
        ax = fig.gca()
        agg.plot(ax=ax)
        ax.set_title("Toplam Tahmin")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("Miktar")
        charts.append({"title": "Toplam Tahmin", "image": _img_to_dataurl(fig)})

    # 2) Son dönem kullanım oranı (varsa kapasite)
    if "kapasite" in result_df.columns and result_df["kapasite"].notna().any():
        last_date = result_df["tarih"].max()
        last_df = result_df[result_df["tarih"] == last_date].copy()

        key_cols = [c for c in [depo_col, urun_col] if c and c in last_df.columns]
        if not key_cols:
            key_cols = []  # tek seri
            last_df["anahtar"] = "seri"
            key_cols = ["anahtar"]

        last_df["oran"] = last_df["tahmin_mean"] / last_df["kapasite"]
        top = last_df.sort_values("oran", ascending=False).head(10)

        labels = top[key_cols].astype(str).agg(" / ".join, axis=1) if len(key_cols) > 1 else top[key_cols[0]]
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.bar(labels, top["oran"])
        ax2.set_title(f"Son Dönem Kullanım Oranı ({str(last_date)})")
        ax2.set_ylabel("Oran")
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        charts.append({"title": "Kullanım Oranı (Son Dönem)", "image": _img_to_dataurl(fig2)})

    return charts

# ---------- Rotalar ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/forecast", methods=["POST"])
def forecast_endpoint():
    global _LAST_CSV_BYTES

    if "file" not in request.files:
        abort(400, "Talep dosyası yüklenmedi.")
    file = request.files["file"]
    capfile = request.files.get("capfile")  # opsiyonel

    if not file or file.filename == "":
        abort(400, "Geçersiz talep dosyası.")

    h = int(request.form.get("h", "12"))
    s = int(request.form.get("s", "12"))
    col_date = request.form.get("col_date", "tarih").strip()
    col_demand = request.form.get("col_demand", "talep").strip()
    col_depo = (request.form.get("col_depo", "depo") or "").strip() or None
    col_urun = (request.form.get("col_urun", "urun") or "").strip() or None
    freq = (request.form.get("freq", "") or "").strip() or None

    res_df, cap_df = run_pipeline(file, capfile, h, s, col_date, col_demand, col_depo, col_urun, freq)

    # CSV cache
    csv_buf = io.StringIO()
    res_df.to_csv(csv_buf, index=False)
    _LAST_CSV_BYTES = csv_buf.getvalue().encode("utf-8")

    # Charts
    charts = build_charts(res_df.copy(), cap_df, col_depo, col_urun)

    # JSON-safe
    out_df = res_df.copy()
    if "tarih" in out_df.columns:
        out_df["tarih"] = out_df["tarih"].astype(str)
    out_df = out_df.replace([np.inf, -np.inf], np.nan)
    data = pd.read_json(out_df.to_json(orient="records"))

    resp = {
        "h": h,
        "s": s,
        "n_kayit": int(len(out_df)),
        "columns": list(out_df.columns),
        "data": data.to_dict(orient="records"),
        "capacity_source": "xlsx" if (capfile and getattr(capfile, "filename", "")) else "none",
        "charts": charts,  # [{title, image(data URL)}]
    }
    return jsonify(resp)

@app.route("/download_last.csv", methods=["GET"])
def download_last():
    global _LAST_CSV_BYTES
    if not _LAST_CSV_BYTES:
        abort(404, "Önce /forecast çağrısı ile bir çıktı üretin.")
    return send_file(
        io.BytesIO(_LAST_CSV_BYTES),
        mimetype="text/csv",
        as_attachment=True,
        download_name="forecast_capacity.csv",
    )

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
