# app.py
# Flask tabanlı Talep Tahminleme + Kapasite Planlama tek dosya uygulaması
# - Girdi: .xlsx (geçmiş talep verisi)
# - DB: DATABASE_URL ortam değişkeni ile bağlanır, ./database.sql içeriğini çalıştırıp depo parametrelerini çeker
# - Modeller: SARIMA (statsmodels) ve Lineer Regresyon (scikit-learn)
# - Çıktı: Basit HTML arayüz + JSON ve CSV indirme
# Not: Tüm Python fonksiyonları bu dosyada tutulmuştur.

import os
import io
import warnings

import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    render_template,
    abort
)

# ML / Zaman serisi
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

# DB bağlantısı
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Son çalıştırmanın CSV çıktısını hafızada tutmak için
_LAST_CSV_BYTES = None

# ----------------------------
# Veri hazırlama ve kontroller
# ----------------------------
def read_excel_file(file_storage, col_date, col_demand, freq=None):
    """Excel dosyasını okur, tarih parse eder, gerekli sütunları doğrular."""
    try:
        df = pd.read_excel(file_storage)
    except Exception as e:
        abort(400, f"XLSX okunamadı: {e}")

    cols_needed = [col_date, col_demand]
    for c in cols_needed:
        if c not in df.columns:
            abort(400, f"Gerekli sütun yok: {c}")

    # Tarih parse
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.dropna(subset=[col_date, col_demand])
    # Talep sayısal
    df[col_demand] = pd.to_numeric(df[col_demand], errors="coerce")
    df = df.dropna(subset=[col_demand])

    if freq:
        df = df.sort_values(col_date)
    return df

def infer_freq_or_use(df, date_col, freq):
    """Frekans verilmişse onu döndürür, değilse pandas ile tahmin etmeye çalışır."""
    if freq and isinstance(freq, str) and freq.strip():
        return freq.strip()
    try:
        return pd.infer_freq(df.sort_values(date_col)[date_col])
    except Exception:
        return None  # ileri adımda fallback

def ensure_continuous_index(s, freq_hint):
    """Seriyi belirtilen frekansa göre eksiksiz zaman eksenine yerleştirir."""
    s = s.copy().sort_index()
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
    s = s.reindex(full_idx)
    s = s.interpolate(limit_direction="both")
    return s, freq_hint

# --------------------------------
# Model: SARIMA
# --------------------------------
def fit_predict_sarima(y, h, s):
    """Basit SARIMA(1,1,1)x(1,1,1,s) ile tahmin üretir."""
    try:
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, int(s)),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=h).predicted_mean
        fc.index = pd.date_range(
            y.index[-1] + (y.index[-1] - y.index[-2] if len(y) > 1 else pd.Timedelta(days=1)),
            periods=h,
            freq=y.index.freq or y.index.inferred_freq
        )
        return fc
    except Exception as e:
        raise RuntimeError(f"SARIMA başarısız: {e}")

# --------------------------------
# Model: Lineer Regresyon (özellik tabanlı)
# --------------------------------
def build_lr_features(y, s):
    """LR için öznitelikler: trend, sin-cos mevsimsellik, lag1."""
    n = len(y)
    t = np.arange(n).reshape(-1, 1)
    s = int(s) if s and int(s) > 0 else 12
    sin1 = np.sin(2 * np.pi * np.arange(n) / s).reshape(-1, 1)
    cos1 = np.cos(2 * np.pi * np.arange(n) / s).reshape(-1, 1)
    lag1 = np.roll(y.values, 1).reshape(-1, 1)
    lag1[0] = y.values[0]
    X = np.hstack([t, sin1, cos1, lag1])
    return X

def fit_predict_lr(y, h, s):
    """Lineer Regresyon ile ileri tahmin. Lag1 için iteratif yaklaşım."""
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
        sin1 = np.sin(2 * np.pi * t_next / s)
        cos1 = np.cos(2 * np.pi * t_next / s)
        x_next = np.array([[t_next, sin1, cos1, last_val]])
        y_hat = model.predict(x_next)[0]
        preds.append(y_hat)
        last_val = y_hat

    next_index = pd.date_range(
        y.index[-1] + (y.index[-1] - y.index[-2] if len(y) > 1 else pd.Timedelta(days=1)),
        periods=h,
        freq=y.index.freq or y.index.inferred_freq
    )
    return pd.Series(preds, index=next_index)

# --------------------------------
# Kapasite verisini DB'den çekme
# --------------------------------
def load_capacity_from_db():
    """
    DATABASE_URL varsa bağlanır.
    ./database.sql içindeki son SELECT çıktısını DataFrame olarak döndürür.
    """
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return pd.DataFrame()

    sql_path = os.path.join(os.getcwd(), "database.sql")
    if not os.path.exists(sql_path):
        return pd.DataFrame()

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_text = f.read().strip()
    if not sql_text:
        return pd.DataFrame()

    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        try:
            df = pd.read_sql_query(text(sql_text), conn)
        except Exception:
            last_select = None
            for stmt in [s.strip() for s in sql_text.split(";") if s.strip()]:
                if stmt.lower().startswith("select"):
                    last_select = stmt
            if last_select is None:
                return pd.DataFrame()
            df = pd.read_sql_query(text(last_select), conn)
    return df

# --------------------------------
# Grup bazında tahmin ve kapasite
# --------------------------------
def forecast_group(df_group, h, s, date_col, demand_col, freq_hint):
    """Tek seri için SARIMA ve LR tahmini üretir, ortalamalarını verir."""
    ts = df_group.set_index(date_col)[demand_col].sort_index()
    ts, _ = ensure_continuous_index(ts, freq_hint)

    sar = fit_predict_sarima(ts, h, s)
    lr = fit_predict_lr(ts, h, s)

    out = pd.DataFrame({
        "tarih": sar.index,
        "sarima": sar.values,
        "linreg": lr.values
    })
    out["tahmin_mean"] = out[["sarima", "linreg"]].mean(axis=1)
    return out

def merge_capacity(forecasts_df, capacity_df, depo_col, urun_col):
    """Kapasite ile birleştir ve kullanım oranı hesapla."""
    if capacity_df is None or capacity_df.empty:
        forecasts_df["kapasite"] = np.nan
        forecasts_df["kullanim_orani"] = np.nan
        return forecasts_df

    cap = capacity_df.copy()
    cap.columns = [c.lower() for c in cap.columns]

    cap_depo = next((c for c in ["depo", "warehouse", "site"] if c in cap.columns), None)
    cap_kap = next((c for c in ["kapasite", "capacity", "kapasite_birim", "kapasite_toplam"] if c in cap.columns), None)
    cap_urun = "urun" if "urun" in cap.columns else None

    keys = []
    if depo_col in forecasts_df.columns and cap_depo:
        keys.append((depo_col, cap_depo))
    if urun_col and urun_col in forecasts_df.columns and cap_urun:
        keys.append((urun_col, cap_urun))

    if not keys or not cap_kap:
        forecasts_df["kapasite"] = np.nan
        forecasts_df["kullanim_orani"] = np.nan
        return forecasts_df

    left_on = [k[0] for k in keys]
    right_on = [k[1] for k in keys]

    merged = forecasts_df.merge(
        cap[[*right_on, cap_kap]].drop_duplicates(),
        how="left",
        left_on=left_on,
        right_on=right_on
    ).drop(columns=right_on, errors="ignore")
    merged = merged.rename(columns={cap_kap: "kapasite"})
    merged["kullanim_orani"] = merged["tahmin_mean"] / merged["kapasite"]
    return merged

# --------------------------------
# Ana işlem fonksiyonu
# --------------------------------
def run_pipeline(xlsx_file, h, s, col_date, col_demand, col_depo, col_urun, freq):
    df = read_excel_file(xlsx_file, col_date, col_demand, freq=freq)
    freq_hint = infer_freq_or_use(df, col_date, freq)

    group_cols = [c for c in [col_depo, col_urun] if c and c in df.columns]
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        df["_grp"] = "serie"
        group_cols = ["_grp"]
        grouped = df.groupby(group_cols)

    cap_df = load_capacity_from_db()

    out_frames = []
    for keys, g in grouped:
        fc = forecast_group(g, h, s, col_date, col_demand, freq_hint)
        if isinstance(keys, tuple):
            for name, val in zip(group_cols, keys):
                fc[name] = val
        else:
            fc[group_cols[0]] = keys
        out_frames.append(fc)

    res = pd.concat(out_frames, ignore_index=True)

    depo_col = col_depo if col_depo in res.columns else None
    urun_col = col_urun if col_urun in res.columns else None
    res2 = merge_capacity(res, cap_df, depo_col, urun_col)

    lead_cols = []
    if "tarih" in res2.columns: lead_cols.append("tarih")
    if depo_col and depo_col in res2.columns: lead_cols.append(depo_col)
    if urun_col and urun_col in res2.columns: lead_cols.append(urun_col)
    ordered = lead_cols + [c for c in ["sarima", "linreg", "tahmin_mean", "kapasite", "kullanim_orani"] if c in res2.columns]
    other_cols = [c for c in res2.columns if c not in ordered]
    res2 = res2[ordered + other_cols]
    return res2

# --------------------------------
# Rotalar
# --------------------------------
@app.route("/", methods=["GET"])
def index():
    # templates/index.html dosyası gereklidir.
    return render_template("index.html")

@app.route("/forecast", methods=["POST"])
def forecast_endpoint():
    global _LAST_CSV_BYTES

    if "file" not in request.files:
        abort(400, "Dosya yüklenmedi.")

    file = request.files["file"]
    if not file or file.filename == "":
        abort(400, "Geçersiz dosya.")

    h = int(request.form.get("h", "12"))
    s = int(request.form.get("s", "12"))

    col_date = request.form.get("col_date", "tarih").strip()
    col_demand = request.form.get("col_demand", "talep").strip()
    col_depo = request.form.get("col_depo", "depo").strip() or None
    col_urun = request.form.get("col_urun", "urun").strip() or None

    freq = request.form.get("freq", "").strip() or None

    res_df = run_pipeline(file, h, s, col_date, col_demand, col_depo, col_urun, freq)

    csv_buf = io.StringIO()
    res_df.to_csv(csv_buf, index=False)
    _LAST_CSV_BYTES = csv_buf.getvalue().encode("utf-8")

    out_df = res_df.copy()
    if "tarih" in out_df.columns:
        out_df["tarih"] = out_df["tarih"].astype(str)

    resp = {
        "h": h,
        "s": s,
        "n_kayit": int(len(out_df)),
        "columns": list(out_df.columns),
        "data": out_df.to_dict(orient="records"),
        "capacity_source": "database.sql + DATABASE_URL" if os.getenv("DATABASE_URL") else "none"
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
        download_name="forecast_capacity.csv"
    )

# --------------------------------
# Çalıştırma
# --------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
