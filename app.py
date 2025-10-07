# app.py
# Flask tabanlı Talep Tahminleme + Kapasite Planlama tek dosya uygulaması
# - Girdi: .xlsx (geçmiş talep verisi)
# - DB: DATABASE_URL ortam değişkeni ile bağlanır, ./database.sql içeriğini çalıştırıp depo parametrelerini çeker
# - Modeller: SARIMA (statsmodels) ve Lineer Regresyon (scikit-learn)
# - Çıktı: Basit HTML arayüz + JSON ve CSV indirme
# Not: Tüm Python fonksiyonları bu dosyada tutulmuştur.

import os
import io
import json
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    render_template_string,
    abort
)

# ML / Zaman serisi
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

# DB bağlantısı
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ----------------------------
# Yardımcı: HTML şablon
# ----------------------------
INDEX_HTML = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Talep Tahminleme + Kapasite Planlama</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 20px; max-width: 960px; }
    label { display:block; margin: 6px 0 2px; }
    input, select { padding: 8px; width: 100%; box-sizing: border-box; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .btn { margin-top: 12px; padding: 10px 14px; background:#111; color:#fff; border:0; border-radius:8px; cursor:pointer;}
    small { color:#555; }
    pre { background:#f7f7f7; padding:12px; border-radius:8px; overflow:auto; }
    .hint { color:#444; font-size: 14px; }
    .ok { background:#e6ffed; padding:8px 12px; border-radius:8px; margin-top:10px; display:inline-block;}
  </style>
</head>
<body>
  <div class="card">
    <h2>Eti Maden · Talep Tahminleme + Kapasite Planlama</h2>
    <form id="f" enctype="multipart/form-data" method="POST" action="/forecast">
      <label>Geçmiş veri (.xlsx)</label>
      <input type="file" name="file" accept=".xlsx" required />

      <div class="row">
        <div>
          <label>Tahmin ufku (h, dönem sayısı)</label>
          <input type="number" name="h" value="12" min="1" />
        </div>
        <div>
          <label>Mevsimsellik periyodu (s)</label>
          <input type="number" name="s" value="12" min="1" />
        </div>
      </div>

      <div class="row">
        <div>
          <label>Tarih sütunu adı</label>
          <input type="text" name="col_date" value="tarih" />
          <small>Varsayılan: <code>tarih</code></small>
        </div>
        <div>
          <label>Talep sütunu adı</label>
          <input type="text" name="col_demand" value="talep" />
          <small>Varsayılan: <code>talep</code></small>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Depo sütunu adı (opsiyonel)</label>
          <input type="text" name="col_depo" value="depo" />
          <small>Varsayılan: <code>depo</code></small>
        </div>
        <div>
          <label>Ürün sütunu adı (opsiyonel)</label>
          <input type="text" name="col_urun" value="urun" />
          <small>Varsayılan: <code>urun</code></small>
        </div>
      </div>

      <label>Frekans (pandas offset alias, opsiyonel)</label>
      <input type="text" name="freq" value="" placeholder="M, MS, W, D vb. boş bırakılırsa otomatik" />

      <button class="btn" type="submit">Çalıştır</button>
    </form>

    <p class="hint">
      .xlsx dosyasında en az <b>tarih</b> ve <b>talep</b> sütunları bulunmalıdır.
      İsteğe bağlı olarak <b>depo</b> ve/veya <b>urun</b> sütunlarıyla çoklu seri çalışır.
    </p>
    <p class="hint">
      <b>database.sql</b> dosyası aynı klasörde olmalıdır. Uygulama, <code>DATABASE_URL</code> ile
      veritabanına bağlanır ve bu SQL'i çalıştırarak depo parametrelerini çeker.
      Örn. Postgres: <code>postgresql+psycopg2://kullanici:parola@host:port/db</code>
    </p>
    <div class="ok">POST /forecast JSON döndürür. CSV indirmek için: <code>GET /download_last.csv</code></div>
    <pre>curl -F "file=@veri.xlsx" -F "h=12" -F "s=12" http://localhost:8080/forecast</pre>
  </div>
</body>
</html>
"""

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

    # Frekans ataması
    if freq:
        df = df.sort_values(col_date)
        # Düz indeksli zaman serisine dönüştürmek için resample opsiyonunu gruplarken kullanacağız
    return df

def infer_freq_or_use(df, date_col, freq):
    """Frekans verilmişse onu döndürür, değilse pandas ile tahmin etmeye çalışır."""
    if freq and isinstance(freq, str) and freq.strip():
        return freq.strip()
    # Tahmin
    try:
        return pd.infer_freq(df.sort_values(date_col)[date_col])
    except Exception:
        return None  # ileri adımda fallback

def ensure_continuous_index(s, freq_hint):
    """Seriyi belirtilen frekansa göre eksiksiz zaman eksenine yerleştirir."""
    s = s.copy()
    s = s.sort_index()
    if freq_hint is None:
        # fallback: medyan farktan gün cinsinden yakın frekans
        diffs = s.index.to_series().diff().dropna()
        if len(diffs) == 0:
            freq_hint = "D"
        else:
            median_days = diffs.median() / np.timedelta64(1, 'D')
            if median_days >= 27 and median_days <= 31:
                freq_hint = "M"
            elif median_days >= 6 and median_days <= 8:
                freq_hint = "W"
            else:
                freq_hint = "D"
    # Reindex
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq=freq_hint)
    s = s.reindex(full_idx)
    # Ara değerler lineer doldurulabilir ya da ileri doldurma
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
        fc.index = pd.date_range(y.index[-1] + (y.index[-1] - y.index[-2] if len(y) > 1 else pd.Timedelta(days=1)),
                                 periods=h, freq=y.index.freq or y.index.inferred_freq)
        return fc
    except Exception as e:
        raise RuntimeError(f"SARIMA başarısız: {e}")

# --------------------------------
# Model: Lineer Regresyon (özellik tabanlı)
# --------------------------------
def build_lr_features(y, s):
    """LR için basit öznitelikler: trend, sin-cos mevsimsellik, gecikme."""
    n = len(y)
    t = np.arange(n).reshape(-1, 1)
    # mevsimsellik
    s = int(s) if s and int(s) > 0 else 12
    sin1 = np.sin(2 * np.pi * np.arange(n) / s).reshape(-1, 1)
    cos1 = np.cos(2 * np.pi * np.arange(n) / s).reshape(-1, 1)
    # lag1
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

    # ileri projeksiyon
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
        last_val = y_hat  # iteratif lag

    # tarih indeksleri
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
    DATABASE_URL ortam değişkeni varsa bağlanır.
    ./database.sql içeriğini okur ve tek bir SELECT script'i olarak çalıştırır.
    Sonuç bir DataFrame döndürür.
    Beklenen örnek şema (esnek):
      - depo (veya warehouse)
      - kapasite (ya da kapasite_birim, capacity, vb.)
      - opsiyonel: urun, tarihsel kapasite, vb.
    """
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return pd.DataFrame()  # DB yoksa boş dön

    sql_path = os.path.join(os.getcwd(), "database.sql")
    if not os.path.exists(sql_path):
        return pd.DataFrame()

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_text = f.read().strip()
    if not sql_text:
        return pd.DataFrame()

    engine = create_engine(db_url, pool_pre_ping=True)

    # Birden fazla statement varsa, çoğu sürücü tek çağrıda çalıştırmayabilir.
    # Bu yüzden TRY: SELECT üretmesi beklenen son statement'ı çalıştır.
    # Basit yaklaşım: tamamını text(sql) ile çalıştırıp mümkünse DataFrame'e al.
    with engine.connect() as conn:
        try:
            df = pd.read_sql_query(text(sql_text), conn)
        except Exception:
            # Bazı DB'lerde text(sql) ile SELECT döndürmeyebilir.
            # Son SELECT'i bulmayı deneyelim.
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
    # Seri oluştur
    ts = df_group.set_index(date_col)[demand_col].sort_index()
    ts, freq_used = ensure_continuous_index(ts, freq_hint)

    # Modeller
    sar = fit_predict_sarima(ts, h, s)
    lr = fit_predict_lr(ts, h, s)

    # Birleştir
    out = pd.DataFrame({
        "tarih": sar.index,
        "sarima": sar.values,
        "linreg": lr.values
    })
    out["tahmin_mean"] = out[["sarima", "linreg"]].mean(axis=1)
    return out

def merge_capacity(forecasts_df, capacity_df, depo_col, urun_col):
    """Kapasite bilgisi ile birleştir ve kullanım oranı hesapla."""
    if capacity_df is None or capacity_df.empty:
        forecasts_df["kapasite"] = np.nan
        forecasts_df["kullanim_orani"] = np.nan
        return forecasts_df

    # Esnek eşleşme: depo ve ürün sütunlarını tahmin tablosunda mevcut anahtarlarla eşleştir
    # Beklenen kolon isimlerini normalize edelim
    cap = capacity_df.copy()
    cap_cols = [c.lower() for c in cap.columns]
    cap.columns = cap_cols

    # depo kolonunu bul
    cap_depo = None
    for cand in ["depo", "warehouse", "site"]:
        if cand in cap.columns:
            cap_depo = cand
            break

    # kapasite kolonunu bul
    cap_kap = None
    for cand in ["kapasite", "capacity", "kapasite_birim", "kapasite_toplam"]:
        if cand in cap.columns:
            cap_kap = cand
            break

    # ürün kolonunu bul (opsiyonel)
    cap_urun = "urun" if "urun" in cap.columns else None

    # Eşleşme anahtarları
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

    merged = forecasts_df.merge(cap[[*right_on, cap_kap]].drop_duplicates(), how="left",
                                left_on=left_on, right_on=right_on)
    merged = merged.drop(columns=right_on, errors="ignore")
    merged = merged.rename(columns={cap_kap: "kapasite"})
    merged["kullanim_orani"] = merged["tahmin_mean"] / merged["kapasite"]
    return merged

# --------------------------------
# Ana işlem fonksiyonu
# --------------------------------
def run_pipeline(xlsx_file, h, s, col_date, col_demand, col_depo, col_urun, freq):
    df = read_excel_file(xlsx_file, col_date, col_demand, freq=freq)
    freq_hint = infer_freq_or_use(df, col_date, freq)

    # Gruplama boyutları
    group_cols = [c for c in [col_depo, col_urun] if c and c in df.columns]
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        # Tek seri için sentetik grup
        df["_grp"] = "serie"
        group_cols = ["_grp"]
        grouped = df.groupby(group_cols)

    # Kapasite verisi
    cap_df = load_capacity_from_db()

    # Her grup için tahmin
    out_frames = []
    for keys, g in grouped:
        fc = forecast_group(g, h, s, col_date, col_demand, freq_hint)
        # Grup anahtarlarını ekle
        if isinstance(keys, tuple):
            for name, val in zip(group_cols, keys):
                fc[name] = val
        else:
            fc[group_cols[0]] = keys
        out_frames.append(fc)

    res = pd.concat(out_frames, ignore_index=True)

    # Kapasite ile birleştir
    depo_col = col_depo if col_depo in res.columns else None
    urun_col = col_urun if col_urun in res.columns else None
    res2 = merge_capacity(res, cap_df, depo_col, urun_col)

    # Sütun sırası
    lead_cols = []
    if "tarih" in res2.columns: lead_cols.append("tarih")
    if depo_col and depo_col in res2.columns: lead_cols.append(depo_col)
    if urun_col and urun_col in res2.columns: lead_cols.append(urun_col)
    ordered = lead_cols + [c for c in ["sarima", "linreg", "tahmin_mean", "kapasite", "kullanim_orani"]
                           if c in res2.columns]
    other_cols = [c for c in res2.columns if c not in ordered]
    res2 = res2[ordered + other_cols]

    return res2

# --------------------------------
# Rotalar
# --------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/forecast", methods=["POST"])
def forecast_endpoint():
    global _LAST_CSV_BYTES

    if "file" not in request.files:
        abort(400, "Dosya yüklenmedi.")

    file = request.files["file"]
    if not file or file.filename == "":
        abort(400, "Geçersiz dosya.")

    # Parametreler
    h = int(request.form.get("h", "12"))
    s = int(request.form.get("s", "12"))

    col_date = request.form.get("col_date", "tarih").strip()
    col_demand = request.form.get("col_demand", "talep").strip()
    col_depo = request.form.get("col_depo", "depo").strip()
    col_urun = request.form.get("col_urun", "urun").strip()
    col_depo = col_depo if col_depo else None
    col_urun = col_urun if col_urun else None

    freq = request.form.get("freq", "").strip()
    freq = freq if freq else None

    # Çalıştır
    res_df = run_pipeline(file, h, s, col_date, col_demand, col_depo, col_urun, freq)

    # CSV önbelleğe al
    csv_buf = io.StringIO()
    res_df.to_csv(csv_buf, index=False)
    _LAST_CSV_BYTES = csv_buf.getvalue().encode("utf-8")

    # JSON döndürme
    # Tarihler serileştirilebilir formata
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
    # Railway default port
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
