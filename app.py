# app.py — Flask + MySQL (Railway)
import os, io, json, uuid, warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template, abort
from sklearn.linear_model import LinearRegression
import pymysql

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# -------- ENV (Railway Variables) --------
MYSQL_HOST = os.getenv("MYSQLHOST")                    # MySQL servisindeki RAILWAY_PRIVATE_DOMAIN
MYSQL_PORT = int(os.getenv("MYSQLPORT", "3306"))
MYSQL_USER = os.getenv("MYSQLUSER", "root")
MYSQL_PASSWORD = os.getenv("MYSQLPASSWORD")
MYSQL_DB = os.getenv("MYSQLDATABASE")
PORT = int(os.getenv("PORT", "8080"))

# -------- DB helpers --------
def db():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )

def ensure_schema():
    with db() as c, c.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_results(
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            run_id CHAR(36) NOT NULL,
            tarih DATETIME NULL,
            urun VARCHAR(255) NOT NULL,
            linreg DOUBLE NULL,
            wma DOUBLE NULL,
            tahmin_mean DOUBLE NULL,
            kapasite DOUBLE NULL,
            kullanim_orani DOUBLE NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

_schema_ready = False

@app.before_request
def _init_db_once():
    # DB hazır değilse uygulama çökmesin
    global _schema_ready
    if _schema_ready:
        return
    try:
        ensure_schema()
        _schema_ready = True
    except Exception as e:
        # health dışı isteklerde hata yükseltmeyelim, log yeter
        print("ensure_schema error:", e)

# -------- IO --------
def read_excel_file(file_storage, col_date, col_demand, col_depo, col_urun):
    try:
        df = pd.read_excel(file_storage)
    except Exception as e:
        abort(400, f"XLSX okunamadı: {e}")
    df.columns = df.columns.str.strip().str.lower()

    for c in [col_date, col_demand]:
        if c not in df.columns:
            abort(400, f"Gerekli sütun yok: {c}")

    if col_depo and col_depo not in df.columns:
        df[col_depo] = np.nan
    if col_urun and col_urun not in df.columns:
        abort(400, f"Ürün bazlı tahmin için '{col_urun}' sütunu gerekli.")

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df[col_demand] = pd.to_numeric(df[col_demand], errors="coerce")
    df = df.dropna(subset=[col_date, col_demand, col_urun])
    return df

def read_capacity_excel(cap_storage):
    if not cap_storage:
        return pd.DataFrame()
    try:
        cap = pd.read_excel(cap_storage)
    except Exception as e:
        abort(400, f"Kapasite XLSX okunamadı: {e}")

    cap.columns = cap.columns.str.strip().str.lower()
    kap_col = next((c for c in ["kapasite", "capacity"] if c in cap.columns), None)
    if kap_col is None:
        abort(400, "Kapasite dosyasında 'kapasite' veya 'capacity' kolonu yok.")

    has_urun = "urun" in cap.columns
    has_depo = "depo" in cap.columns

    if has_urun and has_depo:
        out = (cap[["urun", kap_col]]
               .groupby("urun", as_index=False).sum()
               .rename(columns={kap_col: "kapasite"}))
    elif has_urun:
        out = cap[["urun", kap_col]].rename(columns={kap_col: "kapasite"})
    else:
        return pd.DataFrame(columns=["urun", "kapasite"])

    out["urun"] = out["urun"].astype(str)
    return out[["urun", "kapasite"]]

# -------- Time series utils --------
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

# -------- Models --------
def fit_predict_lr(y: np.ndarray, h: int):
    n = len(y)
    X = np.arange(n).reshape(-1, 1)
    m = LinearRegression().fit(X, y.astype(float))
    X_fut = np.arange(n, n + h).reshape(-1, 1)
    return m.predict(X_fut).tolist()

def predict_wma(y: np.ndarray, h: int, window: int = 3, weights: np.ndarray | None = None):
    if window < 1:
        window = 3
    y_list = list(y.astype(float))
    if weights is None:
        weights = np.arange(1, window + 1, dtype=float)
    else:
        weights = np.array(weights, dtype=float)
        if len(weights) != window:
            window = len(weights)
    w = weights / weights.sum()

    preds = []
    for _ in range(h):
        src = y_list[-window:] if len(y_list) >= window else ([y_list[-1]] * (window - len(y_list)) + y_list)
        val = float(np.dot(src[-window:], w))
        preds.append(val)
        y_list.append(val)
    return preds

# -------- Pipeline --------
def run_pipeline(xlsx_file, capfile, h, col_date, col_demand, col_depo, col_urun, wma_window):
    df = read_excel_file(xlsx_file, col_date, col_demand, col_depo, col_urun)

    agg = (
        df[[col_date, col_urun, col_demand]]
        .groupby([col_urun, col_date], as_index=False)
        .sum(numeric_only=True)
        .rename(columns={col_urun: "urun", col_date: "tarih", col_demand: "talep"})
    )

    cap_df = read_capacity_excel(capfile) if capfile else pd.DataFrame()

    outs = []
    for urun, g in agg.groupby("urun", as_index=False):
        g = g.sort_values("tarih")
        y = g["talep"].to_numpy(dtype=float)
        if len(y) < 2:
            continue
        lr = fit_predict_lr(y, h)
        wma = predict_wma(y, h, window=wma_window)
        fut_idx = next_index(g["tarih"].iloc[-1], h, g["tarih"])
        out = pd.DataFrame({
            "tarih": fut_idx,
            "urun": urun,
            "linreg": lr,
            "wma": wma,
        })
        out["tahmin_mean"] = out[["linreg", "wma"]].mean(axis=1)
        outs.append(out)

    if not outs:
        abort(400, "Yeterli veri yok veya 'urun' sütunu boş.")

    res = pd.concat(outs, ignore_index=True)

    if not cap_df.empty:
        res = res.merge(cap_df, how="left", on="urun")
        res["kullanim_orani"] = res["tahmin_mean"] / res["kapasite"]

    lead = ["tarih", "urun"]
    ordered = lead + [c for c in ["linreg", "wma", "tahmin_mean", "kapasite", "kullanim_orani"] if c in res.columns]
    other = [c for c in res.columns if c not in ordered]
    res = res[ordered + other]

    return res, cap_df

def save_results_to_db(run_id: str, df: pd.DataFrame):
    if "tarih" in df.columns:
        df = df.copy()
        df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")

    rows = []
    for _, r in df.iterrows():
        rows.append((
            run_id,
            None if pd.isna(r.get("tarih")) else r["tarih"].to_pydatetime(),
            str(r.get("urun", "")),
            None if pd.isna(r.get("linreg")) else float(r["linreg"]),
            None if pd.isna(r.get("wma")) else float(r["wma"]),
            None if pd.isna(r.get("tahmin_mean")) else float(r["tahmin_mean"]),
            None if "kapasite" not in df.columns or pd.isna(r.get("kapasite")) else float(r["kapasete"]) if "kapasete" in r else float(r["kapasite"]),
            None if "kullanim_orani" not in df.columns or pd.isna(r.get("kullanim_orani")) else float(r["kullanim_orani"]),
        ))

    if not rows:
        return

    with db() as c, c.cursor() as cur:
        cur.executemany("""
            INSERT INTO forecast_results
            (run_id, tarih, urun, linreg, wma, tahmin_mean, kapasite, kullanim_orani)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, rows)

# -------- Routes --------
@app.route("/")
def index():
    # templates/index.html varsa render eder, yoksa kısa info döner
    try:
        return render_template("index.html")
    except Exception:
        return "OK", 200

@app.route("/forecast", methods=["POST"])
def forecast_endpoint():
    global _LAST_CSV_BYTES

    if "file" not in request.files:
        abort(400, "Talep dosyası yüklenmedi.")
    file = request.files["file"]
    capfile = request.files.get("capfile")  # opsiyonel

    h = int(request.form.get("h", "12"))
    wma_window = int(request.form.get("wma_window", "3"))

    col_date = (request.form.get("col_date", "tarih") or "tarih").strip().lower()
    col_demand = (request.form.get("col_demand", "talep") or "talep").strip().lower()
    col_depo = (request.form.get("col_depo", "depo") or "").strip().lower() or "depo"
    col_urun = (request.form.get("col_urun", "urun") or "urun").strip().lower()

    res_df, _ = run_pipeline(file, capfile, h, col_date, col_demand, col_depo, col_urun, wma_window)

    run_id = str(uuid.uuid4())
    try:
        save_results_to_db(run_id, res_df)
    except Exception as e:
        print("DB save error:", e)

    csv_buf = io.StringIO()
    res_df.to_csv(csv_buf, index=False)
    _LAST_CSV_BYTES = csv_buf.getvalue().encode("utf-8")

    out_df = res_df.copy()
    if "tarih" in out_df.columns:
        out_df["tarih"] = out_df["tarih"].astype(str)
    out_df = out_df.replace([np.inf, -np.inf], np.nan)

    data = json.loads(out_df.to_json(orient="records"))
    return jsonify({
        "run_id": run_id,
        "h": h,
        "wma_window": wma_window,
        "n_kayit": len(out_df),
        "columns": list(out_df.columns),
        "data": data,
        "capacity_source": "xlsx" if (capfile and getattr(capfile, "filename", "")) else "none",
    })

@app.route("/runs/<run_id>")
def get_run(run_id):
    with db() as c, c.cursor() as cur:
        cur.execute("""
            SELECT tarih, urun, linreg, wma, tahmin_mean, kapasite, kullanim_orani
            FROM forecast_results
            WHERE run_id=%s
            ORDER BY tarih
        """, (run_id,))
        rows = cur.fetchall()
    return jsonify({"run_id": run_id, "rows": rows})

@app.route("/download_last.csv")
def download_last():
    if not globals().get("_LAST_CSV_BYTES"):
        abort(404, "Önce /forecast çalıştırılmalı.")
    return send_file(
        io.BytesIO(_LAST_CSV_BYTES),
        mimetype="text/csv",
        as_attachment=True,
        download_name="forecast_capacity.csv",
    )

@app.get("/health")
def health():
    try:
        with db() as c:
            c.ping(reconnect=True)
        db_ok = True
    except Exception as e:
        print("health db ping error:", e)
        db_ok = False
    return {"ok": True, "db": db_ok}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
