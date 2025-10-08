# Kullanıcı kayıt + giriş (Blueprint)
import os
import pymysql
from flask import Blueprint, render_template, request, redirect, url_for, session

from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint("auth", __name__)

# ENV (MySQL için app.py ile aynı isimler)
MYSQL_HOST = os.getenv("MYSQLHOST")
MYSQL_PORT = int(os.getenv("MYSQLPORT", "3306"))
MYSQL_USER = os.getenv("MYSQLUSER", "root")
MYSQL_PASSWORD = os.getenv("MYSQLPASSWORD")
MYSQL_DB = os.getenv("MYSQLDATABASE")

def db():
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB,
        autocommit=True, cursorclass=pymysql.cursors.DictCursor
    )

# Users tablosu yoksa oluştur
def ensure_users():
    with db() as c, c.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          email VARCHAR(255) NOT NULL UNIQUE,
          password_hash VARCHAR(255) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

@auth_bp.before_app_request
def _ensure_users_once():
    # hafif kontrol: tablo var mı diye 1 kez dener
    if not getattr(_ensure_users_once, "done", False):
        try:
            ensure_users()
            _ensure_users_once.done = True
        except Exception as e:
            print("users ensure error:", e)

@auth_bp.get("/login")
def login_page():
    return render_template("login.html")

@auth_bp.post("/login")
def login():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    if not email or not password:
        return redirect(url_for("auth.login_page"))
    with db() as c, c.cursor() as cur:
        cur.execute("SELECT id, password_hash FROM users WHERE email=%s", (email,))
        row = cur.fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return redirect(url_for("auth.login_page"))
    session["user_id"] = row["id"]
    session["user_email"] = email
    return redirect(url_for("index"))

@auth_bp.get("/register")
def register_page():
    return render_template("register.html")

@auth_bp.post("/register")
def register():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    if not email or len(password) < 6:
        return redirect(url_for("auth.register_page"))
    pwd = generate_password_hash(password)
    try:
        with db() as c, c.cursor() as cur:
            cur.execute("INSERT INTO users(email, password_hash) VALUES(%s,%s)", (email, pwd))
    except Exception:
        # e-posta zaten var vb.
        return redirect(url_for("auth.register_page"))
    return redirect(url_for("auth.login_page"))

@auth_bp.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login_page"))
