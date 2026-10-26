"""Cek kesiapan proyek sebelum upload ke GitHub / deploy ke Render.

Cara pakai (venv aktif, di folder proyek):
    python check_setup.py
"""
import json
import os
import re
import sys
from importlib import metadata

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
problems = []


def ok(msg):
    print(f"[OK]    {msg}")


def bad(msg):
    print(f"[GAGAL] {msg}")
    problems.append(msg)


# 1. File wajib
wajib = ["app.py", "templates/index.html", "model.pkl", "scaler.pkl",
         "columns.pkl", "num_cols.pkl", "requirements.txt", ".python-version"]
for f in wajib:
    if os.path.exists(f):
        ok(f"File ada: {f}")
    else:
        bad(f"File tidak ada: {f}  (jalankan sel penyimpanan model di notebook)")

# 2. requirements.txt terkunci & cocok dengan lingkungan
if os.path.exists("requirements.txt"):
    pins = {}
    for line in open("requirements.txt"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
        else:
            bad(f"requirements.txt: baris belum terkunci versinya -> '{line}'")
    for p in ["flask", "gunicorn", "scikit-learn", "pandas", "numpy", "joblib"]:
        if p not in pins:
            bad(f"requirements.txt: paket '{p}' belum ada / belum memakai ==versi")
            continue
        try:
            terpasang = metadata.version(p)
        except metadata.PackageNotFoundError:
            bad(f"Paket '{p}' belum terpasang di lingkungan ini")
            continue
        if terpasang == pins[p]:
            ok(f"{p}=={terpasang} cocok dengan requirements.txt")
        else:
            bad(f"{p}: terpasang {terpasang}, requirements.txt {pins[p]}")

# 3. Versi Python
if os.path.exists(".python-version"):
    v = open(".python-version").read().strip()
    now = f"{sys.version_info.major}.{sys.version_info.minor}"
    if v.startswith(now):
        ok(f"Versi Python cocok ({now})")
    else:
        bad(f".python-version = {v}, tetapi Python lokal {now}")

# 4. Versi scikit-learn saat training
if os.path.exists("model_info.json"):
    import sklearn
    info = json.load(open("model_info.json"))
    if info.get("scikit_learn") == sklearn.__version__:
        ok(f"Model dilatih & dijalankan dengan scikit-learn {sklearn.__version__}")
    else:
        bad(f"Model dilatih dengan scikit-learn {info.get('scikit_learn')}, "
            f"terpasang {sklearn.__version__}")
else:
    bad("model_info.json tidak ada (pakai sel penyimpanan model yang baru)")

# 5a. Template harus versi terbaru
if os.path.exists("templates/index.html"):
    tpl = open("templates/index.html", encoding="utf-8").read()
    if "pay_fields" in tpl and 'class="label"' in tpl:
        ok("templates/index.html adalah versi terbaru")
    else:
        bad("templates/index.html masih VERSI LAMA. Ganti seluruh isinya dengan "
            "index.html versi baru, lalu simpan (Ctrl+S).")

# 5b. Aplikasi bisa dimuat dan memprediksi
try:
    sys.path.insert(0, BASE)
    from app import app, columns

    client = app.test_client()
    ok(f"app.py berhasil dimuat ({len(columns)} fitur model)")
    aman = {"limit_bal": "300000", "age": "45", "sex": "1", "education": "1",
            "marriage": "1", "PAY_0": "-1", "PAY_2": "-1", "PAY_3": "-1",
            "PAY_4": "-1", "PAY_5": "-1", "PAY_6": "-1",
            "bill_amt1": "5000", "pay_amt1": "5000"}
    risiko = dict(aman, limit_bal="20000", age="25", sex="2", education="3",
                  marriage="2", PAY_0="3", PAY_2="3", PAY_3="2",
                  bill_amt1="19000", pay_amt1="0")
    for nama, data in [("nasabah aman", aman), ("nasabah berisiko", risiko)]:
        html = client.post("/", data=data).get_data(as_text=True)
        m = re.search(r'class="label">([^<]+)', html)
        p = re.search(r"<strong>([\d.]+)%", html)
        if m and p and 0 <= float(p.group(1)) <= 100:
            ok(f"Prediksi {nama}: {m.group(1)} ({p.group(1)}%)")
        else:
            err = re.search(r'class="error">([^<]+)', html)
            sebab = err.group(1) if err else "halaman tidak berisi hasil (cek index.html)"
            bad(f"Prediksi {nama} gagal. Penyebab: {sebab}")
    if client.get("/health").status_code == 200:
        ok("Endpoint /health berfungsi")
except Exception as e:
    bad(f"app.py gagal dimuat/dijalankan: {e}")

# 6. Siap untuk GitHub
if os.path.exists(".gitignore") and ".pkl" in open(".gitignore").read():
    bad(".gitignore memblokir file .pkl (model tidak akan terupload)")
else:
    ok(".gitignore tidak memblokir file .pkl")
for f in ["model.pkl", "scaler.pkl", "columns.pkl", "num_cols.pkl"]:
    if os.path.exists(f) and os.path.getsize(f) > 90 * 1024 * 1024:
        bad(f"{f} terlalu besar untuk GitHub (>90 MB)")

print()
if problems:
    print(f"Ada {len(problems)} masalah. Perbaiki dulu sebelum deploy.")
    sys.exit(1)
print("Semua pemeriksaan lulus. Aman untuk upload ke GitHub dan deploy.")