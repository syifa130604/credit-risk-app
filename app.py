import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.5  # ambang batas keputusan (decision boundary)


# ---------------------------------------------------------------
# 1. Muat model, scaler, dan daftar kolom hasil training (Joblib)
# ---------------------------------------------------------------
def load_file(name):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File '{name}' tidak ditemukan di {BASE_DIR}. "
            "Jalankan sel penyimpanan model di notebook, lalu pastikan file .pkl "
            "berada satu folder dengan app.py."
        )
    return joblib.load(path)


model = load_file("model.pkl")
scaler = load_file("scaler.pkl")
columns = list(load_file("columns.pkl"))
num_cols = list(load_file("num_cols.pkl"))


def warn_if_version_mismatch():
    """Beri peringatan di log bila versi scikit-learn berbeda dari saat training."""
    info_path = os.path.join(BASE_DIR, "model_info.json")
    if not os.path.exists(info_path):
        return
    import sklearn
    with open(info_path) as f:
        info = json.load(f)
    trained = info.get("scikit_learn")
    if trained and trained != sklearn.__version__:
        print(
            f"PERINGATAN: model dilatih dengan scikit-learn {trained}, "
            f"tetapi yang terpasang {sklearn.__version__}. "
            "Samakan versi di requirements.txt."
        )


warn_if_version_mismatch()

# ---------------------------------------------------------------
# 2. Validasi kolom model (gagal cepat dengan pesan yang jelas)
# ---------------------------------------------------------------
PAY_FIELDS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
CATEGORICAL = {"SEX": "sex", "EDUCATION": "education", "MARRIAGE": "marriage"}
KNOWN = {"LIMIT_BAL", "AGE", *PAY_FIELDS,
         *[f"BILL_AMT{i}" for i in range(1, 7)],
         *[f"PAY_AMT{i}" for i in range(1, 7)]}


def is_known(col):
    return col in KNOWN or any(col == p or col.startswith(p + "_") for p in CATEGORICAL)


UNKNOWN = [c for c in columns if not is_known(c)]
if UNKNOWN:
    raise RuntimeError(
        f"Kolom model tidak dikenali oleh form: {UNKNOWN}. "
        "Tambahkan input untuk kolom tersebut di app.py dan index.html."
    )

# Nilai awal form (contoh nasabah dengan riwayat bayar normal)
DEFAULTS = {
    "limit_bal": "100000", "age": "35",
    "sex": "2", "education": "2", "marriage": "1",
    "PAY_0": "0", "PAY_2": "0", "PAY_3": "0",
    "PAY_4": "0", "PAY_5": "0", "PAY_6": "0",
    "bill_amt1": "30000", "pay_amt1": "3000",
}


# ---------------------------------------------------------------
# 3. Fungsi bantu validasi input
# ---------------------------------------------------------------
def parse_number(form, name, label, minimum=0, maximum=10_000_000, as_int=False, optional=False):
    raw = (form.get(name) or "").strip()
    if raw == "":
        if optional:
            return None
        raise ValueError(f"{label} wajib diisi.")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{label} harus berupa angka.")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{label} tidak valid.")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} harus antara {minimum:,} dan {maximum:,}.")
    return int(value) if as_int else value


def parse_choice(form, name, label, allowed):
    try:
        value = int(form.get(name))
    except (TypeError, ValueError):
        raise ValueError(f"{label} tidak valid.")
    if value not in allowed:
        raise ValueError(f"{label} tidak valid.")
    return value


def build_features(form):
    """Ubah input form menjadi satu baris fitur sesuai kolom saat training."""
    values = {}
    values["LIMIT_BAL"] = parse_number(form, "limit_bal", "Batas kredit", minimum=1)
    values["AGE"] = parse_number(form, "age", "Usia", minimum=18, maximum=100, as_int=True)

    for name in PAY_FIELDS:
        values[name] = parse_choice(form, name, f"Status pembayaran {name}", range(-2, 9))

    # Tagihan & pembayaran bulan terakhir wajib; bulan lain opsional.
    # Jika dikosongkan, dipakai nilai bulan terakhir.
    bill1 = parse_number(form, "bill_amt1", "Tagihan bulan terakhir", minimum=-1_000_000)
    pay1 = parse_number(form, "pay_amt1", "Pembayaran bulan terakhir")
    for i in range(1, 7):
        b = parse_number(form, f"bill_amt{i}", f"Tagihan {i} bulan lalu",
                         minimum=-1_000_000, optional=True) if i > 1 else bill1
        p = parse_number(form, f"pay_amt{i}", f"Pembayaran {i} bulan lalu",
                         optional=True) if i > 1 else pay1
        values[f"BILL_AMT{i}"] = bill1 if b is None else b
        values[f"PAY_AMT{i}"] = pay1 if p is None else p

    choices = {
        "SEX": parse_choice(form, "sex", "Jenis kelamin", (1, 2)),
        "EDUCATION": parse_choice(form, "education", "Pendidikan", (1, 2, 3, 4)),
        "MARRIAGE": parse_choice(form, "marriage", "Status pernikahan", (1, 2, 3)),
    }

    row = {c: values.get(c, 0) for c in columns}
    # Cocokkan kolom one-hot (mis. SEX_2) secara otomatis dari nama kolom model
    for prefix, chosen in choices.items():
        if prefix in row:                      # kolom asli (tanpa one-hot)
            row[prefix] = chosen
        for col in columns:
            if col.startswith(prefix + "_"):
                try:
                    row[col] = 1 if float(col[len(prefix) + 1:]) == chosen else 0
                except ValueError:
                    row[col] = 0

    return pd.DataFrame([row])[columns]


# ---------------------------------------------------------------
# 4. Route
# ---------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    form = dict(DEFAULTS)

    if request.method == "POST":
        form = request.form.to_dict()
        try:
            X = build_features(request.form)
            X[num_cols] = scaler.transform(X[num_cols])
            prob = float(model.predict_proba(X)[0][1])

            if prob >= THRESHOLD:
                label, status = "Berisiko Gagal Bayar", "risk"
            else:
                label, status = "Tidak Berisiko", "safe"

            result = {
                "label": label,
                "status": status,
                "prob": round(prob * 100, 2),
                "prob_safe": round((1 - prob) * 100, 2),
                "threshold": int(THRESHOLD * 100),
            }
        except ValueError as e:
            error = str(e)
        except Exception as e:  # jangan tampilkan halaman error mentah ke pengguna
            app.logger.exception("Gagal memprediksi")
            error = "Terjadi kesalahan saat memproses data. Periksa kembali isian Anda."

    return render_template("index.html", result=result, error=error, form=form)


@app.route("/health")
def health():
    """Dipakai untuk mengecek aplikasi hidup (juga bisa dipakai Render)."""
    return jsonify(status="ok", n_features=len(columns))


if __name__ == "__main__":
    app.run(debug=True)