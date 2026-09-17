import base64, os, sqlite3, csv, io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(APP_DIR, "attendance.db")
DATASET = os.path.join(APP_DIR, "dataset")
MODEL = os.path.join(APP_DIR, "trainer.yml")
os.makedirs(DATASET, exist_ok=True)

app = Flask(__name__)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    con.execute("""CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_no TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        UNIQUE(student_id, date),
        FOREIGN KEY(student_id) REFERENCES students(id)
    )""")
    con.commit()
    con.close()

def decoder(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def train_model():
    if not hasattr(cv2, "face"):
        raise RuntimeError("opencv-contrib-python is required for LBPH face recognition.")
    faces, labels = [], []
    con = db()
    students = con.execute("SELECT id FROM students").fetchall()
    con.close()
    for row in students:
        sid = row["id"]
        folder = os.path.join(DATASET, str(sid))
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(sid)
    if not faces:
        raise RuntimeError("No face samples found. Register at least one student.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL)
    return len(faces)

def recognize(frame):
    if not os.path.exists(MODEL):
        return None, None, "Model not trained"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5, minSize=(80,80))
    if len(faces) == 0:
        return None, None, "No face detected"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL)
    best = None
    for (x,y,w,h) in faces:
        label, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        # LBPH confidence: lower is better. 65 is a practical starting threshold.
        if best is None or confidence < best[1]:
            best = (label, confidence)
    if best is None or best[1] > 65:
        return None, best[1] if best else None, "Face not recognized"
    con = db()
    student = con.execute("SELECT * FROM students WHERE id=?", (best[0],)).fetchone()
    con.close()
    if not student:
        return None, best[1], "Unknown student"
    return dict(student), best[1], "Recognized"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/api/students", methods=["GET"])
def students():
    con = db()
    rows = [dict(x) for x in con.execute("SELECT * FROM students ORDER BY id DESC").fetchall()]
    con.close()
    return jsonify(rows)

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    roll = (data.get("roll_no") or "").strip()
    images = data.get("images") or []
    if not name or not roll:
        return jsonify(ok=False, message="Name and roll number are required."), 400
    if len(images) < 10:
        return jsonify(ok=False, message="Capture at least 10 samples."), 400

    con = db()
    try:
        cur = con.execute(
            "INSERT INTO students(roll_no,name,created_at) VALUES(?,?,?)",
            (roll, name, datetime.now().isoformat(timespec="seconds"))
        )
        sid = cur.lastrowid
        con.commit()
    except sqlite3.IntegrityError:
        con.close()
        return jsonify(ok=False, message="Roll number already exists."), 409
    con.close()

    folder = os.path.join(DATASET, str(sid))
    os.makedirs(folder, exist_ok=True)
    saved = 0
    for i, item in enumerate(images):
        frame = decoder(item)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = FACE_CASCADE.detectMultiScale(gray, 1.2, 5, minSize=(80,80))
        if len(detected) == 0:
            continue
        # save the largest detected face
        x,y,w,h = max(detected, key=lambda r:r[2]*r[3])
        crop = gray[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(folder, f"{saved:03d}.jpg"), crop)
        saved += 1

    if saved < 5:
        shutil = __import__("shutil")
        shutil.rmtree(folder, ignore_errors=True)
        con = db(); con.execute("DELETE FROM students WHERE id=?", (sid,)); con.commit(); con.close()
        return jsonify(ok=False, message="Could not detect enough faces. Please register again in good lighting."), 400
    try:
        train_model()
    except Exception as e:
        return jsonify(ok=False, message=str(e)), 500
    return jsonify(ok=True, message=f"{name} registered with {saved} face samples.")

@app.route("/api/recognize", methods=["POST"])
def recognize_api():
    data = request.get_json()
    frame = decoder(data.get("image",""))
    if frame is None:
        return jsonify(ok=False, message="Invalid image."), 400
    student, confidence, status = recognize(frame)
    if not student:
        return jsonify(ok=False, status=status, confidence=confidence, message=status)
    now = datetime.now()
    date, tm = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    con = db()
    try:
        con.execute(
            "INSERT INTO attendance(student_id,date,time) VALUES(?,?,?)",
            (student["id"], date, tm)
        )
        con.commit()
        marked = True
    except sqlite3.IntegrityError:
        marked = False
    con.close()
    return jsonify(ok=True, marked=marked, student=student, confidence=confidence,
                   message="Attendance marked." if marked else "Already marked today.")

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

@app.route("/api/attendance")
def attendance_api():
    con = db()
    rows = [dict(x) for x in con.execute("""
        SELECT a.id, s.name, s.roll_no, a.date, a.time
        FROM attendance a JOIN students s ON s.id=a.student_id
        ORDER BY a.date DESC, a.time DESC
    """).fetchall()]
    con.close()
    return jsonify(rows)

@app.route("/api/export")
def export_csv():
    con = db()
    rows = con.execute("""
        SELECT s.name, s.roll_no, a.date, a.time
        FROM attendance a JOIN students s ON s.id=a.student_id
        ORDER BY a.date DESC, a.time DESC
    """).fetchall()
    con.close()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Name","Roll No","Date","Time"])
    writer.writerows([[r["name"],r["roll_no"],r["date"],r["time"]] for r in rows])
    mem = io.BytesIO(out.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="attendance.csv")

if __name__ == "__main__":
    init_db()
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=False)
