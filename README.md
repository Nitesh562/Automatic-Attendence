# Automatic Attendance System Using Face Recognition

A ready-to-run BCA project using **Python + Flask + OpenCV LBPH + SQLite**. It works from a PC or a mobile browser on the same network.

## Features
- Student registration using webcam/mobile camera
- 15 face samples per registration
- LBPH face recognition
- Automatic attendance with date/time
- Prevents duplicate attendance on the same day
- Attendance table
- CSV export
- SQLite database
- Responsive interface for desktop and mobile

## Requirements
- Python 3.10–3.12 recommended
- Webcam/camera
- Chrome/Edge/Firefox
- For mobile: phone and computer should be on the same Wi-Fi network

## Windows installation

Open Command Prompt in this folder:

```text
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open:
`http://127.0.0.1:5000`

## Mobile use
Find the computer's local IP address with `ipconfig`, for example `192.168.1.10`.

On the phone, open:
`http://192.168.1.10:5000`

Allow camera permission.

If Windows Firewall asks, allow Python on the Private network.

### Camera security note
Modern browsers may restrict camera access on non-HTTPS network URLs. If your phone blocks the camera over the local IP, run the app on the computer and use Chrome/Edge on the computer, or deploy behind HTTPS for remote/mobile access.

## Usage
1. Open Register.
2. Enter student name and roll number.
3. Start camera and capture 15 samples.
4. Save registration. The recognition model is trained automatically.
5. Open Attendance.
6. The system scans the camera every 1.5 seconds.
7. A recognized student is marked once for that date.
8. Use Export CSV to download attendance.

## Project structure

```text
automatic_attendance_system/
  app.py
  requirements.txt
  README.md
  attendance.db          (created automatically)
  trainer.yml            (created after registration)
  dataset/               (created automatically)
  templates/
    index.html
    register.html
    attendance.html
  static/
    style.css
```

## Recognition threshold
The default LBPH threshold is 65 in `app.py`. Lower values are stricter; higher values are more permissive. Lighting, camera quality, pose, and number of samples affect recognition.

## Academic project note
This is a functional prototype suitable for a BCA final-year project. For production deployment, add authenticated admin accounts, HTTPS, encryption/access controls, consent and retention policies, liveness/anti-spoofing, audit logs, and stronger recognition models.
