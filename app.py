"""
app.py
------
LockIn — Flask Web Server
Main entry point. Connects frontend (browser/mobile) to Python ML backend.

Run:
    python app.py

Then open on:
    Laptop  : http://localhost:5000
    Mobile  : http://<your-laptop-ip>:5000
              (Find IP: run 'ipconfig' on Windows or 'ifconfig' on Mac/Linux)
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Try to import ngrok for automatic tunneling
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

from modules.face_encoder import enroll_person, list_enrolled, delete_person
from modules.face_recognizer import recognize_all_faces_in_image
from modules.attendance_manager import (
    mark_attendance, mark_unknown,
    get_today_attendance, get_attendance_summary,
    get_all_attendance_files
)

# ── LOGGING SETUP ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ── FLASK APP SETUP ────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow mobile browser to send requests to laptop server

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


# ══════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/enroll')
def enroll_page():
    """Enrollment page"""
    return render_template('enroll.html')


@app.route('/attendance')
def attendance_page():
    """Mark attendance page"""
    return render_template('attendance.html')


@app.route('/dashboard')
def dashboard_page():
    """Attendance dashboard / records page"""
    return render_template('dashboard.html')


# ══════════════════════════════════════════════════════════════
# API — ENROLLMENT
# ══════════════════════════════════════════════════════════════

@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """
    Enroll a new person.
    Expects JSON: { name, student_id, department, role, images: [base64, ...] }
    """
    data = request.get_json()

    name       = data.get('name', '').strip()
    student_id = data.get('student_id', '').strip()
    department = data.get('department', '').strip()
    role       = data.get('role', 'Student').strip()
    images     = data.get('images', [])

    if not name:
        return jsonify({'success': False, 'message': 'Name is required.'}), 400
    if not images:
        return jsonify({'success': False, 'message': 'No images provided.'}), 400
    if len(images) < 3:
        return jsonify({'success': False, 'message': 'At least 3 face images required.'}), 400

    result = enroll_person(name, student_id, department, role, images)
    status_code = 200 if result['success'] else 400
    return jsonify(result), status_code


@app.route('/api/enrolled', methods=['GET'])
def api_enrolled():
    """Get list of all enrolled persons."""
    people = list_enrolled()
    return jsonify({'success': True, 'data': people, 'count': len(people)})


@app.route('/api/delete/<name>', methods=['DELETE'])
def api_delete(name):
    """Delete a person from face database."""
    result = delete_person(name)
    return jsonify(result)


# ══════════════════════════════════════════════════════════════
# API — ATTENDANCE / RECOGNITION
# ══════════════════════════════════════════════════════════════

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """
    Recognize face(s) in an image and mark attendance.
    Handles MULTIPLE faces in one frame simultaneously.

    Expects JSON: { image: '<base64 string>' }
    Returns: { faces: [ { name, is_known, confidence, distance, bbox, status } ] }
    """
    data = request.get_json()
    image_b64 = data.get('image', '')

    if not image_b64:
        return jsonify({'success': False, 'message': 'No image provided.'}), 400

    # Recognize all faces in frame
    faces = recognize_all_faces_in_image(image_b64)

    if not faces:
        return jsonify({
            'success': True,
            'faces': [],
            'message': 'No face detected in frame.'
        })

    # Mark attendance for each detected face
    results = []
    for face in faces:
        if face['is_known']:
            att_result = mark_attendance(
                face['name'],
                face['info'],
                face['confidence'],
                face['distance']
            )
            face['attendance_status'] = att_result['message']
            face['already_marked'] = att_result.get('already_marked', False)
        else:
            mark_unknown()
            face['attendance_status'] = 'Unknown — not enrolled'
            face['already_marked'] = False
        results.append(face)

    return jsonify({
        'success': True,
        'faces': results,
        'count': len(results)
    })


# ══════════════════════════════════════════════════════════════
# API — ATTENDANCE RECORDS
# ══════════════════════════════════════════════════════════════

@app.route('/api/attendance/today', methods=['GET'])
def api_today_attendance():
    """Get today's attendance records."""
    records = get_today_attendance()
    summary = get_attendance_summary()
    return jsonify({'success': True, 'records': records, 'summary': summary})


@app.route('/api/attendance/files', methods=['GET'])
def api_attendance_files():
    """List all attendance CSV files."""
    files = get_all_attendance_files()
    return jsonify({'success': True, 'files': files})


@app.route('/api/attendance/download/<filename>', methods=['GET'])
def api_download_csv(filename):
    """Download a specific attendance CSV file."""
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


# ══════════════════════════════════════════════════════════════
# API — ENROLLMENT IMAGES
# ══════════════════════════════════════════════════════════════

@app.route('/api/enrollment-images/<name>', methods=['GET'])
def api_enrollment_images(name):
    """Return saved enrollment image filenames for a person."""
    import urllib.parse
    safe_name  = urllib.parse.unquote(name).replace(' ', '_')
    person_dir = os.path.join(os.path.dirname(__file__), 'enrollments', safe_name)
    if not os.path.exists(person_dir):
        return jsonify({'success': True, 'images': [], 'count': 0})
    images = sorted([
        f for f in os.listdir(person_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    return jsonify({'success': True, 'images': images,
                    'count': len(images), 'folder': person_dir})


@app.route('/api/enrollment-images/<name>/<filename>', methods=['GET'])
def api_serve_enrollment_image(name, filename):
    """Serve a specific enrollment image."""
    import urllib.parse
    safe_name  = urllib.parse.unquote(name).replace(' ', '_')
    person_dir = os.path.join(os.path.dirname(__file__), 'enrollments', safe_name)
    return send_from_directory(person_dir, filename)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import socket

    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('encodings', exist_ok=True)
    os.makedirs('enrollments', exist_ok=True)

    # Get local IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = '0.0.0.0'

    # ── SSL SETUP ────────────────────────────────────────────
    # Browsers block camera on plain HTTP for non-localhost URLs.
    # We use HTTPS with an adhoc cert so mobile camera works.

    use_ssl = True
    protocol = 'https'
    ssl_context = 'adhoc'

    # ── NGROK TUNNEL ─────────────────────────────────────────
    # ENABLED FOR FASTER TUNNELING
    public_url = None
    if NGROK_AVAILABLE:
        try:
            # Disconnect all existing tunnels first
            for tunnel in ngrok.get_tunnels():
                ngrok.disconnect(tunnel.public_url)
            public_url = ngrok.connect(5000, "http").public_url
            print("  ✓ ngrok tunnel established")
            
            # If ngrok provides HTTPS, we don't need local SSL
            use_ssl = False
            protocol = 'http'
            ssl_context = None
            
        except Exception as e:
            print(f"  ⚠ ngrok error: {e}")

    print("\n" + "="*60)
    print("  🔒  LockIn — AI Face Recognition Attendance")
    print("="*60)
    
    if public_url:
        url_str = str(public_url)
        print(f"\n  🌐 PUBLIC URL (Safari, iPhone, any browser):")
        print(f"     {url_str}")
        print(f"\n  📱 Open on Safari/browser at:")
        print(f"     {url_str}/attendance")
        print(f"     OR")
        print(f"     {url_str}/enroll")
    
    if use_ssl:
        print(f"\n  ✓ SSL enabled — camera will work on mobile!")
        print(f"  Laptop  →  https://localhost:5000")
        print(f"  Mobile  →  https://{local_ip}:5000")
    else:
        if not public_url:
            print(f"\n  ⚠  SSL not enabled — mobile camera WON'T work!")
            print(f"  Laptop only  →  http://localhost:5000")
    
    print("="*60)
    print("  Phone must be on SAME WiFi as laptop.")
    print("="*60 + "\n")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        ssl_context=ssl_context
    )