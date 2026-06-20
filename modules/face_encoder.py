"""
face_encoder.py
---------------
Handles all face encoding and enrollment logic.
- Computes 128-D face embeddings using dlib/face_recognition
- Saves all 5 enrollment images permanently to /enrollments/<name>/
- Stores mean + all encodings in face_db.pkl
"""

import os
import pickle
import numpy as np
import face_recognition
from PIL import Image
import io
import base64
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

ENCODINGS_PATH   = os.path.join(os.path.dirname(__file__), '..', 'encodings', 'face_db.pkl')
ENROLLMENTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'enrollments')


# ── DB helpers ────────────────────────────────────────────────

def load_face_db() -> dict:
    """Load face database from disk."""
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_face_db(db: dict):
    """Save face database to disk."""
    os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(db, f)


# ── Image helpers ─────────────────────────────────────────────

def decode_image_from_base64(b64_string: str) -> np.ndarray:
    """Convert base64 image string from browser to numpy RGB array."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


def _save_enrollment_images(name: str, image_b64_list: list) -> str:
    """
    Save all captured enrollment images permanently to disk.

    Folder structure:
        enrollments/
            Ahmed_Khan/
                capture_1_2024-01-15_09-30-00.jpg
                capture_2_2024-01-15_09-30-01.jpg
                ...

    Returns the folder path where images were saved.
    """
    # Make folder name safe (replace spaces/special chars)
    safe_name   = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    person_dir  = os.path.join(ENROLLMENTS_DIR, safe_name)
    os.makedirs(person_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for i, b64 in enumerate(image_b64_list, start=1):
        try:
            if ',' in b64:
                b64 = b64.split(',')[1]
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Save as JPEG
            filename = f'capture_{i}_{timestamp}.jpg'
            filepath = os.path.join(person_dir, filename)
            img.save(filepath, 'JPEG', quality=90)
            logger.info(f"Saved enrollment image: {filepath}")
        except Exception as e:
            logger.warning(f"Could not save image {i} for {name}: {e}")

    return person_dir


# ── Face encoding ─────────────────────────────────────────────

def get_face_encodings_from_image(img_array: np.ndarray) -> list:
    """
    Detect all faces in image and return (encoding, bbox) tuples.
    Uses HOG model — fast on CPU.
    """
    face_locations = face_recognition.face_locations(img_array, model='hog')
    if not face_locations:
        return []
    encodings = face_recognition.face_encodings(img_array, face_locations)
    return list(zip(encodings, face_locations))


# ── Enrollment ────────────────────────────────────────────────

def enroll_person(name: str, student_id: str, department: str,
                  role: str, image_b64_list: list) -> dict:
    """
    Enroll a person in the face database.

    Steps:
    1. Decode each image and extract 128-D face encoding
    2. Compute mean encoding across all captures
    3. Save all images permanently to enrollments/<name>/
    4. Store encodings + info in face_db.pkl

    Args:
        name:            Full name
        student_id:      Student/Employee ID
        department:      Department or class
        role:            Student / Teacher / Staff / Admin
        image_b64_list:  List of base64-encoded face images (5 recommended)

    Returns:
        dict with success, message, captures, image_folder
    """
    db             = load_face_db()
    all_encodings  = []

    for i, b64 in enumerate(image_b64_list, start=1):
        try:
            img     = decode_image_from_base64(b64)
            results = get_face_encodings_from_image(img)
            if results:
                encoding, _ = results[0]
                all_encodings.append(encoding)
                logger.info(f"Encoded face {i}/{len(image_b64_list)} for {name}")
            else:
                logger.warning(f"No face detected in image {i} for {name}")
        except Exception as e:
            logger.warning(f"Could not process image {i}: {e}")
            continue

    if len(all_encodings) == 0:
        return {
            'success': False,
            'message': 'No faces detected in any image. Try again with better lighting.'
        }

    # Mean embedding for robust matching
    mean_encoding = np.mean(all_encodings, axis=0)

    # Save images to disk permanently
    image_folder = _save_enrollment_images(name, image_b64_list)

    # Store in DB
    db[name] = {
        'mean_encoding': mean_encoding,
        'all_encodings': all_encodings,
        'info': {
            'id':           student_id,
            'department':   department,
            'role':         role,
            'captures':     len(all_encodings),
            'enrolled_on':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_folder': image_folder
        }
    }

    save_face_db(db)
    logger.info(f"Enrolled {name} — {len(all_encodings)} captures, images at {image_folder}")

    return {
        'success':      True,
        'message':      f'{name} enrolled successfully with {len(all_encodings)} captures.',
        'captures':     len(all_encodings),
        'image_folder': image_folder
    }


def delete_person(name: str) -> dict:
    """Remove a person from face database (keeps images on disk)."""
    db = load_face_db()
    if name in db:
        del db[name]
        save_face_db(db)
        return {'success': True, 'message': f'{name} removed from database.'}
    return {'success': False, 'message': f'{name} not found in database.'}


def list_enrolled() -> list:
    """Return list of all enrolled persons with their info."""
    db = load_face_db()
    result = []
    for name, data in db.items():
        info = data.get('info', {})
        result.append({
            'name':         name,
            'id':           info.get('id', '—'),
            'department':   info.get('department', '—'),
            'role':         info.get('role', '—'),
            'captures':     info.get('captures', 0),
            'enrolled_on':  info.get('enrolled_on', '—'),
            'image_folder': info.get('image_folder', '—')
        })
    return result