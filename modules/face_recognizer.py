"""
face_recognizer.py
------------------
Core face recognition module.
- Compares unknown face encodings against the face database
- Uses Euclidean distance (L2 norm) — same as original project concept
- Implements VOTING across multiple encodings per person for accuracy
- Handles MULTIPLE faces in a single frame simultaneously
"""

import numpy as np
import face_recognition
import logging
from modules.face_encoder import load_face_db, decode_image_from_base64, get_face_encodings_from_image
from modules import cnn_recognizer

logger = logging.getLogger(__name__)

# ── TUNING PARAMETERS ──────────────────────────────────────────
# Lower = stricter matching. 0.45 is standard but we use 0.42 for
# better separation between siblings / similar-looking people.
RECOGNITION_THRESHOLD = 0.42

# For each unknown face, we compare against ALL stored encodings
# (not just the mean). A person is matched if they win by clear margin.
TOLERANCE = 0.42


def euclidean_distance(enc1: np.ndarray, enc2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 128-D face embeddings.
    This is the core ML comparison metric — same as in the project report.
    distance = sqrt( sum( (a_i - b_i)^2 ) )
    """
    return float(np.linalg.norm(enc1 - enc2))


def recognize_face(unknown_encoding: np.ndarray, db: dict) -> dict:
    """
    Compare one unknown face encoding against all persons in DB.

    Strategy:
    1. Compare against ALL stored encodings per person (not just mean)
    2. Compute minimum distance per person
    3. Winner = person with lowest distance
    4. If winner's distance < THRESHOLD → KNOWN, else → UNKNOWN

    Returns:
        dict with name, distance, confidence, is_known
    """
    if not db:
        return {'is_known': False, 'name': 'Unknown', 'distance': 1.0, 'confidence': 0, 'info': {}}

    best_name = None
    best_distance = float('inf')

    for name, data in db.items():
        # Compare against mean encoding first (fast)
        mean_enc = data['mean_encoding']
        mean_dist = euclidean_distance(unknown_encoding, mean_enc)

        # Also compare against ALL individual encodings (more robust)
        all_encs = data.get('all_encodings', [mean_enc])
        all_dists = [euclidean_distance(unknown_encoding, enc) for enc in all_encs]
        min_dist = min(all_dists)

        # Weighted combination: 40% mean, 60% best individual match
        combined = 0.4 * mean_dist + 0.6 * min_dist

        if combined < best_distance:
            best_distance = combined
            best_name = name

    is_known = best_distance < RECOGNITION_THRESHOLD

    # Confidence: how far below the threshold we are (0-100%)
    if is_known:
        confidence = max(0, min(100, int((1 - best_distance / RECOGNITION_THRESHOLD) * 100)))
    else:
        confidence = 0

    return {
        'is_known': is_known,
        'name': best_name if is_known else 'Unknown',
        'distance': round(best_distance, 4),
        'confidence': confidence,
        'info': db[best_name]['info'] if is_known and best_name else {}
    }


def _normalize_name(name: str) -> str:
    """Make names comparable regardless of underscore/space/casing differences
    (CNN classes come from Kaggle folder names like 'Abdullah_Tariq';
    local face_db keys are typed names like 'Abdullah Tariq')."""
    return name.replace('_', ' ').strip().lower()


def _find_db_key(name: str, db: dict):
    """Find the actual db key matching `name`, tolerant of underscore/space/case
    differences. Returns None if no match."""
    if not name:
        return None
    target = _normalize_name(name)
    for key in db.keys():
        if _normalize_name(key) == target:
            return key
    return None


def recognize_face_hybrid(face_crop_rgb: np.ndarray, unknown_encoding: np.ndarray, db: dict) -> dict:
    """
    HYBRID strategy (transfer-learning CNN first, dlib embeddings as fallback):

    1. Run the trained ResNet34 (transfer learning) classifier on the cropped face.
    2. If it's confident (>= CNN_THRESHOLD) AND the predicted person is enrolled
       locally (so we have their info/id/department), trust it.
    3. Otherwise fall back to the existing dlib 128-D distance matcher — this is
       what makes newly-enrolled people (never seen by the CNN) still work.

    Returns the same dict shape as recognize_face(), plus a 'method' field.
    """
    cnn_name, cnn_conf = cnn_recognizer.cnn_predict(face_crop_rgb)
    db_key = _find_db_key(cnn_name, db)

    if cnn_name is not None and cnn_conf >= cnn_recognizer.CNN_THRESHOLD and db_key is not None:
        return {
            'is_known':   True,
            'name':       db_key,   # use the canonical name as stored locally (consistent with attendance records)
            'distance':   round(1 - cnn_conf, 4),   # kept for schema consistency (lower = better)
            'confidence': int(round(cnn_conf * 100)),
            'info':       db[db_key]['info'],
            'method':     'CNN-TransferLearning(ResNet34)'
        }

    result = recognize_face(unknown_encoding, db)
    result['method'] = 'dlib-128D'
    return result


def recognize_all_faces_in_image(image_b64: str) -> list:
    """
    Main recognition function — handles MULTIPLE faces in one frame.

    Steps:
    1. Decode image from base64 (sent by mobile camera)
    2. Detect ALL face locations using HOG model
    3. Compute 128-D embedding for each face
    4. Match each embedding against DB
    5. Return results for ALL faces simultaneously

    Args:
        image_b64: Base64-encoded image from browser camera

    Returns:
        List of dicts, one per detected face:
        [
          {
            'name': 'Ahmed Khan',
            'is_known': True,
            'distance': 0.31,
            'confidence': 82,
            'bbox': [top, right, bottom, left],
            'info': { 'id': '...', 'department': '...', 'role': '...' }
          },
          ...
        ]
    """
    try:
        img_array = decode_image_from_base64(image_b64)
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return []

    # Detect face locations — 'hog' is CPU-fast, 'cnn' is more accurate (needs GPU)
    face_locations = face_recognition.face_locations(img_array, model='hog')

    if not face_locations:
        return []

    # Get 128-D encodings for all detected faces
    face_encodings = face_recognition.face_encodings(img_array, face_locations)

    # Load DB once (efficient)
    db = load_face_db()

    results = []
    for encoding, location in zip(face_encodings, face_locations):
        face_crop = cnn_recognizer.crop_face(img_array, location)
        match = recognize_face_hybrid(face_crop, encoding, db)
        match['bbox'] = list(location)  # [top, right, bottom, left]
        results.append(match)
        logger.info(
            f"Face: {match['name']} | method={match.get('method')} | "
            f"dist={match['distance']} | conf={match['confidence']}%"
        )

    return results
