"""
attendance_manager.py
---------------------
Manages attendance CSV file.
- Creates attendance_YYYY-MM-DD.csv automatically
- Re-mark logic:
    * Same person within 5 minutes  → "Already marked", NO new row
    * Same person after 5 minutes   → NEW row added (re-entry), old row kept
- File is stored in /data/ folder, opens in Excel directly
"""

import os
import csv
import pandas as pd
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# How many minutes must pass before a re-entry row is written
REMARK_COOLDOWN_MINUTES = 5


def get_today_csv_path() -> str:
    today = date.today().strftime('%Y-%m-%d')
    return os.path.join(DATA_DIR, f'attendance_{today}.csv')


def ensure_csv_exists(csv_path: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Name', 'Student_ID', 'Department', 'Role',
                'Status', 'Time', 'Date', 'Confidence', 'Distance', 'Entry'
            ])
        logger.info(f"Created new attendance file: {csv_path}")


def _load_df(csv_path: str) -> pd.DataFrame:
    columns = ['Name', 'Student_ID', 'Department', 'Role',
               'Status', 'Time', 'Date', 'Confidence', 'Distance', 'Entry']
    try:
        df = pd.read_csv(csv_path)
        if 'Entry' not in df.columns:
            df['Entry'] = 1
        return df
    except Exception:
        return pd.DataFrame(columns=columns)


def _get_last_mark_time(df: pd.DataFrame, name: str):
    person_rows = df[df['Name'] == name]
    if person_rows.empty:
        return None
    today_str = date.today().strftime('%Y-%m-%d')
    latest = None
    for t in person_rows['Time']:
        try:
            dt = datetime.strptime(f"{today_str} {t}", '%Y-%m-%d %H:%M:%S')
            if latest is None or dt > latest:
                latest = dt
        except Exception:
            continue
    return latest


def mark_attendance(name: str, info: dict, confidence: int, distance: float) -> dict:
    """
    Mark attendance with 5-minute cooldown logic.

    - First time today           -> add row, Entry=1
    - Within 5 min of last mark  -> NO new row, return already_marked=True
    - After 5 min of last mark   -> add NEW row with next Entry number
    """
    csv_path = get_today_csv_path()
    ensure_csv_exists(csv_path)

    now      = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')

    df        = _load_df(csv_path)
    last_time = _get_last_mark_time(df, name)

    # ── First time today ──────────────────────────────────────
    if last_time is None:
        new_row = {
            'Name':       name,
            'Student_ID': info.get('id', '—'),
            'Department': info.get('department', '—'),
            'Role':       info.get('role', '—'),
            'Status':     'Present',
            'Time':       time_str,
            'Date':       date_str,
            'Confidence': f'{confidence}%',
            'Distance':   round(distance, 4),
            'Entry':      1
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        msg = f'{name} marked Present at {time_str}'
        logger.info(msg)
        return {'success': True, 'message': msg,
                'already_marked': False, 'cooldown_remaining': 0}

    # ── Within cooldown ───────────────────────────────────────
    elapsed_seconds  = (now - last_time).total_seconds()
    cooldown_seconds = REMARK_COOLDOWN_MINUTES * 60

    if elapsed_seconds < cooldown_seconds:
        remaining = int(cooldown_seconds - elapsed_seconds)
        mins = remaining // 60
        secs = remaining % 60
        msg = (f'Attendance already marked for {name}. '
               f'Next entry allowed in {mins}m {secs}s.')
        logger.info(msg)
        return {'success': True, 'message': msg,
                'already_marked': True, 'cooldown_remaining': remaining}

    # ── Cooldown passed — new re-entry row ────────────────────
    person_rows = df[df['Name'] == name]
    next_entry  = int(person_rows['Entry'].max()) + 1 if not person_rows.empty else 1

    new_row = {
        'Name':       name,
        'Student_ID': info.get('id', '—'),
        'Department': info.get('department', '—'),
        'Role':       info.get('role', '—'),
        'Status':     'Present',
        'Time':       time_str,
        'Date':       date_str,
        'Confidence': f'{confidence}%',
        'Distance':   round(distance, 4),
        'Entry':      next_entry
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    msg = f'{name} re-entered at {time_str} (Entry #{next_entry})'
    logger.info(msg)
    return {'success': True, 'message': msg,
            'already_marked': False, 'cooldown_remaining': 0}


def mark_unknown(image_time: str = None) -> dict:
    csv_path = get_today_csv_path()
    ensure_csv_exists(csv_path)
    now      = datetime.now()
    time_str = image_time or now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')

    df = _load_df(csv_path)
    new_row = {
        'Name': 'Unknown', 'Student_ID': '—', 'Department': '—',
        'Role': '—', 'Status': 'Unknown', 'Time': time_str,
        'Date': date_str, 'Confidence': '0%', 'Distance': '—', 'Entry': 1
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return {'success': True, 'message': f'Unknown face logged at {time_str}'}


def get_today_attendance() -> list:
    csv_path = get_today_csv_path()
    if not os.path.exists(csv_path):
        return []
    try:
        return pd.read_csv(csv_path).to_dict('records')
    except Exception:
        return []


def get_attendance_summary() -> dict:
    records       = get_today_attendance()
    present_names = set(r['Name'] for r in records if r.get('Status') == 'Present')
    unknown       = [r for r in records if r.get('Status') == 'Unknown']
    return {
        'total_present': len(present_names),
        'total_unknown': len(unknown),
        'total_scans':   len(records),
        'date':          date.today().strftime('%Y-%m-%d'),
        'csv_path':      get_today_csv_path()
    }


def get_all_attendance_files() -> list:
    os.makedirs(DATA_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR)
             if f.startswith('attendance_') and f.endswith('.csv')]
    files.sort(reverse=True)
    return files