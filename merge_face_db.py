"""
merge_face_db.py
-----------------
Merges a partner's encodings/face_db.pkl into your local one.

Usage:
    python merge_face_db.py partner_face_db.pkl

What it does:
- Loads your local encodings/face_db.pkl
- Loads the partner's pkl you point it at
- Adds any students from the partner's file that you don't already have
- If a name exists in BOTH (by normalized comparison: underscore/space/case
  insensitive), it asks you what to do instead of silently overwriting
- Backs up your original face_db.pkl before writing anything
- Saves the merged result back to encodings/face_db.pkl

Run this from inside your lockin/ project folder.
"""

import sys
import os
import pickle
import shutil
from datetime import datetime

LOCAL_DB_PATH = os.path.join('encodings', 'face_db.pkl')


def normalize(name: str) -> str:
    return name.replace('_', ' ').strip().lower()


def main():
    if len(sys.argv) != 2:
        print("Usage: python merge_face_db.py <path_to_partner_face_db.pkl>")
        sys.exit(1)

    partner_path = sys.argv[1]

    if not os.path.exists(LOCAL_DB_PATH):
        print(f"ERROR: {LOCAL_DB_PATH} not found. Run this from inside your lockin/ folder.")
        sys.exit(1)

    if not os.path.exists(partner_path):
        print(f"ERROR: {partner_path} not found.")
        sys.exit(1)

    with open(LOCAL_DB_PATH, 'rb') as f:
        local_db = pickle.load(f)

    with open(partner_path, 'rb') as f:
        partner_db = pickle.load(f)

    print(f"Local DB has {len(local_db)} students: {list(local_db.keys())}")
    print(f"Partner DB has {len(partner_db)} students: {list(partner_db.keys())}")
    print()

    local_normalized = {normalize(k): k for k in local_db.keys()}

    added, skipped, conflicts = [], [], []

    for name, record in partner_db.items():
        norm = normalize(name)
        if norm in local_normalized:
            existing_key = local_normalized[norm]
            print(f"CONFLICT: '{name}' (partner) looks like the same person as "
                  f"'{existing_key}' (already in your DB).")
            choice = input(
                "  [k]eep mine / [o]verwrite with partner's / [s]kip this person: "
            ).strip().lower()
            if choice == 'o':
                local_db[existing_key] = record
                conflicts.append((name, 'overwritten'))
            else:
                conflicts.append((name, 'kept local'))
        else:
            local_db[name] = record
            added.append(name)

    if not added and not conflicts:
        print("Nothing to merge — partner DB had no new students.")
        return

    # Backup before writing
    backup_path = LOCAL_DB_PATH + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(LOCAL_DB_PATH, backup_path)
    print(f"\nBacked up original DB to: {backup_path}")

    with open(LOCAL_DB_PATH, 'wb') as f:
        pickle.dump(local_db, f)

    print(f"\nDone. {LOCAL_DB_PATH} now has {len(local_db)} students.")
    if added:
        print(f"Added: {added}")
    if conflicts:
        print(f"Conflicts resolved: {conflicts}")


if __name__ == '__main__':
    main()
