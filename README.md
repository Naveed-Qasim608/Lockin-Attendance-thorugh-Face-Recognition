# 🔒 LockIn — AI Face Recognition Attendance System

A real-time biometric attendance system built with **Python**, **Flask**, and **face_recognition** library. Supports mobile camera via browser, handles multiple faces simultaneously, and saves attendance to CSV automatically.

---

## ✨ Features

- 📸 **Real-time face detection** using dlib HOG model
- 🧠 **128-D face embeddings** — same concept as FaceNet
- 📱 **Mobile camera support** — open on phone browser, no app needed
- 👥 **Multiple faces per frame** — entire class detected at once
- 📊 **Auto CSV attendance** — opens directly in Excel
- 🔐 **Strict matching** — siblings/relatives correctly rejected
- 🗄️ **Modular Python code** — easy to understand and extend

---

## 🗂️ Project Structure

```
lockin/
│
├── app.py                    # Flask server — main entry point
│
├── modules/
│   ├── face_encoder.py       # Enrollment: capture → 128-D embedding → save
│   ├── face_recognizer.py    # Recognition: compare embeddings, Euclidean distance
│   └── attendance_manager.py # CSV read/write, mark present/absent
│
├── templates/
│   ├── base.html             # Shared navbar and layout
│   ├── index.html            # Home page
│   ├── enroll.html           # Enrolment page
│   ├── attendance.html       # Mark attendance page
│   └── dashboard.html        # Records + CSV download
│
├── encodings/
│   └── face_db.pkl           # Stored face embeddings (auto-created)
│
├── data/
│   └── attendance_YYYY-MM-DD.csv   # Daily attendance files (auto-created)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `face_recognition` requires `cmake` and `dlib`.
> On Windows: install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) first.
> On Mac: `brew install cmake`

### Step 2 — Run the server

```bash
python app.py
```

### Step 3 — Open in browser

| Device | URL |
|--------|-----|
| Laptop | `http://localhost:5000` |
| Phone  | `http://<your-laptop-ip>:5000` |

> 📡 Phone and laptop must be on the **same WiFi network**.
> Find your IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)

---

## 🧠 How It Works (ML Concepts)

### Face Embedding (128-D Vector)
Each face is converted into a 128-dimensional vector using a deep CNN. Every dimension encodes a unique facial feature learned during training.

### Euclidean Distance Matching
```
distance = √( Σ (embedding_A[i] - embedding_B[i])² )  for i=0..127
```
If `distance < 0.42` → **MATCH** (known person)
If `distance ≥ 0.42` → **UNKNOWN** (not enrolled)

### Mean Embedding Storage
5 captures are taken during enrollment. The **mean** of all 5 embeddings is stored, making recognition robust to:
- Different lighting
- Slight pose changes
- Wearing/removing glasses

### Voting System (Anti-False-Positive)
During attendance, multiple frames are collected. Each frame votes independently. A person must win the **majority of votes** to be marked present — this prevents siblings or similar-looking people from being misidentified.

---

## 📱 Mobile Camera Usage

1. Run `python app.py` on laptop
2. Note the IP shown in terminal (e.g. `http://192.168.1.5:5000`)
3. Open that URL in **Safari** (iPhone) or **Chrome** (Android)
4. Go to **Mark Attendance** → select **Mobile Cam**
5. Allow camera permission
6. Point at faces — results appear on laptop dashboard

---

## 📊 CSV Attendance Format

Each day creates a new file: `data/attendance_2024-01-15.csv`

| Name | Student_ID | Department | Role | Status | Time | Date | Confidence | Distance |
|------|-----------|------------|------|--------|------|------|------------|----------|
| Ahmed Khan | F21-001 | CS-6A | Student | Present | 09:15:32 | 2024-01-15 | 87% | 0.2841 |
| Unknown | — | — | — | Unknown | 09:17:05 | 2024-01-15 | 0% | — |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.x + Flask |
| Face Detection | dlib HOG (via face_recognition) |
| Face Recognition | 128-D ResNet embeddings |
| Distance Metric | Euclidean (L2 norm) |
| Data Storage | CSV (pandas) + Pickle |
| Frontend | HTML + CSS + JavaScript |
| Mobile Support | Browser getUserMedia API |

---

## 👨‍💻 Team

Built as an AI/ML course project.

---

## 📄 License

MIT License
