# 🏏 Cricket Analyzer

A deep learning + computer vision based **cricket shot analyzer**.  
This project extracts pose keypoints from batting videos using **MediaPipe**, trains a **Temporal Convolutional Network (TCN)** to learn motion patterns, and then compares user-uploaded shots against reference "perfect" shots using **Dynamic Time Warping (DTW)**.  

A **Streamlit app** is included to visualize and interact with the system.

---

## 🚀 Features
- Pose keypoint extraction from cricket videos using **MediaPipe Holistic**.
- Sequence alignment and comparison with **DTW**.
- Trainable **Conv1D / TCN models** for sequence learning.
- Side-by-side video display of user vs reference shot.
- Score + suggestions generated automatically.
- Clean Streamlit interface with futuristic UI.

---

## 📂 Project Structure
cricket_analyzer_project/
│
├── data/ # (ignored) raw + reference videos, extracted npz
│ ├── perfect_shots/ # reference shots (e.g., cover_drive.mp4)
│ └── raw_videos/ # user input videos
│
├── models/ # (ignored) trained models (h5/keras)
│
├── src/
│ ├── app.py # Streamlit app
│ ├── analysis.py # DTW + feedback logic
│ ├── pose_extraction.py # keypoint extraction
│ ├── model_training.py # training pipeline
│ └── utils/ # helper functions
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore


> **Note:** `data/` and `models/` are git-ignored to avoid pushing large files.  
> Store videos/models externally (Google Drive, S3, etc.) and document their paths.

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
2. Create and activate a virtual environment
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Prepare your videos

Organize data like this:

data/
 ├── perfect_shots/
 │    ├── cover_drive.mp4
 │    ├── pull_shot.mp4
 │    └── ...
 └── raw_videos/
      ├── cover_drive/
      │    └── user1.mp4
      └── pull_shot/
           └── user2.mp4

🏋️ Workflow
Step 1 — Extract keypoints
python src/pose_extraction.py


This creates .npz keypoint files in data/.

Step 2 — Train model
python src/model_training.py


This trains a simple Conv1D/TCN model and saves it in models/.

Step 3 — Run the app
streamlit run src/app.py


Upload your batting video and choose a reference shot.
The app will:

Show side-by-side videos (your shot vs reference).

Display similarity score.

Suggest improvements based on joint alignment.

📊 Example Output

Similarity Score: 78%

Suggestions:

Backlift angle slightly off.

Footwork slower than reference.

Follow-through is consistent.

⚠️ Notes

Don’t push heavy .mp4 or .h5 files to GitHub directly.

Use Git LFS
 or external storage.

Adjust hyperparameters in model_training.py for better accuracy.

🧑‍💻 Tech Stack

Python 3.10+

MediaPipe

NumPy / Pandas / Scikit-learn

TensorFlow / Keras

Streamlit

📌 Future Work

Improve model accuracy by training with a larger dataset.

Support bowler action analysis.

Add 3D pose estimation for more precise feedback.

Build mobile-friendly version.

🙌 Acknowledgements

Google MediaPipe
 for pose estimation.

Dynamic Time Warping
 for sequence alignment.

Inspiration from cricket coaching techniques.
