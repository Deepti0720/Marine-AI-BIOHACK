# 🌊 Marine AI — Microscopic Plankton Detection & Taxonomy Platform

> **An AI-powered platform for detecting, localizing, and classifying microscopic marine organisms from microscopy images.**

**Developed by Team BIOHACK**

Marine AI is an intelligent computer-vision web application designed to automate the **identification, bounding-box localization, and taxonomy analysis** of microscopic marine organisms.

Powered by **YOLO-based deep learning models**, interactive visualization, and a **hash-based spatial taxonomy system**, Marine AI enables marine biologists and oceanographers to efficiently analyze dense microscopy samples.

---

## ✨ Features

### 🔬 Dense Object Detection

Marine AI uses an optimized **YOLO object-detection model** to detect microscopic marine organisms in complex microscopy images.

Detection is configured with:

* **Confidence threshold:** `0.10`
* **IoU threshold:** `0.45`
* Bounding-box localization
* Detection of clustered organisms
* Detection of low-contrast and faint organisms
* Real-time inference on uploaded microscopy images

---

### 🖼️ Interactive Sample Gallery

The application includes a pre-loaded microscopy sample gallery for quick testing.

Users can:

* Browse multiple sample images
* Select an image with a single click
* Run AI detection instantly
* Upload custom microscopy images
* View detection results interactively

Sample images are provided in the:

```text
sample_images/
```

directory.

---

### 🔑 Visual Hash Map Taxonomy

Marine AI includes a visual **hash-map-based taxonomy system** for organizing detected species.

The system uses:

```text
hash(species) % 6
```

to determine the bucket assignment.

The implementation demonstrates:

* Hash-based species indexing
* Collision handling using **chaining**
* Visual bucket representation
* Fast average-case lookup
* Taxonomy-to-bucket mapping

The hash-map pipeline provides an educational visualization of how spatial data structures can be integrated into an AI-based classification workflow.

---

### 📊 Interactive Data Analytics

Marine AI provides interactive analytics using **Plotly**.

The dashboard visualizes:

* Species distribution
* Detection counts
* Taxonomy breakdown
* Organism frequency
* Class-wise detection statistics

The analytics section makes it easier to understand the composition of each microscopy sample.

---

### 📄 Automated Report Export

Detection and taxonomy results can be exported for further analysis.

Supported formats include:

* **CSV** — structured detection and taxonomy data
* **PDF** — formatted analytical report summary

This allows researchers to save and share analysis results.

---

# 🦠 Target Taxonomy Classes

Marine AI currently supports **10 marine organism classes**:

| Class ID | Organism           |
| :------: | ------------------ |
|    `0`   | **Calanoida**      |
|    `1`   | **Cyclopoida**     |
|    `2`   | **Dinoflagellate** |
|    `3`   | **Radiolarian**    |
|    `4`   | **Foraminifera**   |
|    `5`   | **Diatom**         |
|    `6`   | **Chaetoceros**    |
|    `7`   | **Ceratium**       |
|    `8`   | **Tintinnid**      |
|    `9`   | **Polychaete**     |

---

# 🏗️ Project Architecture

```text
marine-ai/
│
├── app.py
│   └── Main Streamlit dashboard application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── sample_images/
│   ├── 01_easy.png
│   ├── 02_easy.png
│   ├── ...
│   └── 15_mixed.png
│
└── models/
    └── marine_plankton_model.pt
        └── Trained YOLO model weights
```

---

# ⚙️ Technology Stack

| Technology           | Purpose                        |
| -------------------- | ------------------------------ |
| **Python**           | Core programming language      |
| **Streamlit**        | Interactive web dashboard      |
| **Ultralytics YOLO** | Object detection               |
| **OpenCV**           | Image processing               |
| **Pillow**           | Image handling                 |
| **NumPy**            | Numerical computation          |
| **Pandas**           | Data processing                |
| **Plotly**           | Interactive data visualization |

---

# 🚀 Getting Started

## Prerequisites

Make sure the following are installed:

* **Python 3.9+**
* **Git**
* Internet connection for installing Python dependencies

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/marine-ai.git
cd marine-ai
```

> Replace `YOUR_USERNAME` with your GitHub username.

---

## 2. Create a Virtual Environment

Using a virtual environment is recommended to keep project dependencies isolated.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## 4. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

After starting the application, open:

```text
http://localhost:8501
```

in your browser.

---

# 📦 Requirements

The project uses the following core Python packages:

```text
streamlit
ultralytics
opencv-python-headless
pillow
numpy
pandas
plotly
```

These dependencies are also listed in:

```text
requirements.txt
```

---

# 🧠 How Marine AI Works

The general processing pipeline is:

```text
                 ┌─────────────────────┐
                 │   Microscopy Image   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   YOLO Detection    │
                 │  conf = 0.10        │
                 │  IoU  = 0.45        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Bounding Boxes +    │
                 │ Class Predictions   │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │ Taxonomy Mapping │        │ Detection Counts │
    └────────┬─────────┘        └────────┬─────────┘
             │                           │
             ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │ Visual Hash Map  │        │ Plotly Analytics │
    └────────┬─────────┘        └────────┬─────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
                 ┌─────────────────────┐
                 │ CSV / PDF Reports   │
                 └─────────────────────┘
```

---

# 🔑 Hash Map Implementation

The taxonomy visualization uses a hash-based indexing approach.

Each detected species is assigned to a bucket using:

```python
bucket = hash(species) % 6
```

The system uses **collision chaining** when multiple species are assigned to the same bucket.

### Example

```text
Bucket 0 → [Species A]
Bucket 1 → [Species B, Species F]
Bucket 2 → [Species C]
Bucket 3 → [Species D]
Bucket 4 → [Species E]
Bucket 5 → [Species G]
```

This provides an intuitive visual demonstration of hash-table organization and average **O(1)** lookup behavior.

> **Note:** The O(1) complexity refers to average-case hash-table lookup, not the entire AI inference pipeline.

---

# 📊 Sample Analysis

Marine AI can process microscopy samples containing multiple organisms in the same frame.

The dashboard provides:

* Original microscopy image
* Detected bounding boxes
* Organism labels
* Confidence scores
* Total detections
* Species-wise counts
* Taxonomy distribution
* Hash-map bucket visualization
* Exportable analysis reports

---

# ☁️ Deploy on Streamlit Community Cloud

Marine AI can be deployed online using **Streamlit Community Cloud**.

### Step 1 — Push the Project to GitHub

Create a GitHub repository and push the project:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/marine-ai.git
git push -u origin main
```

### Step 2 — Deploy

1. Open **Streamlit Community Cloud**
2. Sign in using your GitHub account
3. Select **New App**
4. Choose your `marine-ai` repository
5. Select the `main` branch
6. Set the main file to:

```text
app.py
```

7. Click **Deploy**

Once deployment is complete, Streamlit will provide a public URL for the application.

---

# 📁 Important Files

| File / Directory           | Description                         |
| -------------------------- | ----------------------------------- |
| `app.py`                   | Main Streamlit application          |
| `models/`                  | Contains trained YOLO model weights |
| `marine_plankton_model.pt` | Marine plankton detection model     |
| `sample_images/`           | Sample microscopy images            |
| `requirements.txt`         | Python dependencies                 |
| `README.md`                | Project documentation               |

---

# 🎯 Project Goals

Marine AI aims to demonstrate how **artificial intelligence, computer vision, data structures, and interactive visualization** can be combined to assist marine research.

The project focuses on:

* 🤖 Automated plankton detection
* 🔬 Microscopy image analysis
* 🧬 Taxonomic classification
* 📦 Bounding-box localization
* 🗂️ Hash-based taxonomy organization
* 📊 Data-driven visualization
* 📄 Automated scientific reporting

---

# 🔮 Future Improvements

Potential future enhancements include:

* 🌊 Support for larger marine taxonomy datasets
* 🧠 Improved model accuracy through additional training data
* 🔬 Fine-grained species-level classification
* 📹 Real-time microscopy video detection
* ☁️ Cloud-based inference
* 🗺️ Geographic distribution mapping
* 📈 Historical sample comparison
* 🧪 Integration with marine research datasets
* 👥 Multi-user research dashboards
* 📱 Mobile-friendly interface

---

# 👨‍💻 Team

### Team BIOHACK

**Marine AI — Microscopic Plankton Detection & Taxonomy Platform**

Built with:

**Python • YOLO • Streamlit • OpenCV • Plotly • Pandas**

---

# 📜 License

This project is distributed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for more information.

---

## ⭐ Support the Project

If you find **Marine AI** useful or interesting:

* ⭐ Star this repository
* 🍴 Fork the project
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute to the project

---

> 🌊 **Marine AI — Using Artificial Intelligence to explore the microscopic world of our oceans.**
