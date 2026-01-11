# 🧩 Snap Sudoku Solver

An automated Sudoku solver that uses Computer Vision and Tesseract OCR to extract grids from images and solve them using an optimized bitmask-based backtracking algorithm.

![Sudoku Solver Banner](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-orange.svg)

## ✨ Features

- **📸 Image Extraction**: Robust grid detection using OpenCV (Contour detection, Perspective Transformation).
- **🔢 Multi-Strategy OCR**: Intelligent OCR logic with Tesseract that retries different preprocessing techniques (border reduction, PSM modes) to accurately identify digits.
- **⚡ Super-Fast Solver**: Backtracking algorithm optimized with Bitmasks and the Minimum Remaining Values (MRV) heuristic.
- **🌐 Web Interface**: Modern Flask-based web app for uploading images and viewing step-by-step extraction and solution.
- **🧬 Parallel Processing**: Uses `ThreadPoolExecutor` for concurrent digit extraction across all 81 cells.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Image Processing**: OpenCV (cv2), NumPy, PIL (Pillow)
- **OCR Engine**: Tesseract OCR
- **Optimization**: Bitmasking, MRV Heuristic

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR Engine**:
   - **Windows**: [Download Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and install.
   - **Linux**: `sudo apt install tesseract-ocr`
   - **Mac**: `brew install tesseract`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sudoku_solver.git
   cd sudoku_solver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Tesseract Path**:
   In `image_processing.py`, update the path to your Tesseract executable if it's not in your system PATH:
   ```python
   # Windows example
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### Running the Application

Start the Flask server:
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:5000`.

## 📂 Project Structure

- `app.py`: Flask web application and API endpoints.
- `image_processing.py`: Core CV logic (grid detection, perspective transform, OCR strategies).
- `solver.py`: Optimized Sudoku solving algorithm.
- `requirements.txt`: Python dependencies.
- `templates/`: HTML templates for the web interface.
- `static/`: CSS and JavaScript files.
- `uploads/`: Temporary storage for uploaded Sudoku images.

## 🔍 How it Works

1. **Preprocessing**: The image is grayscaled, blurred, and thresholded.
2. **Contour Detection**: Finds the largest 4-sided contour (the Sudoku grid).
3. **Perspective Wrap**: Corrects the angle of the Sudoku grid to a flat 2D square.
4. **Digit Extraction**: The grid is split into 81 cells. Each cell undergoes multiple OCR strategies to filter noise and identify digits.
5. **Solving**: The extracted board is validated and solved using a bitmask-based backtracking solver with MRV.

---
Developed by [bluekap](https://github.com/bluekap)
