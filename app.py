#!/usr/bin/env python3

import os
import io
import base64
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tempfile
import shutil

# Import our modules
from image_processing import extract_sudoku_from_image
from solver import solve_sudoku, is_valid_sudoku, copy_board

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
UPLOAD_FOLDER = 'uploads'
TEMPLATE_FOLDER = 'templates'

for folder in [UPLOAD_FOLDER, TEMPLATE_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Convert image to base64 string for web display"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract_sudoku():
    """Handle file upload and extract Sudoku grid (Step 1)"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'})

    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get base64 for display
        original_image_b64 = image_to_base64(file_path)
        
        # Extract Sudoku
        board = extract_sudoku_from_image(file_path)

        # Clean up
        try:
            os.remove(file_path)
        except:
            pass

        if board is None:
            return jsonify({
                'success': False, 
                'error': 'Could not detect Sudoku grid',
                'original_image': f"data:image/jpeg;base64,{original_image_b64}" if original_image_b64 else None
            })

        return jsonify({
            'success': True,
            'original_image': f"data:image/jpeg;base64,{original_image_b64}",
            'original_board': board
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/solve', methods=['POST'])
def solve_sudoku_api():
    """Solve the Sudoku grid (Step 2)"""
    try:
        data = request.json
        original_board = data.get('board')

        if not original_board:
            return jsonify({'success': False, 'error': 'No board data provided'})

        # Validate
        if not is_valid_sudoku(original_board):
            return jsonify({
                'success': False, 
                'error': 'Invalid Sudoku board detected',
                'is_valid': False
            })

        # Solve
        solution_board = copy_board(original_board)
        solved = solve_sudoku(solution_board)

        if not solved:
            return jsonify({
                'success': False, 
                'error': 'Could not solve puzzle',
                'solved_board': None
            })

        return jsonify({
            'success': True,
            'solved_board': solution_board
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'})


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error. Please try again.'})


if __name__ == '__main__':
    # Import time module for timestamps
    import time

    print("🚀 Starting Snap Sudoku Solver web application...")
    print("📁 Make sure to create a 'templates' folder and place the index.html file there")
    print("🔗 Access the application at: http://localhost:5000")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)