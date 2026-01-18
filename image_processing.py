import cv2
import numpy as np
import sys

# Monkey patch zoneinfo for Python < 3.9
try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo
    sys.modules['zoneinfo'] = zoneinfo

import sys

# Monkey patch zoneinfo for Python < 3.9
try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo
    sys.modules['zoneinfo'] = zoneinfo

from paddleocr import PaddleOCR
import logging

# Initialize PaddleOCR globally
# use_angle_cls=False for faster performance since cells are already upright
# lang='en' for digits/english
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

# Suppress PaddleOCR logging
logging.getLogger('ppocr').setLevel(logging.ERROR)


# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac

def preprocess_image(image_path):
    """Load and preprocess the image for better grid detection"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    
    # Resize for consistency - normalize to max dimension of 1000px
    max_dim = 1000
    height, width = img.shape[:2]
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Apply adaptive threshold to get binary image with optimized parameters
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    return img, gray, thresh


def find_sudoku_grid(thresh_img, debug=False):
    """Find the largest rectangular contour (should be the Sudoku grid)"""
    # Find contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    grid_contour = None
    
    if debug:
        print(f"Found {len(contours)} contours total")

    # Look for the largest rectangular contour
    for idx, contour in enumerate(contours[:10]):  # Check top 10 largest
        area = cv2.contourArea(contour)
        if area < 10000:  # Too small to be the main grid
            if debug and idx < 5:
                print(f"Contour {idx}: area={area:.0f} - too small")
            continue

        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        epsilon = 0.02 * peri
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if debug and idx < 5:
            print(f"Contour {idx}: area={area:.0f}, perimeter={peri:.0f}, sides={len(approx)}")

        # If we found a 4-sided contour
        if len(approx) == 4:
            # Check if it is roughly square-ish to avoid long strips
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            if debug:
                print(f"  -> 4-sided! Bounding box: {w}x{h}, aspect ratio: {aspect_ratio:.2f}")
            
            # Slightly relaxed aspect ratio for skewed images
            if 0.7 <= aspect_ratio <= 1.3:
                grid_contour = approx
                if debug:
                    print(f"  ✓ Grid found! Area: {area:.0f}")
                break
            elif debug:
                print(f"  ✗ Aspect ratio out of range")

    return grid_contour


def order_points(pts):
    """Order points in the order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def perspective_transform(image, grid_contour):
    """Apply perspective transformation to get a top-down view of the grid"""
    # Order the points
    pts = grid_contour.reshape(4, 2)
    rect = order_points(pts)

    # Determine the width and height of the new image
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Set destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def extract_cells(warped_image):
    """Extract individual cells from the warped Sudoku grid"""
    cells = []
    height, width = warped_image.shape[:2]

    cell_height = height // 9
    cell_width = width // 9

    for i in range(9):
        row_cells = []
        for j in range(9):
            # Calculate cell boundaries
            y1 = i * cell_height
            y2 = (i + 1) * cell_height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width

            # Increase padding to avoid capturing grid lines which cause false detections
            padding = 5  # Increased from 2 to avoid grid line artifacts
            cell = warped_image[y1 + padding:y2 - padding, x1 + padding:x2 - padding]
            row_cells.append(cell)

        cells.append(row_cells)

    return cells


def preprocess_cell_for_ocr(cell):
    """Preprocess individual cell for better OCR results"""
    if cell.size == 0:
        return cell

    # Use a larger target size for better recognition
    target_size = 100
    
    # Resize with INTER_CUBIC for better quality
    cell = cv2.resize(cell, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    # Light Gaussian blur to reduce noise
    cell = cv2.GaussianBlur(cell, (3, 3), 0)

    # Apply Otsu's thresholding to get binary image
    # THRESH_BINARY_INV: white text on black background
    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clear borders more effectively to remove grid line artifacts
    border = 10
    cell[0:border, :] = 0
    cell[target_size-border:target_size, :] = 0
    cell[:, 0:border] = 0
    cell[:, target_size-border:target_size] = 0
    
    # Find contours to identify the digit
    contours, _ = cv2.findContours(cell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return empty (white) cell
    if not contours:
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    # Filter contours to find digit-like shapes
    valid_contours = []
    potential_one = None  # Special handling for digit "1"
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # Filter out very small noise and very large blobs
        if area < 30 or area > 6000:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        
        # Check if this could be a "1" (tall and thin)
        # "1" is typically 2-4x taller than it is wide
        if h > 2 * w and area > 100:
            potential_one = c
        
        # Reject very wide flat shapes (likely horizontal lines)
        if w > 3 * h:
            continue
            
        # Reject very tall thin shapes (likely vertical grid lines)
        # Relaxed to allow "1" digits through (which are naturally thin)
        if h > 85 and w < 5:
            continue
            
        valid_contours.append(c)

    # If no valid contours but we have a potential "1", use it
    if not valid_contours and potential_one is not None:
        valid_contours = [potential_one]

    if not valid_contours:
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    # Get bounding box of all valid contours combined
    x_min, y_min = target_size, target_size
    x_max, y_max = 0, 0
    
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Extract the digit region
    digit_w = x_max - x_min
    digit_h = y_max - y_min
    digit = cell[y_min:y_max, x_min:x_max]

    # Create centered image with padding
    centered = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Add 15% padding around digit
    padding_ratio = 0.15
    max_dim = int(target_size * (1 - 2 * padding_ratio))
    
    # Scale digit to fit while preserving aspect ratio
    scale = min(max_dim / digit_w, max_dim / digit_h) if digit_w > 0 and digit_h > 0 else 1
    new_w = int(digit_w * scale)
    new_h = int(digit_h * scale)
    
    if new_w > 0 and new_h > 0:
        # Resize digit with good interpolation
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Center the digit
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        
        centered[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    # Invert to get black text on white background (standard for OCR)
    centered = cv2.bitwise_not(centered)

    return centered


def extract_digit_from_cell(cell, row_idx=None, col_idx=None):
    """Extract digit from a single cell using PaddleOCR with optimized settings"""
    
    # First check if the RAW cell has meaningful content by counting black pixels
    if cell.size == 0:
        return 0
    
    # Fast empty cell check - avoid expensive preprocessing if clearly empty
    # Simple threshold on raw cell
    _, quick_thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    quick_black = np.sum(quick_thresh > 128)
    
    # If very few black pixels, skip all processing immediately
    if quick_black < 30:  # Very low threshold for speed
        return 0
    
    # Only do detailed check if we passed the quick test
    if quick_black < 100:  # Borderline cases
        blurred = cv2.GaussianBlur(cell, (3, 3), 0)
        _, raw_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        raw_thresh = cv2.morphologyEx(raw_thresh, cv2.MORPH_OPEN, kernel)
        raw_black_pixels = np.sum(raw_thresh > 128)
        
        if raw_black_pixels < 50:
            return 0

    # Preprocess the cell
    processed_cell = preprocess_cell_for_ocr(cell)
    
    # Ensure we have a valid image for Paddle
    if processed_cell is None or processed_cell.size == 0:
        return 0

    try:
        # PaddleOCR expects black text on white background (or vice versa, it's robust), 
        # but our processed_cell is white text on black background (inverted).
        # We should probably invert it back to standard "black text on white paper" for best OCR results.
        ocr_input = cv2.bitwise_not(processed_cell)
        
        # Optimize: det=False skips the detection model (DB/EAST) and only runs recognition (CRNN)
        # result format with det=False is [('text', conf), ...]
        result = ocr.ocr(ocr_input, cls=False, det=False)
        
        if not result or result[0] is None:
            return 0
            
        # With det=False, result[0] is often [[text, conf]] or [text, conf]
        res = result[0]
        text = ""
        confidence = 0.0
        
        if isinstance(res, (list, tuple)) and len(res) > 0:
            if isinstance(res[0], (list, tuple)):
                text = str(res[0][0])
                confidence = float(res[0][1])
            else:
                text = str(res[0])
                if len(res) > 1:
                    confidence = float(res[1])
        
        # Filter for digits only
        digits = "".join([c for c in text if c.isdigit()])
        
        # Only accept if we have digits and confidence is reasonably high
        # False detections of '1' usually have lower confidence or come from noise
        if digits and confidence > 0.35:  # Lowered from 0.4 for speed
            # If it detected multiple digits, it's likely noise or a misread
            if len(digits) > 1:
                # If they are all the same digit, it might be okay (e.g. "11" for "1")
                if all(d == digits[0] for d in digits):
                    return int(digits[0])
                return 0
            return int(digits[0])
                
        return 0

    except Exception as e:
        print(f"OCR Error at ({row_idx},{col_idx}): {e}")
        return 0


def preprocess_cell_for_ocr_with_margin(cell, border_reduction=6):
    """
    Preprocess cell with configurable border reduction
    border_reduction: pixels to trim from each edge (lower = more context, more grid lines)
    """
    if cell.size == 0:
        return np.full((100, 100), 255, dtype=np.uint8)
    
    # Apply border reduction to the input cell
    h, w = cell.shape[:2]
    if h > 2 * border_reduction and w > 2 * border_reduction:
        cell = cell[border_reduction:h-border_reduction, border_reduction:w-border_reduction]
    
    # Now apply standard OCR preprocessing
    target_size = 100
    cell = cv2.resize(cell, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    cell = cv2.GaussianBlur(cell, (3, 3), 0)
    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clear borders to remove grid line artifacts
    border = 6
    cell[0:border, :] = 0
    cell[border:target_size, :] = cell[border:target_size, :]
    cell[:, 0:border] = 0
    cell[:, border:target_size] = cell[:, border:target_size]
    
    # Find contours to identify the digit
    contours, _ = cv2.findContours(cell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    # Filter contours
    valid_contours = []
    potential_one = None  # Special handling for digit "1"
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # Check if this could be a "1" (tall and thin)
        if area > 100:
            x, y, w, h = cv2.boundingRect(c)
            if h > 2 * w:
                potential_one = c
        
        if area < 30 or area > 6000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > 3 * h:
            continue
        # Relaxed to allow "1" digits through (which are naturally thin)
        if h > 85 and w < 5:
            continue
        valid_contours.append(c)

    # If no valid contours but we have a potential "1", use it
    if not valid_contours and potential_one is not None:
        valid_contours = [potential_one]

    if not valid_contours:
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    # Get bounding box of all valid contours
    x_min, y_min = target_size, target_size
    x_max, y_max = 0, 0
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    digit_w = x_max - x_min
    digit_h = y_max - y_min
    digit = cell[y_min:y_max, x_min:x_max]

    # Create centered image
    centered = np.zeros((target_size, target_size), dtype=np.uint8)
    padding_ratio = 0.15
    max_dim = int(target_size * (1 - 2 * padding_ratio))
    scale = min(max_dim / digit_w, max_dim / digit_h) if digit_w > 0 and digit_h > 0 else 1
    new_w = int(digit_w * scale)
    new_h = int(digit_h * scale)
    
    if new_w > 0 and new_h > 0:
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        centered[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    centered = cv2.bitwise_not(centered)
    return centered


def extract_sudoku_from_image(image_path):
    """
    Main function to extract Sudoku grid from image
    Returns a 9x9 list of lists with digits (0 for empty cells)
    """
    try:
        print(f"Processing image: {image_path}")

        # Step 1: Preprocess image
        original, gray, thresh = preprocess_image(image_path)
        print("✓ Image preprocessed")

        # Step 2: Find Sudoku grid
        grid_contour = find_sudoku_grid(thresh, debug=True)
        if grid_contour is None:
            print("❌ Could not find Sudoku grid")
            return None
        print("✓ Sudoku grid found")

        # Step 3: Apply perspective transformation
        warped = perspective_transform(gray, grid_contour)
        print("✓ Perspective transformation applied")

        # Step 4: Extract individual cells
        cells = extract_cells(warped)
        print("✓ Cells extracted")

        # Step 5: Extract digits from each cell
        print("Starting digit extraction with PaddleOCR...")
        # Step 5: Extract digits from each cell sequentially
        sudoku_board = [[0] * 9 for _ in range(9)]
        
        for i in range(9):
            for j in range(9):
                # print(f"Processing cell {i},{j}...")
                digit = extract_digit_from_cell(cells[i][j], i, j)
                if digit != 0:
                    sudoku_board[i][j] = digit
                    print(f"Cell ({i},{j}) -> {digit}")

        print("✓ Digits extracted")

        # Print the extracted board for debugging
        print("Extracted Sudoku:")
        for row in sudoku_board:
            print(' '.join(str(d) if d != 0 else '.' for d in row))

        return sudoku_board

    except Exception as e:
        print(f"Error extracting Sudoku: {str(e)}")
        return None


def save_debug_images(image_path, output_dir="debug_output"):
    """Save intermediate processing steps for debugging"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Process image
        original, gray, thresh = preprocess_image(image_path)

        # Save preprocessing steps
        cv2.imwrite(f"{output_dir}/01_original.jpg", original)
        cv2.imwrite(f"{output_dir}/02_gray.jpg", gray)
        cv2.imwrite(f"{output_dir}/03_threshold.jpg", thresh)

        # Find and draw grid contour
        grid_contour = find_sudoku_grid(thresh, debug=False)
        if grid_contour is not None:
            contour_img = original.copy()
            cv2.drawContours(contour_img, [grid_contour], -1, (0, 255, 0), 3)
            cv2.imwrite(f"{output_dir}/04_grid_contour.jpg", contour_img)

            # Apply perspective transformation
            warped = perspective_transform(gray, grid_contour)
            cv2.imwrite(f"{output_dir}/05_warped.jpg", warped)

            # Extract cells
            cells = extract_cells(warped)
            
            # Create visualization showing all cell boundaries
            warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            height, width = warped.shape[:2]
            cell_height = height // 9
            cell_width = width // 9
            
            # Draw grid lines on warped image
            for i in range(10):
                # Horizontal lines
                y = i * cell_height
                color = (0, 255, 0) if i % 3 == 0 else (0, 200, 0)
                thickness = 2 if i % 3 == 0 else 1
                cv2.line(warped_color, (0, y), (width, y), color, thickness)
                
                # Vertical lines
                x = i * cell_width
                cv2.line(warped_color, (x, 0), (x, height), color, thickness)
            
            # Add cell labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(9):
                for j in range(9):
                    # Calculate cell center
                    cx = j * cell_width + cell_width // 2
                    cy = i * cell_height + cell_height // 2
                    
                    # Draw cell index (small, in corner)
                    label = f"{i},{j}"
                    text_size = cv2.getTextSize(label, font, 0.3, 1)[0]
                    text_x = j * cell_width + 2
                    text_y = i * cell_height + 10
                    cv2.putText(warped_color, label, (text_x, text_y), 
                               font, 0.3, (255, 0, 0), 1)
            
            cv2.imwrite(f"{output_dir}/06_cell_grid.jpg", warped_color)
            
            # Save ALL 81 cells (raw and processed)
            print(f"Saving all 81 cells to {output_dir}/...")
            for i in range(9):
                for j in range(9):
                    if cells[i][j].size > 0:
                        # Original warped cell
                        cv2.imwrite(f"{output_dir}/cell_raw_{i}_{j}.jpg", cells[i][j])
                        # Processed for OCR
                        processed = preprocess_cell_for_ocr(cells[i][j])
                        cv2.imwrite(f"{output_dir}/cell_ocr_{i}_{j}.jpg", processed)

        print(f"Debug images saved to {output_dir}/")

    except Exception as e:
        print(f"Error saving debug images: {str(e)}")


print("Image Script loaded successfully")
