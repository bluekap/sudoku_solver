import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import concurrent.futures

# You might need to set the tesseract path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows


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

            # Reduced padding to capture more context around digits
            # This allows slight overlap with neighboring cells for better OCR
            padding = 2  # Reduced from 6 to get more digit context
            cell = warped_image[y1 + padding:y2 - padding, x1 + padding:x2 - padding]
            row_cells.append(cell)

        cells.append(row_cells)

    return cells


def preprocess_cell_for_ocr(cell):
    """Preprocess individual cell for better OCR results"""
    if cell.size == 0:
        return cell

    # Use a larger target size for better Tesseract recognition
    # 100x100 provides good quality without being too large
    target_size = 100
    
    # Resize with INTER_CUBIC for better quality
    cell = cv2.resize(cell, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    # Light Gaussian blur to reduce noise
    cell = cv2.GaussianBlur(cell, (3, 3), 0)

    # Apply Otsu's thresholding to get binary image
    # THRESH_BINARY_INV: white text on black background
    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clear borders to remove grid line artifacts
    border = 6
    cell[0:border, :] = 0
    cell[border:target_size, :] = cell[border:target_size, :]
    cell[:, 0:border] = 0
    cell[:, border:target_size] = cell[:, border:target_size]
    
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
    """Extract digit from a single cell using Tesseract OCR with intelligent retry logic"""
    
    # First check if the RAW cell has meaningful content by counting black pixels
    # Apply a simple threshold to the raw cell to detect digit presence
    if cell.size == 0:
        return 0
    
    # Check raw cell for content (before heavy preprocessing)
    _, raw_thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw_black_pixels = np.sum(raw_thresh > 128)
    
    # If very few black pixels in raw cell, it's likely empty
    # Lowered threshold to catch thin "1" digits
    # Adjusted to 200 to filter out noise at (2,7) which has 165 pixels
    # REVERT: 500 was too high for sample_image_3.png. 250 still missed a very thin '1'.
    # Setting to 150 to be extremely safe.
    if raw_black_pixels < 150:
        return 0
    
    # Define different preprocessing strategies with varying border margins
    # Each strategy extracts different amounts of the cell to handle grid lines
    # Reduced list to improve performance: removed redundant ultra_simple variants (line/block)
    strategies =  [
        {'border_reduction': 6, 'psm': 10, 'name': 'standard', 'use_simple': False},   # Default: remove more border
        {'border_reduction': 3, 'psm': 7, 'name': 'expanded', 'use_simple': False},     # Keep more context
        {'border_reduction': 1, 'psm': 8, 'name': 'minimal', 'use_simple': False},      # Minimal border removal
        {'border_reduction': 0, 'psm': 10, 'name': 'ultra_simple', 'use_simple': True}, # No filtering for thin digits
        {'border_reduction': 0, 'psm': 7, 'name': 'ultra_simple_line', 'use_simple': True}, # Single line mode
        {'border_reduction': 0, 'psm': 6, 'name': 'ultra_simple_block', 'use_simple': True}, # Block mode
    ]
    
    # Cache for the simple preprocessing result since it's shared across multiple strategies
    simple_processed_cell_cache = None

    for attempt, strategy in enumerate(strategies, 1):
        try:
            # For ultra_simple strategy, use basic preprocessing without contour filtering
            if strategy.get('use_simple', False):
                if simple_processed_cell_cache is None:
                    # Simple preprocessing for thin digits like "1"
                    target_size = 100
                    processed_cell = cv2.resize(cell, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
                    processed_cell = cv2.GaussianBlur(processed_cell, (3, 3), 0)
                    _, processed_cell = cv2.threshold(processed_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    # Clear just the edges (very minimal for thin digits)
                    border = 5
                    processed_cell[0:border, :] = 0
                    processed_cell[-border:, :] = 0
                    processed_cell[:, 0:border] = 0
                    processed_cell[:, -border:] = 0
                    
                    # Invert to black on white for OCR
                    processed_cell = cv2.bitwise_not(processed_cell)
                    simple_processed_cell_cache = processed_cell
                
                processed_cell = simple_processed_cell_cache
            else:
                # Standard preprocessing with contour filtering
                processed_cell = preprocess_cell_for_ocr_with_margin(cell, strategy['border_reduction'])
            
            # Check if preprocessing left meaningful content
            # Lower threshold for ultra_simple to catch thin "1"s
            min_pixels = 20 if strategy.get('use_simple', False) else 40
            num_black_pixels = np.sum(processed_cell < 128)
            if num_black_pixels < min_pixels:
                continue  # Skip this strategy if preprocessing removed too much
            
            # Additional validation for ultra_simple: Check if meaningful shapes exist
            # This filters out small noise specs that pass the pixel count but aren't digits
            if strategy.get('use_simple', False):
                 # Invert back to find contours (white on black)
                 inverted_for_contours = cv2.bitwise_not(processed_cell)
                 contours, _ = cv2.findContours(inverted_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                 
                 if not contours:
                     continue
                 
                 # Get max height of any contour
                 max_h = 0
                 for c in contours:
                     _, _, _, h = cv2.boundingRect(c)
                     max_h = max(max_h, h)
                 
                 # If largest blob is too short (less than 20% of cell), it's noise
                 # Digits are usually at least 50% height
                 if max_h < 20:
                     continue

            # Try OCR with this strategy's PSM mode
            config = f'--oem 3 --psm {strategy["psm"]} -c tessedit_char_whitelist=123456789'
            text = pytesseract.image_to_string(processed_cell, config=config).strip()
            
            # Validate the result
            if text:
                text = ''.join(filter(str.isdigit, text))
                if text and 1 <= int(text) <= 9:
                    digit = int(text[0])
                    
                    # Validate based on pixel density to filter noise
                    # "1" can be very thin, but other digits usually have more pixels
                    min_pixels = 150 if digit == 1 else 200
                    
                    if raw_black_pixels < min_pixels:
                        # If we think it's a digit but it has too few pixels, it's likely noise or a ghost
                        # We continue to try other strategies, or if all fail, it will return 0
                        continue

                    if attempt > 1:
                        print(f"Cell ({row_idx},{col_idx}) | Raw: {raw_black_pixels:4d} | OCR: '{digit}' (attempt {attempt}, {strategy['name']}, PSM {strategy['psm']})")
                    else:
                        print(f"Cell ({row_idx},{col_idx}) | Raw: {raw_black_pixels:4d} | OCR: '{digit}'")
                    return digit
            
        except Exception as e:
            if attempt == len(strategies):  # Only print error on last attempt
                print(f"OCR Error at ({row_idx},{col_idx}): {e}")
            continue
    
    # If all attempts failed but we have significant raw content, log it
    if raw_black_pixels > 300:  # Significant content in raw cell
        print(f"Cell ({row_idx},{col_idx}) | Raw: {raw_black_pixels:4d} | OCR: '' (failed all {len(strategies)} strategies)")
    
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
        # Step 5: Extract digits from each cell in parallel
        sudoku_board = [[0] * 9 for _ in range(9)]
        
        def process_cell(args):
            r, c, cell_img = args
            return r, c, extract_digit_from_cell(cell_img, r, c)

        # Prepare tasks
        tasks = []
        for i in range(9):
            for j in range(9):
                tasks.append((i, j, cells[i][j]))
        
        # Execute in parallel
        print("Starting parallel digit extraction...")
        # Optimal max_workers usually roughly corresponds to CPU cores.
        # Too many might cause thrashing or high memory usage. 4-8 is usually safe.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_cell, tasks)
            
            for r, c, digit in results:
                sudoku_board[r][c] = digit

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
