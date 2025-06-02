import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

# You might need to set the tesseract path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows


# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac

def preprocess_image(image_path):
    """Load and preprocess the image for better grid detection"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive threshold to get binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    return img, gray, thresh


def find_sudoku_grid(thresh_img):
    """Find the largest rectangular contour (should be the Sudoku grid)"""
    # Find contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    grid_contour = None

    # Look for the largest rectangular contour
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If we found a 4-sided contour with sufficient area, it's likely our grid
        if len(approx) == 4 and cv2.contourArea(contour) > 10000:
            grid_contour = approx
            break

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

            # Extract cell with some padding removed to avoid grid lines
            padding = 5
            cell = warped_image[y1 + padding:y2 - padding, x1 + padding:x2 - padding]
            row_cells.append(cell)

        cells.append(row_cells)

    return cells


def preprocess_cell_for_ocr(cell):
    """Preprocess individual cell for better OCR results"""
    if cell.size == 0:
        return cell

    # Resize cell to a standard size
    cell = cv2.resize(cell, (64, 64))

    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)

    # Apply additional threshold
    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cell


def extract_digit_from_cell(cell):
    """Extract digit from a single cell using OCR"""
    # Preprocess the cell
    processed_cell = preprocess_cell_for_ocr(cell)

    # Check if cell is mostly empty (no digit)
    if np.sum(processed_cell) < 1000:  # Threshold for empty cells
        return 0

    try:
        # Use Tesseract to extract text
        config = '--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
        text = pytesseract.image_to_string(processed_cell, config=config).strip()

        # Validate the result
        if text and text.isdigit() and 1 <= int(text) <= 9:
            return int(text)
        else:
            return 0
    except:
        return 0


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
        grid_contour = find_sudoku_grid(thresh)
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
        sudoku_board = []
        for i in range(9):
            row = []
            for j in range(9):
                digit = extract_digit_from_cell(cells[i][j])
                row.append(digit)
            sudoku_board.append(row)

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
        grid_contour = find_sudoku_grid(thresh)
        if grid_contour is not None:
            contour_img = original.copy()
            cv2.drawContours(contour_img, [grid_contour], -1, (0, 255, 0), 3)
            cv2.imwrite(f"{output_dir}/04_grid_contour.jpg", contour_img)

            # Apply perspective transformation
            warped = perspective_transform(gray, grid_contour)
            cv2.imwrite(f"{output_dir}/05_warped.jpg", warped)

            # Extract and save some cells
            cells = extract_cells(warped)
            for i in range(3):  # Save first 3 rows of cells
                for j in range(3):  # Save first 3 columns
                    if cells[i][j].size > 0:
                        cv2.imwrite(f"{output_dir}/cell_{i}_{j}.jpg", cells[i][j])

        print(f"Debug images saved to {output_dir}/")

    except Exception as e:
        print(f"Error saving debug images: {str(e)}")


print("Image Script loaded successfully")
