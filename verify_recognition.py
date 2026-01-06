
import os
import sys
from image_processing import extract_sudoku_from_image, save_debug_images

def verify_image(image_path):
    """Test extraction on a specific image"""
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}\n")

    print("Saving debug images...")
    save_debug_images(image_path)

    print(f"Testing extraction on {image_path}...")
    board = extract_sudoku_from_image(image_path)

    if board:
        print("\nFinal Extracted Board:")
        for row in board:
            print(' '.join(str(d) if d != 0 else '.' for d in row))
            
        count = sum(1 for row in board for d in row if d != 0)
        print(f"\nTotal digits found: {count}")
        return board
    else:
        print("FAILURE: No board returned.")
        return None

def verify_multiple():
    """Test on multiple images"""
    images = ['sample_image.png', 'sample_image_2.png']
    
    results = {}
    for img in images:
        if os.path.exists(img):
            results[img] = verify_image(img)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for img, board in results.items():
        if board:
            count = sum(1 for row in board for d in row if d != 0)
            print(f"✓ {img}: {count} digits extracted")
        else:
            print(f"✗ {img}: FAILED")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific image
        verify_image(sys.argv[1])
    else:
        # Test all images
        verify_multiple()
