
import time
import os
from image_processing import extract_sudoku_from_image

img = 'sample_image.png'
if not os.path.exists(img):
    print(f"Image {img} not found. using sample_image_2.png if available")
    img = 'sample_image_2.png'

if not os.path.exists(img):
    print("No sample images found.")
    exit(1)

print(f"Testing performance on {img}")
start = time.time()
extract_sudoku_from_image(img)
end = time.time()
print(f"Duration: {end - start:.2f}s")
