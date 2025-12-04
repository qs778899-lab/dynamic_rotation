"""
Default images information for sequential tracking demo.
This file contains information about the available demo images.
"""

import os

# Get the directory of this file
CURRENT_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")

# Default image paths
DEFAULT_IMAGES = {
    'image1': os.path.join(IMAGES_DIR, "1.jpg"),
    'image2': os.path.join(IMAGES_DIR, "2.jpg"), 
    'image3': os.path.join(IMAGES_DIR, "3.jpg"),
    'image4': os.path.join(IMAGES_DIR, "4.jpg"),
    'image5': os.path.join(IMAGES_DIR, "5.jpg"),
}

# Sequential pairs for tracking
SEQUENTIAL_PAIRS = [
    ('image1', 'image2'),  # image1 as reference for image2
    ('image2', 'image3'),  # image2 as reference for image3
    ('image3', 'image4'),  # image3 as reference for image4
    ('image4', 'image5'),  # image4 as reference for image5
]

def get_available_images():
    """Get list of available image files."""
    available = {}
    for name, path in DEFAULT_IMAGES.items():
        if os.path.exists(path):
            available[name] = path
    return available

def get_image_sequence():
    """Get sequential list of image paths for processing."""
    available = get_available_images()
    sequence = []
    for i in range(1, 6):  # 1.jpg to 5.jpg
        image_name = f'image{i}'
        if image_name in available:
            sequence.append(available[image_name])
    return sequence

def print_images_info():
    """Print information about available images."""
    print("Available demo images:")
    available = get_available_images()
    for name, path in available.items():
        print(f"  {name}: {path}")
    
    print(f"\nImages directory: {IMAGES_DIR}")
    print(f"Total available images: {len(available)}")
    
    sequence = get_image_sequence()
    print(f"\nSequence for processing: {len(sequence)} images")
    for i, path in enumerate(sequence):
        print(f"  {i+1}. {os.path.basename(path)}")

if __name__ == "__main__":
    print_images_info()
