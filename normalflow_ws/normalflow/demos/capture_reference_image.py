import argparse
import os
import cv2
import numpy as np
import yaml
from datetime import datetime

from gs_sdk.gs_device import Camera, FastCamera
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.utils import erode_contact_mask

"""
This script captures reference images from GelSight sensor for later use in object tracking.

Usage:
    python capture_reference_image.py [--output_dir OUTPUT_DIR] [--calib_model_path CALIB_MODEL_PATH] [--config_path CONFIG_PATH]

Instructions:
    1. Run this script and wait for the sensor to initialize
    2. Press the object against the sensor 
    3. Press 'c' to capture and save the current frame
    4. Press 's' to show contact information
    5. Press 'q' to quit

Arguments:
    --output_dir: Directory to save captured images (default: ./captured_images/)
    --calib_model_path: Path to calibration model
    --config_path: Path to sensor configuration file
    --streamer: Sensor streamer type ('opencv' or 'ffmpeg')
"""

# Default paths
calib_model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")


def capture_reference_images():
    # Argument Parser
    parser = argparse.ArgumentParser(
        description="Capture reference images from GelSight sensor."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Directory to save captured images",
        default="./normalflow/demos/images",
    )
    parser.add_argument(
        "-b",
        "--calib_model_path",
        type=str,
        help="Path to calibration model",
        default=calib_model_path,
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="Path to sensor configuration file",
        default=config_path,
    )
    parser.add_argument(
        "-s",
        "--streamer",
        type=str,
        choices=["opencv", "ffmpeg"],
        help="Sensor streamer type",
        default="opencv",
    )
    parser.add_argument(
        "--preview_contact",
        action="store_true",
        help="Show contact area preview (requires background collection)",
    )
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Images will be saved to: {args.output_dir}")

    # Read sensor configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        ppmm = config["ppmm"]
        imgh = config["imgh"]
        imgw = config["imgw"]
        raw_imgh = config["raw_imgh"]
        raw_imgw = config["raw_imgw"]
        framerate = config["framerate"]

    # Connect to sensor
    print("Connecting to sensor...")
    if args.streamer == "opencv":
        device = Camera(device_name, imgh, imgw)
    elif args.streamer == "ffmpeg":
        device = FastCamera(device_name, imgh, imgw, raw_imgh, raw_imgw, framerate)
    
    device.connect()
    print("Sensor connected successfully!")

    # Initialize reconstructor if contact preview is requested
    recon = None
    if args.preview_contact:
        print("Initializing reconstructor for contact preview...")
        recon = Reconstructor(args.calib_model_path, device="cpu")
        
        # Collect background images
        print("Collecting background images (remove any objects from sensor)...")
        input("Press Enter when sensor surface is clear...")
        
        bg_images = []
        for i in range(10):
            image = device.get_image()
            bg_images.append(image)
            print(f"Collecting background image {i+1}/10...")
        
        bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
        recon.load_bg(bg_image)
        print("Background collection completed!")

    print("\n" + "="*60)
    print("CAPTURE REFERENCE IMAGES")
    print("="*60)
    print("Controls:")
    print("  'c' - Capture and save current frame")
    if args.preview_contact:
        print("  's' - Show contact information")
    print("  'q' - Quit")
    print("="*60)

    capture_count = 0
    is_running = True
    
    while is_running:
        # Get current frame
        image = device.get_image()
        display_image = image.copy()
        
        # Add status text to display
        cv2.putText(
            display_image,
            f"Captures: {capture_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        cv2.putText(
            display_image,
            "Press 'c' to capture, 'q' to quit",
            (10, display_image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Show contact information if available
        if recon is not None:
            try:
                G, H, C = recon.get_surface_info(image, ppmm)
                C = erode_contact_mask(C)
                contact_area = np.sum(C)
                
                # Add contact info to display
                if contact_area > 0:
                    cv2.putText(
                        display_image,
                        f"Contact Area: {contact_area} pixels",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if contact_area >= 500 else (0, 255, 255),
                        2,
                    )
                    
                    # Draw contact contours
                    if contact_area >= 100:  # Only draw if there's meaningful contact
                        contours, _ = cv2.findContours(
                            (C * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(display_image, contours, -1, (0, 255, 0), 2)
                        
                        if contours:
                            # Find and mark center of largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            M = cv2.moments(largest_contour)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.circle(display_image, (cx, cy), 5, (0, 0, 255), -1)
                                cv2.putText(
                                    display_image,
                                    f"Center: ({cx}, {cy})",
                                    (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )
                else:
                    cv2.putText(
                        display_image,
                        "No Contact Detected",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
            except Exception as e:
                cv2.putText(
                    display_image,
                    f"Contact analysis error: {str(e)[:30]}...",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        # Display the frame
        cv2.imshow("GelSight Sensor - Reference Image Capture", display_image)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            is_running = False
            print("Quitting...")
            
        elif key == ord('c'):
            # Capture current frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reference_image_{timestamp}_{capture_count:03d}.jpg"
            filepath = os.path.join(args.output_dir, filename)
            
            # Save the original image (not the display version with annotations)
            success = cv2.imwrite(filepath, image)
            
            if success:
                capture_count += 1
                print(f"✓ Captured image saved: {filepath}")
                
                # Show contact info for this capture if available
                if recon is not None:
                    try:
                        G, H, C = recon.get_surface_info(image, ppmm)
                        C = erode_contact_mask(C)
                        contact_area = np.sum(C)
                        print(f"  Contact area: {contact_area} pixels")
                        
                        if contact_area >= 500:
                            print("  ✓ Sufficient contact area for tracking")
                        elif contact_area > 0:
                            print("  ⚠ Contact area may be too small for reliable tracking")
                        else:
                            print("  ⚠ No contact detected")
                    except Exception as e:
                        print(f"  Contact analysis failed: {e}")
                
                # Brief flash effect
                flash_image = np.ones_like(display_image) * 255
                cv2.imshow("GelSight Sensor - Reference Image Capture", flash_image)
                cv2.waitKey(100)
                
            else:
                print(f"✗ Failed to save image: {filepath}")
                
        elif key == ord('s') and recon is not None:
            # Show detailed contact information
            try:
                G, H, C = recon.get_surface_info(image, ppmm)
                C = erode_contact_mask(C)
                contact_area = np.sum(C)
                
                print(f"\n--- Contact Information ---")
                print(f"Contact area: {contact_area} pixels")
                print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
                print(f"Contact ratio: {contact_area/(image.shape[0]*image.shape[1])*100:.2f}%")
                
                if contact_area > 0:
                    contours, _ = cv2.findContours(
                        (C * 255).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            print(f"Contact center: ({cx}, {cy})")
                        
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        print(f"Contact bounding box: ({x}, {y}) to ({x+w}, {y+h})")
                        print(f"Bounding box size: {w}x{h} pixels")
                
                if contact_area >= 500:
                    print("✓ Contact area is sufficient for tracking")
                else:
                    print("⚠ Contact area may be insufficient for reliable tracking")
                    
                print("-------------------------\n")
                
            except Exception as e:
                print(f"Error analyzing contact: {e}")

    # Cleanup
    device.release()
    cv2.destroyAllWindows()
    
    print(f"\nCapture session completed!")
    print(f"Total images captured: {capture_count}")
    print(f"Images saved in: {args.output_dir}")
    
    if capture_count > 0:
        print("\nTo use a captured image as reference:")
        print(f"python realtime_object_tracking.py --reference_image {args.output_dir}/reference_image_XXXXXX_XXX.jpg")


if __name__ == "__main__":
    capture_reference_images()