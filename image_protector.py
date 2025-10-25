#!/usr/bin/env python3
"""
Image Anti-Photography Protection System
Simple folder-based terminal interface
"""

import os
import sys
from utils.image_utils import ImageUtils
from protection.frequency_domain import FrequencyDomainProtection
from protection.infrared_patterns import InfraredPatternProtection
from protection.adversarial_noise import AdversarialNoiseProtection
from protection.bayer_interference import BayerInterferenceProtection
from protection.screen_moire import ScreenMoireProtection

def get_images_from_folder(folder_path):
    """Get all supported image files from a folder"""
    if not os.path.exists(folder_path):
        return []

    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_extensions:
            images.append(filename)

    return sorted(images)

def select_image(images):
    """Let user select which image to process"""
    if len(images) == 0:
        print("No supported images found in the input folder.")
        return None

    if len(images) == 1:
        print(f"Found 1 image: {images[0]}")
        return images[0]

    print(f"\nFound {len(images)} images:")
    for i, image in enumerate(images, 1):
        print(f"  {i}. {image}")

    while True:
        try:
            choice = input(f"\nSelect image (1-{len(images)}) or 'all' for batch processing: ").strip().lower()

            if choice == 'all':
                return 'all'

            choice_num = int(choice)
            if 1 <= choice_num <= len(images):
                return images[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(images)}")
        except ValueError:
            print("Please enter a valid number or 'all'")

def select_protection_level():
    """Let user select protection level"""
    print("\nProtection levels:")
    print("  1. Light - Moderate camera distortion, imperceptible on screen")
    print("  2. Medium - Strong camera distortion (recommended)")
    print("  3. Heavy - Maximum camera distortion, complete image destruction")
    print("  4. Extreme - Experimental maximum strength")

    while True:
        try:
            choice = input("\nSelect protection level (1-4): ").strip()
            choice_num = int(choice)

            if choice_num == 1:
                return 'light', {
                    'frequency': 0.4, 'infrared': 0.3, 'adversarial': 0.3,
                    'bayer': 0.3, 'moire': 0.3
                }
            elif choice_num == 2:
                return 'medium', {
                    'frequency': 0.7, 'infrared': 0.6, 'adversarial': 0.6,
                    'bayer': 0.6, 'moire': 0.6
                }
            elif choice_num == 3:
                return 'heavy', {
                    'frequency': 0.9, 'infrared': 0.85, 'adversarial': 0.85,
                    'bayer': 0.85, 'moire': 0.85
                }
            elif choice_num == 4:
                return 'extreme', {
                    'frequency': 1.0, 'infrared': 1.0, 'adversarial': 1.0,
                    'bayer': 1.0, 'moire': 1.0
                }
            else:
                print("Please enter 1, 2, 3, or 4")
        except ValueError:
            print("Please enter a valid number (1, 2, 3, or 4)")

def protect_image(input_path, output_path, protection_settings):
    """Protect a single image"""
    try:
        # Load image
        print(f"Loading {os.path.basename(input_path)}...")
        image = ImageUtils.load_image(input_path)

        # Initialize protection algorithms
        freq_protection = FrequencyDomainProtection()
        ir_protection = InfraredPatternProtection()
        adv_protection = AdversarialNoiseProtection()
        bayer_protection = BayerInterferenceProtection()
        moire_protection = ScreenMoireProtection()

        # Apply protections in strategic order
        protected_image = image.copy()

        print("Applying Bayer sensor interference...")
        protected_image = bayer_protection.apply(protected_image, protection_settings['bayer'])

        print("Applying screen moiré patterns...")
        protected_image = moire_protection.apply(protected_image, protection_settings['moire'])

        print("Applying high-frequency noise protection...")
        protected_image = freq_protection.apply(protected_image, protection_settings['frequency'])

        print("Applying infrared spectrum protection...")
        protected_image = ir_protection.apply(protected_image, protection_settings['infrared'])

        print("Applying adversarial noise protection...")
        protected_image = adv_protection.apply(protected_image, protection_settings['adversarial'])

        # Save protected image
        print(f"Saving to {os.path.basename(output_path)}...")
        ImageUtils.save_image(protected_image, output_path)

        # Calculate difference for reporting
        import numpy as np
        difference = np.mean(np.abs(protected_image - image))

        print(f"✓ Protection complete! (difference: {difference:.6f})")
        return True

    except Exception as e:
        print(f"✗ Error processing image: {e}")
        return False

def main():
    """Main application"""
    print("=" * 60)
    print("Image Anti-Photography Protection System")
    print("=" * 60)

    # Setup input folder
    input_folder = "input_images"
    output_folder = "protected_images"

    print(f"\nInput folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Get images from input folder
    images = get_images_from_folder(input_folder)

    if len(images) == 0:
        print(f"\nNo images found in '{input_folder}' folder.")
        print("Please add some images (jpg, png, bmp, tiff) to the input_images folder and run again.")
        return 1

    # Select image(s) to process
    selected = select_image(images)
    if selected is None:
        return 1

    # Select protection level
    level_name, protection_settings = select_protection_level()

    print(f"\nUsing {level_name} protection:")
    print(f"  Bayer Interference: {protection_settings['bayer']}")
    print(f"  Screen Moiré: {protection_settings['moire']}")
    print(f"  Frequency Domain: {protection_settings['frequency']}")
    print(f"  Infrared Spectrum: {protection_settings['infrared']}")
    print(f"  Adversarial Noise: {protection_settings['adversarial']}")

    # Process image(s)
    print("\n" + "-" * 40)

    if selected == 'all':
        print(f"Processing {len(images)} images...")
        success_count = 0

        for image_name in images:
            input_path = os.path.join(input_folder, image_name)

            # Create output filename with protection level
            name, ext = os.path.splitext(image_name)
            output_name = f"{name}_protected_{level_name}{ext}"
            output_path = os.path.join(output_folder, output_name)

            print(f"\n[{success_count + 1}/{len(images)}] Processing {image_name}...")

            if protect_image(input_path, output_path, protection_settings):
                success_count += 1

        print(f"\n" + "=" * 40)
        print(f"Batch processing complete: {success_count}/{len(images)} images protected")

    else:
        input_path = os.path.join(input_folder, selected)

        # Create output filename
        name, ext = os.path.splitext(selected)
        output_name = f"{name}_protected_{level_name}{ext}"
        output_path = os.path.join(output_folder, output_name)

        if protect_image(input_path, output_path, protection_settings):
            print(f"\n" + "=" * 40)
            print("Protection complete!")
            print(f"Protected image saved as: {output_name}")
        else:
            return 1

    print(f"\nProtected images are in the '{output_folder}' folder.")
    print("\nTo test effectiveness:")
    print("1. Display the protected image FULL SCREEN on your device")
    print("2. Take a photo with your iPhone camera")
    print("3. View the captured photo on Mac Preview")
    print("4. The protected image should appear completely distorted with:")
    print("   - Severe color fringing and moiré patterns")
    print("   - Pixelated/blocky artifacts")
    print("   - Rainbow interference patterns")
    print("\nNote: The image looks normal on screen but breaks when photographed!")

    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        exit(1)