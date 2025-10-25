import numpy as np
import cv2
from utils.image_utils import ImageUtils

class BayerInterferenceProtection:
    """
    Bayer Pattern Interference Protection - exploits camera sensor demosaicing

    This creates patterns that align with typical camera sensor Bayer filters,
    causing severe aliasing and color distortion when photographed while
    remaining imperceptible on screen displays.
    """

    def __init__(self):
        self.name = "Bayer Pattern Interference Protection"

    def apply(self, image, strength=0.5):
        """
        Apply Bayer pattern interference to image

        Args:
            image (numpy.ndarray): Input image in RGB format [0, 1]
            strength (float): Protection strength [0.1, 1.0]

        Returns:
            numpy.ndarray: Protected image
        """
        ImageUtils.validate_image(image)

        # Normalize strength
        strength = np.clip(strength, 0.1, 1.0)

        # Ensure we have a color image
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)

        protected_image = image.copy()

        # Apply multiple Bayer interference techniques
        protected_image = self._apply_rggb_pattern_interference(protected_image, strength)
        protected_image = self._apply_demosaic_confusion(protected_image, strength)
        protected_image = self._apply_chroma_subsampling_exploit(protected_image, strength)
        protected_image = self._apply_sensor_aliasing_patterns(protected_image, strength)

        return np.clip(protected_image, 0, 1)

    def _apply_rggb_pattern_interference(self, image, strength):
        """
        Create patterns that interfere with RGGB Bayer filter pattern

        Most digital cameras use RGGB (Red-Green-Green-Blue) Bayer pattern.
        This creates a 2x2 pixel grid pattern that will cause severe aliasing.

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with Bayer interference
        """
        height, width = image.shape[:2]
        protected_image = image.copy()

        # Create 2x2 Bayer pattern interference (RGGB)
        # Pattern that repeats every 2 pixels to align with sensor
        y_indices = np.arange(height)
        x_indices = np.arange(width)
        Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')

        # RGGB Bayer pattern:
        # R  G
        # G  B

        # Create alternating pattern for Red channel (top-left of 2x2)
        red_bayer_mask = ((Y % 2 == 0) & (X % 2 == 0)).astype(float)
        # Green channel appears twice (top-right and bottom-left)
        green_bayer_mask = (((Y % 2 == 0) & (X % 2 == 1)) | ((Y % 2 == 1) & (X % 2 == 0))).astype(float)
        # Blue channel (bottom-right of 2x2)
        blue_bayer_mask = ((Y % 2 == 1) & (X % 2 == 1)).astype(float)

        # Create high-frequency noise that aligns with Bayer pattern
        # This will cause color fringing and artifacts during demosaicing

        # Generate base pattern at Nyquist frequency for the sensor
        base_noise = self._generate_nyquist_pattern(height, width, strength)

        # Apply pattern with Bayer-specific modulation
        # Red channel: enhance at red positions
        red_interference = base_noise * (1.0 + red_bayer_mask * 2.0)
        protected_image[:, :, 0] += red_interference * strength * 0.020

        # Green channel: enhance at green positions (but inverted to confuse demosaicing)
        green_interference = base_noise * (1.0 - green_bayer_mask * 1.5)
        protected_image[:, :, 1] += green_interference * strength * 0.022

        # Blue channel: enhance at blue positions
        blue_interference = base_noise * (1.0 + blue_bayer_mask * 2.0)
        protected_image[:, :, 2] += blue_interference * strength * 0.020

        return protected_image

    def _generate_nyquist_pattern(self, height, width, strength):
        """
        Generate pattern at Nyquist frequency (highest frequency before aliasing)

        This creates maximum interference with sensor sampling

        Args:
            height, width (int): Image dimensions
            strength (float): Pattern strength

        Returns:
            numpy.ndarray: Nyquist frequency pattern
        """
        y = np.arange(height)
        x = np.arange(width)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Create pattern that alternates every pixel (Nyquist limit)
        # This is the maximum frequency that can be represented
        nyquist_h = np.power(-1.0, Y)
        nyquist_v = np.power(-1.0, X)
        nyquist_pattern = nyquist_h * nyquist_v

        # Add diagonal Nyquist patterns
        nyquist_diag1 = np.power(-1.0, X + Y)
        nyquist_diag2 = np.power(-1.0, X - Y)

        # Combine patterns with phase shifts for maximum interference
        combined = (0.3 * nyquist_pattern +
                   0.25 * nyquist_h +
                   0.25 * nyquist_v +
                   0.1 * nyquist_diag1 +
                   0.1 * nyquist_diag2)

        # Modulate with lower frequency carrier for natural appearance
        carrier_freq = 8 + int(strength * 12)
        carrier = np.sin(2 * np.pi * X / carrier_freq) * np.cos(2 * np.pi * Y / carrier_freq)

        # Combine high-frequency pattern with carrier
        pattern = combined * (0.5 + 0.5 * carrier)

        # Normalize
        pattern = pattern - np.mean(pattern)
        if np.std(pattern) > 0:
            pattern = pattern / (np.std(pattern) * 3.0)

        return pattern

    def _apply_demosaic_confusion(self, image, strength):
        """
        Create patterns that specifically confuse demosaicing algorithms

        Demosaicing reconstructs full RGB from Bayer pattern. We exploit this
        by creating false correlation between adjacent pixels.

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with demosaic confusion
        """
        height, width = image.shape[:2]
        protected_image = image.copy()

        # Create inter-channel correlation patterns that break demosaicing
        # Demosaicing assumes smooth color transitions - we violate this

        y = np.arange(height)
        x = np.arange(width)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Create checkerboard pattern (2x2 blocks)
        checkerboard = ((X // 2) % 2) ^ ((Y // 2) % 2)
        checkerboard = checkerboard.astype(float) * 2 - 1  # Convert to -1, 1

        # Create false edge patterns at pixel boundaries
        pixel_edge_h = np.abs(Y % 2 - 0.5) * 2  # 0 at even rows, 1 at odd rows
        pixel_edge_v = np.abs(X % 2 - 0.5) * 2  # 0 at even cols, 1 at odd cols

        # Combine into demosaic confusion pattern
        confusion_base = checkerboard * pixel_edge_h * pixel_edge_v

        # Add high-frequency modulation
        hf_freq = 30 + int(strength * 40)
        hf_mod = np.sin(2 * np.pi * X / hf_freq) * np.sin(2 * np.pi * Y / hf_freq)

        confusion_pattern = confusion_base * (0.7 + 0.3 * hf_mod)

        # Apply to channels in anti-correlated way (opposite of what demosaicing expects)
        protected_image[:, :, 0] += confusion_pattern * strength * 0.015
        protected_image[:, :, 1] -= confusion_pattern * strength * 0.018  # Inverted for green
        protected_image[:, :, 2] += confusion_pattern * strength * 0.015

        return protected_image

    def _apply_chroma_subsampling_exploit(self, image, strength):
        """
        Exploit camera and JPEG chroma subsampling (4:2:2 or 4:2:0)

        Cameras often use chroma subsampling during compression, which
        can amplify carefully crafted color patterns.

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with chroma subsampling exploitation
        """
        height, width = image.shape[:2]

        # Convert to YCbCr-like representation
        # Y = luminance, Cb/Cr = chrominance
        Y = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        Cb = image[:, :, 2] - Y
        Cr = image[:, :, 0] - Y

        # Create patterns at 2x2 block boundaries (matches 4:2:0 subsampling)
        y_indices = np.arange(height)
        x_indices = np.arange(width)
        Y_grid, X_grid = np.meshgrid(y_indices, x_indices, indexing='ij')

        # Create high-frequency chroma patterns
        # These will be amplified by subsampling/upsampling process
        chroma_freq = 2  # Align with subsampling blocks
        cb_pattern = np.power(-1.0, X_grid // chroma_freq) * np.power(-1.0, Y_grid // chroma_freq)
        cr_pattern = -cb_pattern  # Anti-correlated

        # Add higher frequency modulation
        mod_freq = 16 + int(strength * 24)
        modulation = np.sin(2 * np.pi * X_grid / mod_freq) * np.cos(2 * np.pi * Y_grid / mod_freq)

        # Apply chroma patterns
        Cb_modified = Cb + cb_pattern * modulation * strength * 0.012
        Cr_modified = Cr + cr_pattern * modulation * strength * 0.012

        # Convert back to RGB
        protected_image = image.copy()
        protected_image[:, :, 0] = Y + Cr_modified
        protected_image[:, :, 1] = Y - 0.344136 * Cb_modified - 0.714136 * Cr_modified
        protected_image[:, :, 2] = Y + Cb_modified

        return protected_image

    def _apply_sensor_aliasing_patterns(self, image, strength):
        """
        Create patterns that cause sensor aliasing artifacts

        These patterns interact with the sensor pixel grid to create
        moiré patterns and color artifacts.

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with sensor aliasing patterns
        """
        height, width = image.shape[:2]
        protected_image = image.copy()

        y = np.arange(height)
        x = np.arange(width)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Create patterns at specific frequencies that cause aliasing
        # These frequencies are chosen to interact poorly with typical sensor pitches

        # Pattern 1: Near-Nyquist diagonal pattern
        diag_freq = 2.5  # Just above Nyquist
        diagonal_pattern = np.sin(2 * np.pi * (X + Y) / diag_freq)

        # Pattern 2: Moiré-inducing grid
        moire_freq = 3 + strength * 2
        moire_h = np.sin(2 * np.pi * X / moire_freq)
        moire_v = np.sin(2 * np.pi * Y / moire_freq)
        moire_pattern = moire_h * moire_v

        # Pattern 3: Radial pattern (causes spiral artifacts)
        center_y, center_x = height // 2, width // 2
        radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        radial_freq = 4 + strength * 3
        radial_pattern = np.sin(2 * np.pi * radius / radial_freq)

        # Pattern 4: Fine checkerboard with rotation
        angle = np.arctan2(Y - center_y, X - center_x)
        rotated_x = (X - center_x) * np.cos(0.1) - (Y - center_y) * np.sin(0.1)
        rotated_y = (X - center_x) * np.sin(0.1) + (Y - center_y) * np.cos(0.1)
        checker_freq = 3
        checker_pattern = np.sign(np.sin(2 * np.pi * rotated_x / checker_freq) *
                                  np.sin(2 * np.pi * rotated_y / checker_freq))

        # Combine aliasing patterns
        aliasing_pattern = (0.3 * diagonal_pattern +
                          0.3 * moire_pattern +
                          0.2 * radial_pattern +
                          0.2 * checker_pattern)

        # Normalize
        aliasing_pattern = aliasing_pattern - np.mean(aliasing_pattern)
        if np.std(aliasing_pattern) > 0:
            aliasing_pattern = aliasing_pattern / (np.std(aliasing_pattern) * 2.0)

        # Apply to each channel with different phases to maximize color distortion
        protected_image[:, :, 0] += aliasing_pattern * strength * 0.018
        protected_image[:, :, 1] += np.roll(aliasing_pattern, 1, axis=0) * strength * 0.018
        protected_image[:, :, 2] += np.roll(aliasing_pattern, 1, axis=1) * strength * 0.018

        return protected_image
