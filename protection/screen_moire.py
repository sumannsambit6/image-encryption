import numpy as np
import cv2
from utils.image_utils import ImageUtils

class ScreenMoireProtection:
    """
    Screen Moiré Pattern Protection - exploits screen pixel structure

    Creates patterns optimized for screen display that cause severe moiré
    patterns when photographed due to interaction between screen pixel grid
    and camera sensor grid.
    """

    def __init__(self):
        self.name = "Screen Moiré Pattern Protection"

    def apply(self, image, strength=0.5):
        """
        Apply screen moiré patterns to image

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

        # Apply multiple screen-optimized patterns
        protected_image = self._apply_lcd_subpixel_pattern(protected_image, strength)
        protected_image = self._apply_screen_grid_interference(protected_image, strength)
        protected_image = self._apply_rgb_stripe_pattern(protected_image, strength)
        protected_image = self._apply_pentile_confusion(protected_image, strength)

        return np.clip(protected_image, 0, 1)

    def _apply_lcd_subpixel_pattern(self, image, strength):
        """
        Create patterns that align with LCD subpixel structure

        Most LCD screens use RGB subpixel arrangement. This creates patterns
        that look smooth on the screen but create moiré when photographed.

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with LCD subpixel patterns
        """
        height, width = image.shape[:2]
        protected_image = image.copy()

        # LCD screens typically have RGB stripes horizontally
        # Each pixel consists of R, G, B subpixels side by side

        y = np.arange(height)
        x = np.arange(width)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Create subpixel-aligned patterns
        # Pattern that aligns with typical RGB stripe layout (3 subpixels per pixel)
        subpixel_phase = (X % 3) / 3.0 * 2 * np.pi

        # Red subpixel enhancement (leftmost)
        red_subpixel = np.cos(subpixel_phase)
        # Green subpixel enhancement (center)
        green_subpixel = np.cos(subpixel_phase - 2 * np.pi / 3)
        # Blue subpixel enhancement (rightmost)
        blue_subpixel = np.cos(subpixel_phase - 4 * np.pi / 3)

        # Create high-frequency carrier that interacts with camera sensor
        carrier_freq = 7 + int(strength * 10)
        carrier = np.sin(2 * np.pi * X / carrier_freq) * np.cos(2 * np.pi * Y / carrier_freq)

        # Modulate subpixel patterns with carrier
        red_pattern = red_subpixel * carrier
        green_pattern = green_subpixel * carrier
        blue_pattern = blue_subpixel * carrier

        # Apply patterns
        # These look smooth on screen but cause color fringing when photographed
        pattern_strength = strength * 0.018
        protected_image[:, :, 0] += red_pattern * pattern_strength
        protected_image[:, :, 1] += green_pattern * pattern_strength
        protected_image[:, :, 2] += blue_pattern * pattern_strength

        return protected_image

    def _apply_screen_grid_interference(self, image, strength):
        """
        Create patterns that interfere with screen pixel grid

        This creates moiré patterns when camera sensor grid samples
        the screen pixel grid at non-integer ratios.

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with screen grid interference
        """
        height, width = image.shape[:2]

        y = np.arange(height)
        x = np.arange(width)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Create patterns at frequencies that create strong moiré
        # When screen at ~100 PPI is photographed, these create beating patterns

        # Pattern 1: Fine grid slightly offset from pixel grid
        grid_freq = 1.414  # Irrational number ensures moiré
        grid_h = np.sin(2 * np.pi * X / grid_freq)
        grid_v = np.sin(2 * np.pi * Y / grid_freq)
        grid_pattern = grid_h * grid_v

        # Pattern 2: Diagonal grid at golden ratio frequency
        golden_freq = 1.618  # Golden ratio - maximizes moiré
        diag_pattern = np.sin(2 * np.pi * (X + Y) / golden_freq) * \
                      np.sin(2 * np.pi * (X - Y) / golden_freq)

        # Pattern 3: Circular interference pattern
        center_y, center_x = height // 2, width // 2
        radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        radial_freq = 2.5 + strength * 1.5
        circular_pattern = np.sin(2 * np.pi * radius / radial_freq)

        # Combine patterns
        interference = (0.4 * grid_pattern +
                       0.4 * diag_pattern +
                       0.2 * circular_pattern)

        # Normalize
        interference = interference - np.mean(interference)
        if np.std(interference) > 0:
            interference = interference / (np.std(interference) * 2.0)

        # Apply to image
        protected_image = image.copy()
        pattern_strength = strength * 0.020

        # Apply with slight phase shifts between channels for maximum moiré
        protected_image[:, :, 0] += interference * pattern_strength
        protected_image[:, :, 1] += np.roll(interference, shift=1, axis=1) * pattern_strength
        protected_image[:, :, 2] += np.roll(interference, shift=2, axis=1) * pattern_strength

        return protected_image

    def _apply_rgb_stripe_pattern(self, image, strength):
        """
        Create RGB stripe patterns matching screen subpixel layout

        This exploits the horizontal RGB stripe arrangement common
        in LCD and OLED displays.

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with RGB stripe patterns
        """
        height, width = image.shape[:2]
        protected_image = image.copy()

        y = np.arange(height)
        x = np.arange(width)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Create vertical stripe patterns (perpendicular to RGB stripes)
        # These interfere with the horizontal subpixel arrangement

        # Fine vertical stripes at problematic frequency
        stripe_freq = 2.236  # sqrt(5) - creates strong moiré
        vertical_stripes = np.sin(2 * np.pi * X / stripe_freq)

        # Add horizontal modulation to create 2D moiré
        horiz_freq = 5 + int(strength * 8)
        horizontal_mod = np.sin(2 * np.pi * Y / horiz_freq)

        # Combine into checkerboard-like pattern
        stripe_pattern = vertical_stripes * (0.5 + 0.5 * horizontal_mod)

        # Create complementary pattern for different channels
        # This causes color separation when photographed
        phase_shift = 2 * np.pi / 3

        red_stripe = np.sin(2 * np.pi * X / stripe_freq)
        green_stripe = np.sin(2 * np.pi * X / stripe_freq + phase_shift)
        blue_stripe = np.sin(2 * np.pi * X / stripe_freq + 2 * phase_shift)

        # Modulate all with horizontal variation
        red_stripe *= (0.5 + 0.5 * horizontal_mod)
        green_stripe *= (0.5 + 0.5 * horizontal_mod)
        blue_stripe *= (0.5 + 0.5 * horizontal_mod)

        # Apply stripe patterns
        pattern_strength = strength * 0.018
        protected_image[:, :, 0] += red_stripe * pattern_strength
        protected_image[:, :, 1] += green_stripe * pattern_strength
        protected_image[:, :, 2] += blue_stripe * pattern_strength

        return protected_image

    def _apply_pentile_confusion(self, image, strength):
        """
        Create patterns that confuse PenTile OLED displays when photographed

        Many OLED screens use PenTile (RGBG) subpixel arrangement.
        This creates specific patterns for that layout.

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with PenTile confusion patterns
        """
        height, width = image.shape[:2]
        protected_image = image.copy()

        y = np.arange(height)
        x = np.arange(width)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # PenTile pattern (RGBG diamond arrangement):
        # Row 0:  R - G - R - G
        # Row 1:  - B - G - B -
        # Row 2:  R - G - R - G
        # (where - represents gaps)

        # Create diagonal pattern matching PenTile layout
        pentile_h = X % 2
        pentile_v = Y % 2

        # Red/Blue positions (checkerboard)
        rb_positions = pentile_h ^ pentile_v
        # Green positions (everywhere)
        g_positions = np.ones_like(X)

        # Create interference pattern at PenTile frequency
        pentile_freq = 2 * np.sqrt(2)  # Diagonal spacing
        pentile_pattern = np.sin(2 * np.pi * (X + Y) / pentile_freq) * \
                         np.cos(2 * np.pi * (X - Y) / pentile_freq)

        # Create color-specific patterns
        # Red gets enhanced at R positions
        red_enhancement = pentile_pattern * (1.0 + rb_positions * pentile_v)
        # Green gets complex modulation (it's everywhere in PenTile)
        green_enhancement = pentile_pattern * (1.0 + 0.5 * np.sin(2 * np.pi * X / 2))
        # Blue gets enhanced at B positions
        blue_enhancement = pentile_pattern * (1.0 + rb_positions * (1 - pentile_v))

        # Add high-frequency texture
        hf_freq = 20 + int(strength * 30)
        hf_texture = np.sin(2 * np.pi * X / hf_freq) * np.sin(2 * np.pi * Y / hf_freq)

        # Modulate with texture
        red_final = red_enhancement * (0.7 + 0.3 * hf_texture)
        green_final = green_enhancement * (0.7 + 0.3 * hf_texture)
        blue_final = blue_enhancement * (0.7 + 0.3 * hf_texture)

        # Normalize patterns
        for pattern in [red_final, green_final, blue_final]:
            pattern -= np.mean(pattern)
            if np.std(pattern) > 0:
                pattern /= (np.std(pattern) * 2.5)

        # Apply patterns
        pattern_strength = strength * 0.016
        protected_image[:, :, 0] += red_final * pattern_strength
        protected_image[:, :, 1] += green_final * pattern_strength
        protected_image[:, :, 2] += blue_final * pattern_strength

        return protected_image

    def analyze_moire_potential(self, image):
        """
        Analyze the moiré pattern potential of an image

        Args:
            image (numpy.ndarray): Input image

        Returns:
            dict: Moiré analysis results
        """
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)

        # Analyze each channel for moiré-inducing patterns
        results = {}

        for i, channel_name in enumerate(['red', 'green', 'blue']):
            channel = image[:, :, i]

            # Check for fine patterns (1-4 pixel frequency)
            fft = np.fft.fft2(channel)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)

            # Check high-frequency content
            h, w = magnitude.shape
            # Look at frequencies near Nyquist
            hf_region = magnitude[h//4:3*h//4, w//4:3*w//4]
            hf_energy = np.mean(hf_region)

            # Check for periodic patterns
            # Look for peaks in FFT (indicates periodic content)
            threshold = np.mean(magnitude) + 2 * np.std(magnitude)
            num_peaks = np.sum(magnitude > threshold)

            results[f'{channel_name}_hf_energy'] = hf_energy
            results[f'{channel_name}_periodic_peaks'] = num_peaks

        # Overall moiré potential score
        total_hf = sum(results[f'{ch}_hf_energy'] for ch in ['red', 'green', 'blue'])
        total_peaks = sum(results[f'{ch}_periodic_peaks'] for ch in ['red', 'green', 'blue'])

        results['moire_potential'] = total_hf + total_peaks / 100

        return results
