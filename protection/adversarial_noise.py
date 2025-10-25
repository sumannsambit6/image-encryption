import numpy as np
import cv2
from utils.image_utils import ImageUtils

class AdversarialNoiseProtection:
    """
    Adversarial noise generation that confuses camera auto-focus, exposure,
    and white balance systems while remaining imperceptible to human vision
    """

    def __init__(self):
        self.name = "Adversarial Camera Interference Protection"

    def apply(self, image, strength=0.5):
        """
        Apply adversarial noise protection to image

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

        # Apply different adversarial techniques
        protected_image = self._apply_autofocus_confusion(protected_image, strength)
        protected_image = self._apply_exposure_interference(protected_image, strength)
        protected_image = self._apply_white_balance_disruption(protected_image, strength)
        protected_image = self._apply_edge_enhancement_confusion(protected_image, strength)
        protected_image = self._apply_motion_blur_simulation(protected_image, strength)

        return np.clip(protected_image, 0, 1)

    def _apply_autofocus_confusion(self, image, strength):
        """
        Add patterns that confuse camera autofocus systems

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with autofocus confusion patterns
        """
        height, width = image.shape[:2]

        # Create false edge patterns that mislead autofocus
        false_edges = self._generate_false_edges(height, width, strength)

        # Create contrast variations that confuse phase-detection AF
        contrast_noise = self._generate_contrast_noise(height, width, strength)

        # Combine patterns
        af_confusion = false_edges + contrast_noise

        # Apply to image with strong interference for camera systems
        # Autofocus systems are sensitive to edge information
        edge_weight = strength * 0.012
        protected_image = image.copy()

        # Apply stronger to green channel (often used for AF)
        protected_image[:, :, 1] += af_confusion * edge_weight * 1.2
        protected_image[:, :, 0] += af_confusion * edge_weight * 0.8
        protected_image[:, :, 2] += af_confusion * edge_weight * 0.6

        return protected_image

    def _generate_false_edges(self, height, width, strength):
        """
        Generate false edge patterns that mislead autofocus

        Args:
            height, width (int): Image dimensions
            strength (float): Pattern strength

        Returns:
            numpy.ndarray: False edge pattern
        """
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Create multiple frequency components for false edges
        # High frequency patterns that look like edges to AF systems
        freq1 = 30 + int(strength * 50)
        freq2 = 25 + int(strength * 40)
        freq3 = 40 + int(strength * 60)

        # Horizontal "edges"
        horizontal_edges = np.sin(2 * np.pi * Y / freq1) * np.exp(-np.abs(Y - height/2) / (height * 0.3))

        # Vertical "edges"
        vertical_edges = np.sin(2 * np.pi * X / freq2) * np.exp(-np.abs(X - width/2) / (width * 0.3))

        # Diagonal "edges"
        diagonal_edges = np.sin(2 * np.pi * (X + Y) / freq3) * np.sin(2 * np.pi * (X - Y) / freq3)

        # Combine edge patterns
        false_edges = (0.4 * horizontal_edges +
                      0.4 * vertical_edges +
                      0.2 * diagonal_edges)

        # Normalize
        false_edges = false_edges - np.mean(false_edges)
        if np.std(false_edges) > 0:
            false_edges = false_edges / np.std(false_edges)

        return false_edges

    def _generate_contrast_noise(self, height, width, strength):
        """
        Generate contrast variations that confuse phase-detection AF

        Args:
            height, width (int): Image dimensions
            strength (float): Pattern strength

        Returns:
            numpy.ndarray: Contrast noise pattern
        """
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Create periodic contrast variations
        contrast_freq = 20 + int(strength * 35)
        contrast_pattern = np.sin(2 * np.pi * X / contrast_freq) * np.cos(2 * np.pi * Y / contrast_freq)

        # Add random noise component
        np.random.seed(42)
        random_noise = np.random.normal(0, 0.1, (height, width))

        # Apply spatial filtering to create correlated noise
        kernel = np.ones((3, 3)) / 9
        filtered_noise = cv2.filter2D(random_noise, -1, kernel)

        # Combine periodic and random components
        contrast_noise = 0.6 * contrast_pattern + 0.4 * filtered_noise

        # Normalize
        contrast_noise = contrast_noise - np.mean(contrast_noise)
        if np.std(contrast_noise) > 0:
            contrast_noise = contrast_noise / np.std(contrast_noise)

        return contrast_noise

    def _apply_exposure_interference(self, image, strength):
        """
        Add patterns that interfere with camera exposure systems

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with exposure interference
        """
        height, width = image.shape[:2]

        # Create luminance variations that confuse light metering
        exposure_noise = self._generate_exposure_confusion(height, width, strength)

        # Apply exposure interference
        # Affects all channels to influence overall brightness perception
        interference_weight = strength * 0.010

        protected_image = image.copy()
        for channel in range(3):
            protected_image[:, :, channel] += exposure_noise * interference_weight

        return protected_image

    def _generate_exposure_confusion(self, height, width, strength):
        """
        Generate patterns that confuse camera exposure metering

        Args:
            height, width (int): Image dimensions
            strength (float): Pattern strength

        Returns:
            numpy.ndarray: Exposure confusion pattern
        """
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Create patterns that simulate bright/dark areas
        # These mislead spot metering and matrix metering systems

        # Radial brightness variation
        center_x, center_y = width // 2, height // 2
        radial_dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        radial_pattern = np.sin(2 * np.pi * radial_dist / (max_dist * 0.3))

        # Grid-based brightness variation
        grid_freq = 15 + int(strength * 25)
        grid_pattern = (np.sin(2 * np.pi * X / grid_freq) *
                       np.sin(2 * np.pi * Y / grid_freq))

        # Asymmetric brightness pattern
        asymmetric_freq = 8 + int(strength * 15)
        asymmetric_pattern = (np.sin(2 * np.pi * X / (width / asymmetric_freq)) *
                            np.cos(2 * np.pi * Y / (height / asymmetric_freq)))

        # Combine patterns
        exposure_pattern = (0.4 * radial_pattern +
                          0.35 * grid_pattern +
                          0.25 * asymmetric_pattern)

        # Normalize
        exposure_pattern = exposure_pattern - np.mean(exposure_pattern)
        if np.std(exposure_pattern) > 0:
            exposure_pattern = exposure_pattern / np.std(exposure_pattern)

        return exposure_pattern

    def _apply_white_balance_disruption(self, image, strength):
        """
        Apply color cast patterns that disrupt white balance

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with white balance disruption
        """
        height, width = image.shape[:2]

        # Generate color cast patterns
        wb_disruption = self._generate_white_balance_confusion(height, width, strength)

        # Apply different color casts to different channels
        wb_weight = strength * 0.010

        protected_image = image.copy()

        # Red cast pattern (affects warm/cool balance)
        red_cast_1d = np.sin(np.pi * np.arange(width) / width)
        red_cast_2d = np.tile(red_cast_1d, (height, 1))
        protected_image[:, :, 0] += wb_disruption * red_cast_2d * wb_weight * 1.2

        # Blue cast pattern (complementary to red)
        blue_cast_1d = np.cos(np.pi * np.arange(width) / width)
        blue_cast_2d = np.tile(blue_cast_1d, (height, 1))
        protected_image[:, :, 2] += wb_disruption * blue_cast_2d * wb_weight * 1.0

        # Green cast pattern (affects magenta/green balance)
        green_cast_1d = np.sin(2 * np.pi * np.arange(height) / height)
        green_cast_2d = np.tile(green_cast_1d.reshape(-1, 1), (1, width))
        protected_image[:, :, 1] += wb_disruption * green_cast_2d * wb_weight * 0.8

        return protected_image

    def _generate_white_balance_confusion(self, height, width, strength):
        """
        Generate patterns that confuse white balance algorithms

        Args:
            height, width (int): Image dimensions
            strength (float): Pattern strength

        Returns:
            numpy.ndarray: White balance confusion pattern
        """
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Create color temperature variation patterns
        temp_freq = 10 + int(strength * 20)
        temp_variation = np.sin(2 * np.pi * X / (width / temp_freq))

        # Create tint variation patterns
        tint_freq = 12 + int(strength * 18)
        tint_variation = np.cos(2 * np.pi * Y / (height / tint_freq))

        # Combine temperature and tint variations
        wb_pattern = 0.6 * temp_variation + 0.4 * tint_variation

        # Add some spatial correlation
        kernel = np.ones((5, 5)) / 25
        wb_pattern = cv2.filter2D(wb_pattern, -1, kernel)

        # Normalize
        wb_pattern = wb_pattern - np.mean(wb_pattern)
        if np.std(wb_pattern) > 0:
            wb_pattern = wb_pattern / np.std(wb_pattern)

        return wb_pattern

    def _apply_edge_enhancement_confusion(self, image, strength):
        """
        Apply patterns that confuse edge enhancement algorithms

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with edge enhancement confusion
        """
        height, width = image.shape[:2]

        # Generate false edge enhancement cues
        edge_confusion = self._generate_edge_confusion(height, width, strength)

        # Apply to image
        edge_weight = strength * 0.010

        protected_image = image.copy()

        # Apply primarily to luminance-related channels
        # Green channel (highest luminance contribution)
        protected_image[:, :, 1] += edge_confusion * edge_weight * 1.0

        # Red and blue channels (lower contribution)
        protected_image[:, :, 0] += edge_confusion * edge_weight * 0.7
        protected_image[:, :, 2] += edge_confusion * edge_weight * 0.5

        return protected_image

    def _generate_edge_confusion(self, height, width, strength):
        """
        Generate patterns that confuse edge enhancement

        Args:
            height, width (int): Image dimensions
            strength (float): Pattern strength

        Returns:
            numpy.ndarray: Edge confusion pattern
        """
        # Create high-frequency patterns that mimic edges
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Multiple frequency edge-like patterns
        freq1 = 35 + int(strength * 45)
        freq2 = 28 + int(strength * 38)
        freq3 = 42 + int(strength * 52)

        # Create patterns that look like edges to algorithms
        pattern1 = np.sin(2 * np.pi * X / freq1) * np.exp(-np.abs(X - width/2) / (width * 0.2))
        pattern2 = np.sin(2 * np.pi * Y / freq2) * np.exp(-np.abs(Y - height/2) / (height * 0.2))
        pattern3 = np.sin(2 * np.pi * (X + Y) / freq3)

        # Combine patterns
        edge_pattern = 0.4 * pattern1 + 0.4 * pattern2 + 0.2 * pattern3

        # Apply edge-detection-like filtering
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        edge_x = cv2.filter2D(edge_pattern, -1, sobel_x)
        edge_y = cv2.filter2D(edge_pattern, -1, sobel_y)
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)

        # Normalize
        edge_magnitude = edge_magnitude - np.mean(edge_magnitude)
        if np.std(edge_magnitude) > 0:
            edge_magnitude = edge_magnitude / np.std(edge_magnitude)

        return edge_magnitude

    def _apply_motion_blur_simulation(self, image, strength):
        """
        Apply patterns that simulate motion for stabilization confusion

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with motion blur simulation
        """
        height, width = image.shape[:2]

        # Generate subtle motion patterns
        motion_pattern = self._generate_motion_pattern(height, width, strength)

        # Apply motion simulation
        motion_weight = strength * 0.008

        protected_image = image.copy()

        # Apply motion patterns that confuse stabilization
        for channel in range(3):
            protected_image[:, :, channel] += motion_pattern * motion_weight

        return protected_image

    def _generate_motion_pattern(self, height, width, strength):
        """
        Generate motion-like patterns

        Args:
            height, width (int): Image dimensions
            strength (float): Pattern strength

        Returns:
            numpy.ndarray: Motion pattern
        """
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Create directional patterns that simulate motion
        motion_freq = 20 + int(strength * 30)

        # Horizontal motion pattern
        horizontal_motion = np.sin(2 * np.pi * X / motion_freq) * np.exp(-np.abs(Y - height/2) / (height * 0.4))

        # Vertical motion pattern
        vertical_motion = np.sin(2 * np.pi * Y / motion_freq) * np.exp(-np.abs(X - width/2) / (width * 0.4))

        # Rotational motion pattern
        center_x, center_y = width // 2, height // 2
        angle = np.arctan2(Y - center_y, X - center_x)
        rotational_motion = np.sin(8 * angle) * np.exp(-np.sqrt((X - center_x)**2 + (Y - center_y)**2) / (min(width, height) * 0.3))

        # Combine motion patterns
        motion_pattern = (0.4 * horizontal_motion +
                         0.4 * vertical_motion +
                         0.2 * rotational_motion)

        # Normalize
        motion_pattern = motion_pattern - np.mean(motion_pattern)
        if np.std(motion_pattern) > 0:
            motion_pattern = motion_pattern / np.std(motion_pattern)

        return motion_pattern

    def analyze_adversarial_content(self, image):
        """
        Analyze the adversarial content of an image

        Args:
            image (numpy.ndarray): Input image

        Returns:
            dict: Adversarial analysis results
        """
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)

        # Calculate edge content (affects autofocus)
        gray = np.mean(image, axis=2)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_content = np.mean(edge_magnitude)

        # Calculate color variation (affects white balance)
        color_std = np.mean([np.std(image[:, :, i]) for i in range(3)])

        # Calculate high-frequency energy
        fft_image = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft_image)
        high_freq_energy = np.mean(fft_magnitude[fft_magnitude.shape[0]//4:3*fft_magnitude.shape[0]//4,
                                                fft_magnitude.shape[1]//4:3*fft_magnitude.shape[1]//4])

        # Calculate brightness variation (affects exposure)
        brightness_std = np.std(gray)

        return {
            'edge_content': edge_content,
            'color_variation': color_std,
            'high_freq_energy': high_freq_energy,
            'brightness_variation': brightness_std,
            'adversarial_score': edge_content + color_std + high_freq_energy + brightness_std
        }

    def create_adversarial_test_pattern(self, width=512, height=512, strength=0.5):
        """
        Create a test pattern to visualize adversarial effects

        Args:
            width, height (int): Pattern dimensions
            strength (float): Protection strength

        Returns:
            tuple: (original_pattern, protected_pattern)
        """
        # Create a neutral test pattern
        x = np.linspace(0, 2 * np.pi, width)
        y = np.linspace(0, 2 * np.pi, height)
        X, Y = np.meshgrid(x, y)

        # Create a simple gradient pattern
        red_pattern = (X / (2 * np.pi) + Y / (2 * np.pi)) / 2
        green_pattern = np.ones_like(red_pattern) * 0.5
        blue_pattern = 1 - red_pattern

        original = np.stack([red_pattern, green_pattern, blue_pattern], axis=2)

        # Apply adversarial protection
        protected = self.apply(original, strength)

        return original, protected