import numpy as np
import cv2
from scipy import fftpack
from utils.image_utils import ImageUtils

class FrequencyDomainProtection:
    """
    High-frequency noise injection protection that adds imperceptible patterns
    visible to cameras but filtered out by human vision
    """

    def __init__(self):
        self.name = "High-Frequency Noise Protection"

    def apply(self, image, strength=0.5):
        """
        Apply high-frequency noise protection to image

        Args:
            image (numpy.ndarray): Input image in RGB format [0, 1]
            strength (float): Protection strength [0.1, 1.0]

        Returns:
            numpy.ndarray: Protected image
        """
        ImageUtils.validate_image(image)

        # Normalize strength
        strength = np.clip(strength, 0.1, 1.0)

        # Process each channel separately for color images
        if len(image.shape) == 3:
            protected_channels = []
            for channel in range(image.shape[2]):
                protected_channel = self._apply_frequency_protection(
                    image[:, :, channel], strength
                )
                protected_channels.append(protected_channel)

            protected_image = np.stack(protected_channels, axis=2)
        else:
            # Grayscale image
            protected_image = self._apply_frequency_protection(image, strength)

        return np.clip(protected_image, 0, 1)

    def _apply_frequency_protection(self, channel, strength):
        """
        Apply frequency domain protection to a single channel

        Args:
            channel (numpy.ndarray): Single channel image
            strength (float): Protection strength

        Returns:
            numpy.ndarray: Protected channel
        """
        # Get image dimensions
        height, width = channel.shape

        # Apply 2D FFT
        fft_image = fftpack.fft2(channel)
        fft_shifted = fftpack.fftshift(fft_image)

        # Create frequency domain coordinates
        u = np.arange(height) - height // 2
        v = np.arange(width) - width // 2
        U, V = np.meshgrid(v, u)

        # Calculate distance from center (frequency magnitude)
        D = np.sqrt(U**2 + V**2)

        # Normalize distance
        D_normalized = D / (min(height, width) // 2)

        # Create high-frequency emphasis filter
        # This amplifies higher frequencies while preserving low frequencies
        high_freq_filter = self._create_high_frequency_filter(
            D_normalized, strength
        )

        # Generate structured noise pattern that cameras detect
        noise_pattern = self._generate_camera_visible_noise(
            height, width, strength
        )

        # Apply noise in frequency domain
        fft_noise = fftpack.fft2(noise_pattern)
        fft_noise_shifted = fftpack.fftshift(fft_noise)

        # Combine original image with noise using the filter
        # Balanced strength for imperceptibility to humans while resisting iPhone 16 AI processing
        protected_fft = fft_shifted + (fft_noise_shifted * high_freq_filter * strength * 0.025)

        # Convert back to spatial domain
        protected_fft_unshifted = fftpack.ifftshift(protected_fft)
        protected_channel = np.real(fftpack.ifft2(protected_fft_unshifted))

        return protected_channel

    def _create_high_frequency_filter(self, D_normalized, strength):
        """
        Create a filter that emphasizes high frequencies

        Args:
            D_normalized (numpy.ndarray): Normalized distance matrix
            strength (float): Filter strength

        Returns:
            numpy.ndarray: High-frequency emphasis filter
        """
        # Create a filter that increases with frequency
        # But tapers off at very high frequencies to avoid artifacts

        # Sigmoid-based high-pass characteristics
        cutoff = 0.3  # Start emphasizing after 30% of max frequency
        slope = 10 * strength  # Steepness of transition

        # High-frequency emphasis with rolloff
        filter_response = 1 / (1 + np.exp(-slope * (D_normalized - cutoff)))

        # Add slight attenuation at very high frequencies to prevent aliasing
        very_high_freq_cutoff = 0.8
        very_high_freq_filter = 1 / (1 + np.exp(slope * (D_normalized - very_high_freq_cutoff)))

        # Combine filters
        combined_filter = filter_response * very_high_freq_filter

        return combined_filter

    def _generate_camera_visible_noise(self, height, width, strength):
        """
        Generate noise pattern specifically designed to be visible to cameras
        but imperceptible to human eyes

        Args:
            height (int): Image height
            width (int): Image width
            strength (float): Noise strength

        Returns:
            numpy.ndarray: Camera-visible noise pattern
        """
        # Create multiple frequency components that cameras detect
        noise = np.zeros((height, width))

        # High-frequency checkerboard pattern (camera sensors are sensitive to this)
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Multiple high-frequency patterns
        # Pattern 1: High-frequency checkerboard
        freq1 = 20 + int(strength * 30)  # Frequency based on strength
        pattern1 = np.sin(2 * np.pi * X / freq1) * np.cos(2 * np.pi * Y / freq1)

        # Pattern 2: Diagonal high-frequency waves
        freq2 = 15 + int(strength * 25)
        pattern2 = np.sin(2 * np.pi * (X + Y) / freq2)

        # Pattern 3: Circular high-frequency patterns
        center_x, center_y = width // 2, height // 2
        R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        freq3 = 10 + int(strength * 20)
        pattern3 = np.sin(2 * np.pi * R / freq3)

        # Pattern 4: Random high-frequency noise
        np.random.seed(42)  # For reproducibility
        pattern4 = np.random.normal(0, 0.1, (height, width))

        # High-pass filter the random noise
        pattern4_fft = fftpack.fft2(pattern4)
        pattern4_fft_shifted = fftpack.fftshift(pattern4_fft)

        # Create high-pass filter
        u = np.arange(height) - height // 2
        v = np.arange(width) - width // 2
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        D_normalized = D / (min(height, width) // 2)

        high_pass = np.where(D_normalized > 0.3, 1, 0)
        pattern4_filtered_fft = pattern4_fft_shifted * high_pass
        pattern4_filtered = np.real(fftpack.ifft2(fftpack.ifftshift(pattern4_filtered_fft)))

        # Combine patterns with different weights
        noise = (0.3 * pattern1 +
                0.25 * pattern2 +
                0.2 * pattern3 +
                0.25 * pattern4_filtered)

        # Scale noise amplitude
        # Balanced for imperceptibility to humans while resisting iPhone 16 AI processing
        noise_amplitude = strength * 0.012
        noise = noise * noise_amplitude

        return noise

    def _apply_anti_aliasing(self, image):
        """
        Apply anti-aliasing to prevent visible artifacts

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Anti-aliased image
        """
        # Apply a slight Gaussian blur to reduce aliasing artifacts
        kernel_size = 3
        sigma = 0.5

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def create_test_pattern(self, width=512, height=512, strength=0.5):
        """
        Create a test pattern to visualize the protection effect

        Args:
            width (int): Pattern width
            height (int): Pattern height
            strength (float): Protection strength

        Returns:
            tuple: (original_pattern, protected_pattern)
        """
        # Create a simple test image with various frequency components
        x = np.linspace(0, 2 * np.pi, width)
        y = np.linspace(0, 2 * np.pi, height)
        X, Y = np.meshgrid(x, y)

        # Create test pattern with low and medium frequencies
        original = (np.sin(X) * np.cos(Y) +
                   np.sin(3 * X) * np.cos(3 * Y) * 0.5 +
                   np.sin(0.5 * X) * np.cos(0.5 * Y) * 0.3)

        # Normalize to [0, 1]
        original = (original - original.min()) / (original.max() - original.min())

        # Apply protection
        protected = self._apply_frequency_protection(original, strength)
        protected = np.clip(protected, 0, 1)

        return original, protected

    def analyze_frequency_content(self, image):
        """
        Analyze the frequency content of an image

        Args:
            image (numpy.ndarray): Input image

        Returns:
            dict: Frequency analysis results
        """
        if len(image.shape) == 3:
            # Convert to grayscale for analysis
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Apply FFT
        fft_image = fftpack.fft2(gray)
        fft_magnitude = np.abs(fft_image)

        # Calculate frequency distribution
        height, width = gray.shape
        center_y, center_x = height // 2, width // 2

        # Create distance matrix from center
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Calculate power spectrum in frequency bands
        max_distance = min(height, width) // 2
        low_freq_mask = distances <= max_distance * 0.1
        mid_freq_mask = (distances > max_distance * 0.1) & (distances <= max_distance * 0.5)
        high_freq_mask = distances > max_distance * 0.5

        low_freq_power = np.mean(fft_magnitude[low_freq_mask])
        mid_freq_power = np.mean(fft_magnitude[mid_freq_mask])
        high_freq_power = np.mean(fft_magnitude[high_freq_mask])

        total_power = low_freq_power + mid_freq_power + high_freq_power

        return {
            'low_freq_ratio': low_freq_power / total_power,
            'mid_freq_ratio': mid_freq_power / total_power,
            'high_freq_ratio': high_freq_power / total_power,
            'total_power': total_power
        }