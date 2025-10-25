import numpy as np
import cv2
from utils.image_utils import ImageUtils

class InfraredPatternProtection:
    """
    Infrared spectrum manipulation protection that exploits differences
    between human vision (400-700nm) and camera sensors (350-1000nm)
    """

    def __init__(self):
        self.name = "Infrared Spectrum Manipulation Protection"

    def apply(self, image, strength=0.5):
        """
        Apply infrared spectrum manipulation to image

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
            # Convert grayscale to RGB
            image = np.stack([image, image, image], axis=2)

        protected_image = image.copy()

        # Apply different IR manipulation techniques
        protected_image = self._apply_channel_shifting(protected_image, strength)
        protected_image = self._apply_ir_noise_patterns(protected_image, strength)
        protected_image = self._apply_spectral_confusion(protected_image, strength)
        protected_image = self._apply_near_ir_simulation(protected_image, strength)

        return np.clip(protected_image, 0, 1)

    def _apply_channel_shifting(self, image, strength):
        """
        Apply subtle channel shifting that cameras detect but humans don't see

        Args:
            image (numpy.ndarray): Input RGB image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with channel shifting
        """
        # Create a copy to work with
        shifted_image = image.copy()

        # Get image dimensions
        height, width = image.shape[:2]

        # Create spatial coordinates
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Generate subtle shifting patterns
        # These patterns are designed to be invisible to human eyes
        # but detectable by camera sensors in near-IR range

        # Pattern 1: Micro-level red channel enhancement
        # Simulates near-IR sensitivity of camera sensors
        red_enhancement = self._generate_micro_pattern(X, Y, strength, pattern_type='red_ir')
        shifted_image[:, :, 0] += red_enhancement * strength * 0.015

        # Pattern 2: Blue channel attenuation pattern
        # Cameras often have different blue vs IR sensitivity
        blue_attenuation = self._generate_micro_pattern(X, Y, strength, pattern_type='blue_ir')
        shifted_image[:, :, 2] -= blue_attenuation * strength * 0.010

        # Pattern 3: Green channel modulation
        # Fine-tune green channel for IR confusion
        green_modulation = self._generate_micro_pattern(X, Y, strength, pattern_type='green_ir')
        shifted_image[:, :, 1] += green_modulation * strength * 0.008

        return shifted_image

    def _generate_micro_pattern(self, X, Y, strength, pattern_type='red_ir'):
        """
        Generate micro-level patterns for IR simulation

        Args:
            X, Y (numpy.ndarray): Coordinate matrices
            strength (float): Pattern strength
            pattern_type (str): Type of pattern to generate

        Returns:
            numpy.ndarray: Generated pattern
        """
        height, width = X.shape

        if pattern_type == 'red_ir':
            # Pattern that simulates near-IR reflection
            # High frequency pattern with radial component
            freq = 30 + int(strength * 50)
            radial_dist = np.sqrt((X - width/2)**2 + (Y - height/2)**2)
            pattern = (np.sin(2 * np.pi * X / freq) *
                      np.cos(2 * np.pi * Y / freq) *
                      np.exp(-radial_dist / (width * 0.5)))

        elif pattern_type == 'blue_ir':
            # Pattern for blue channel IR attenuation
            freq = 25 + int(strength * 40)
            pattern = (np.sin(2 * np.pi * (X + Y) / freq) *
                      np.sin(2 * np.pi * (X - Y) / (freq * 1.5)))

        elif pattern_type == 'green_ir':
            # Pattern for green channel IR modulation
            freq1 = 35 + int(strength * 30)
            freq2 = 20 + int(strength * 25)
            pattern = (np.sin(2 * np.pi * X / freq1) *
                      np.cos(2 * np.pi * Y / freq2) * 0.5 +
                      np.sin(2 * np.pi * Y / freq1) *
                      np.cos(2 * np.pi * X / freq2) * 0.5)

        else:
            # Default random pattern
            np.random.seed(42)
            pattern = np.random.normal(0, 0.1, (height, width))

        # Normalize pattern
        pattern = pattern - np.mean(pattern)
        if np.std(pattern) > 0:
            pattern = pattern / np.std(pattern)

        return pattern

    def _apply_ir_noise_patterns(self, image, strength):
        """
        Apply IR-specific noise patterns that cameras detect

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with IR noise patterns
        """
        height, width = image.shape[:2]

        # Create IR-like noise that affects different wavelengths differently
        ir_noise = self._generate_ir_noise(height, width, strength)

        # Apply noise with wavelength-dependent strength
        # Red channel: highest IR sensitivity
        image[:, :, 0] += ir_noise * strength * 0.018

        # Green channel: medium IR sensitivity
        image[:, :, 1] += ir_noise * strength * 0.012

        # Blue channel: lowest IR sensitivity
        image[:, :, 2] += ir_noise * strength * 0.008

        return image

    def _generate_ir_noise(self, height, width, strength):
        """
        Generate noise that simulates IR interference

        Args:
            height, width (int): Image dimensions
            strength (float): Noise strength

        Returns:
            numpy.ndarray: IR noise pattern
        """
        # Create multiple frequency components for IR simulation
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Component 1: High-frequency thermal-like noise
        thermal_freq = 40 + int(strength * 60)
        thermal_noise = np.sin(2 * np.pi * X / thermal_freq) * np.cos(2 * np.pi * Y / thermal_freq)

        # Component 2: Medium-frequency IR reflection pattern
        reflection_freq = 15 + int(strength * 30)
        reflection_noise = np.sin(2 * np.pi * (X + Y) / reflection_freq)

        # Component 3: Low-frequency IR emission simulation
        emission_freq = 8 + int(strength * 15)
        center_x, center_y = width // 2, height // 2
        emission_noise = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (width * height * 0.1)) * \
                        np.sin(2 * np.pi * np.sqrt((X - center_x)**2 + (Y - center_y)**2) / emission_freq)

        # Combine components
        ir_noise = (0.4 * thermal_noise +
                   0.35 * reflection_noise +
                   0.25 * emission_noise)

        # Normalize
        ir_noise = ir_noise - np.mean(ir_noise)
        if np.std(ir_noise) > 0:
            ir_noise = ir_noise / np.std(ir_noise)

        # Scale amplitude
        amplitude = strength * 0.01
        return ir_noise * amplitude

    def _apply_spectral_confusion(self, image, strength):
        """
        Apply spectral confusion patterns that exploit camera sensor characteristics

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with spectral confusion
        """
        # Create spectral confusion by modifying color relationships
        # that cameras interpret differently than human eyes

        # Technique 1: Metameric color shifts
        # Create colors that look the same to humans but different to cameras
        metamer_shift = self._create_metameric_shift(image.shape[:2], strength)

        # Apply metameric shifts differently to each channel
        image[:, :, 0] += metamer_shift * strength * 0.012  # Red
        image[:, :, 1] -= metamer_shift * strength * 0.008  # Green
        image[:, :, 2] += metamer_shift * strength * 0.008  # Blue

        # Technique 2: Color temperature confusion
        temp_confusion = self._create_color_temperature_confusion(image.shape[:2], strength)

        # Apply temperature-based shifts
        image[:, :, 0] += temp_confusion * strength * 0.010
        image[:, :, 2] -= temp_confusion * strength * 0.010

        return image

    def _create_metameric_shift(self, shape, strength):
        """
        Create metameric color shifts

        Args:
            shape (tuple): Image shape (height, width)
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Metameric shift pattern
        """
        height, width = shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Create a complex pattern that simulates metameric differences
        freq1 = 12 + int(strength * 20)
        freq2 = 18 + int(strength * 25)

        shift_pattern = (np.sin(2 * np.pi * X / freq1) * np.sin(2 * np.pi * Y / freq2) +
                        np.cos(2 * np.pi * (X + Y) / (freq1 * 1.5)) * 0.7)

        # Normalize
        shift_pattern = shift_pattern - np.mean(shift_pattern)
        if np.std(shift_pattern) > 0:
            shift_pattern = shift_pattern / np.std(shift_pattern)

        return shift_pattern

    def _create_color_temperature_confusion(self, shape, strength):
        """
        Create color temperature confusion patterns

        Args:
            shape (tuple): Image shape (height, width)
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Color temperature confusion pattern
        """
        height, width = shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Create gradual temperature shifts
        temp_freq = 5 + int(strength * 10)
        temp_pattern = np.sin(2 * np.pi * X / (width / temp_freq)) * np.sin(2 * np.pi * Y / (height / temp_freq))

        # Add some randomness
        np.random.seed(42)
        random_component = np.random.normal(0, 0.1, shape)

        # Apply low-pass filter to random component
        kernel = np.ones((5, 5)) / 25
        random_filtered = cv2.filter2D(random_component, -1, kernel)

        combined_pattern = 0.7 * temp_pattern + 0.3 * random_filtered

        # Normalize
        combined_pattern = combined_pattern - np.mean(combined_pattern)
        if np.std(combined_pattern) > 0:
            combined_pattern = combined_pattern / np.std(combined_pattern)

        return combined_pattern

    def _apply_near_ir_simulation(self, image, strength):
        """
        Simulate near-infrared characteristics that cameras detect

        Args:
            image (numpy.ndarray): Input image
            strength (float): Effect strength

        Returns:
            numpy.ndarray: Image with near-IR simulation
        """
        # Simulate how materials reflect differently in near-IR
        # This creates subtle differences that cameras pick up

        # Calculate luminance for material-based IR simulation
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

        # Create IR reflection pattern based on material properties
        ir_reflection = self._simulate_material_ir_properties(luminance, strength)

        # Apply IR effects
        # Materials with different IR properties show up differently on cameras
        ir_enhancement = ir_reflection * strength * 0.016

        # Red channel gets most IR enhancement
        image[:, :, 0] += ir_enhancement * 1.0

        # Green channel gets moderate enhancement
        image[:, :, 1] += ir_enhancement * 0.6

        # Blue channel gets minimal enhancement
        image[:, :, 2] += ir_enhancement * 0.2

        return image

    def _simulate_material_ir_properties(self, luminance, strength):
        """
        Simulate how different materials reflect IR differently

        Args:
            luminance (numpy.ndarray): Image luminance
            strength (float): Effect strength

        Returns:
            numpy.ndarray: IR property simulation
        """
        height, width = luminance.shape

        # Create patterns that vary with material type (approximated by luminance)
        # Dark areas (low luminance): often absorb more IR
        # Light areas (high luminance): often reflect more IR

        # Base IR reflection based on luminance
        base_ir = np.tanh(luminance * 2 - 1)  # Sigmoid-like mapping

        # Add spatial variation for material texture
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # High-frequency texture that affects IR properties
        texture_freq = 25 + int(strength * 40)
        texture_ir = np.sin(2 * np.pi * X / texture_freq) * np.cos(2 * np.pi * Y / texture_freq) * 0.3

        # Combine base IR with texture
        material_ir = base_ir + texture_ir

        # Add some randomness for natural variation
        np.random.seed(42)
        random_variation = np.random.normal(0, 0.05, luminance.shape)

        # Apply slight blur to random variation
        kernel = np.ones((3, 3)) / 9
        random_variation = cv2.filter2D(random_variation, -1, kernel)

        final_ir = material_ir + random_variation

        # Normalize
        final_ir = final_ir - np.mean(final_ir)
        if np.std(final_ir) > 0:
            final_ir = final_ir / np.std(final_ir)

        return final_ir

    def analyze_ir_content(self, image):
        """
        Analyze the IR-like content of an image

        Args:
            image (numpy.ndarray): Input image

        Returns:
            dict: IR analysis results
        """
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)

        # Calculate channel differences that indicate IR characteristics
        red_dominant = np.mean(image[:, :, 0] - image[:, :, 2])
        green_balance = np.mean(image[:, :, 1] - (image[:, :, 0] + image[:, :, 2]) / 2)

        # Calculate high-frequency content in red channel (IR indicator)
        red_channel = image[:, :, 0]
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq_red = cv2.filter2D(red_channel, -1, kernel)
        red_high_freq_energy = np.mean(np.abs(high_freq_red))

        return {
            'red_dominance': red_dominant,
            'green_balance': green_balance,
            'red_high_freq_energy': red_high_freq_energy,
            'estimated_ir_enhancement': red_dominant + red_high_freq_energy
        }

    def create_ir_test_pattern(self, width=512, height=512, strength=0.5):
        """
        Create a test pattern to visualize IR protection effects

        Args:
            width, height (int): Pattern dimensions
            strength (float): Protection strength

        Returns:
            tuple: (original_pattern, protected_pattern)
        """
        # Create a test pattern with various elements
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        X, Y = np.meshgrid(x, y)

        # Create RGB test pattern
        red_pattern = (np.sin(X) + 1) / 2
        green_pattern = (np.sin(Y) + 1) / 2
        blue_pattern = (np.sin(X + Y) + 1) / 2

        original = np.stack([red_pattern, green_pattern, blue_pattern], axis=2)

        # Apply IR protection
        protected = self.apply(original, strength)

        return original, protected