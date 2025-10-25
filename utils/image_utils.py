import cv2
import numpy as np
from PIL import Image
import os

class ImageUtils:
    """Utility class for image loading, saving, and validation"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    @staticmethod
    def load_image(file_path):
        """
        Load an image from file path and return as numpy array

        Args:
            file_path (str): Path to image file

        Returns:
            numpy.ndarray: Image as numpy array in RGB format

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
            Exception: If image cannot be loaded
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ImageUtils.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")

        try:
            # Load image using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(file_path)
                image = np.array(pil_image.convert('RGB'))
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Ensure image is in float format [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0

            return image

        except Exception as e:
            raise Exception(f"Failed to load image {file_path}: {str(e)}")

    @staticmethod
    def save_image(image, file_path, quality=95):
        """
        Save numpy array image to file

        Args:
            image (numpy.ndarray): Image array in RGB format
            file_path (str): Output file path
            quality (int): JPEG quality (0-100), ignored for PNG

        Raises:
            ValueError: If image format is invalid
            Exception: If image cannot be saved
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Convert to uint8 if needed
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Clamp values to [0, 1] and convert to [0, 255]
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)

            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()

            if ext in {'.jpg', '.jpeg'}:
                # Save as JPEG using OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

            elif ext == '.png':
                # Save as PNG using OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])

            else:
                # Use PIL for other formats
                pil_image = Image.fromarray(image, 'RGB')
                if ext in {'.jpg', '.jpeg'}:
                    pil_image.save(file_path, 'JPEG', quality=quality, optimize=True)
                else:
                    pil_image.save(file_path)

        except Exception as e:
            raise Exception(f"Failed to save image to {file_path}: {str(e)}")

    @staticmethod
    def validate_image(image):
        """
        Validate that image is a proper numpy array

        Args:
            image (numpy.ndarray): Image to validate

        Returns:
            bool: True if valid

        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")

        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValueError("Color image must have 1, 3, or 4 channels")

        if image.size == 0:
            raise ValueError("Image is empty")

        return True

    @staticmethod
    def normalize_image(image):
        """
        Normalize image to [0, 1] range

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Normalized image
        """
        ImageUtils.validate_image(image)

        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype in [np.float32, np.float64]:
            return np.clip(image, 0, 1).astype(np.float32)
        else:
            # Convert to float and normalize by max value
            image_float = image.astype(np.float32)
            if image_float.max() > 1:
                image_float = image_float / image_float.max()
            return np.clip(image_float, 0, 1)

    @staticmethod
    def denormalize_image(image):
        """
        Convert normalized image [0, 1] to uint8 [0, 255]

        Args:
            image (numpy.ndarray): Normalized image

        Returns:
            numpy.ndarray: Uint8 image
        """
        ImageUtils.validate_image(image)
        image_normalized = np.clip(image, 0, 1)
        return (image_normalized * 255).astype(np.uint8)

    @staticmethod
    def resize_image(image, target_size, maintain_aspect=True):
        """
        Resize image to target size

        Args:
            image (numpy.ndarray): Input image
            target_size (tuple): (width, height)
            maintain_aspect (bool): Whether to maintain aspect ratio

        Returns:
            numpy.ndarray: Resized image
        """
        ImageUtils.validate_image(image)

        height, width = image.shape[:2]
        target_width, target_height = target_size

        if maintain_aspect:
            # Calculate scale to fit within target size
            scale = min(target_width / width, target_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = target_width, target_height

        # Resize using OpenCV
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        return resized

    @staticmethod
    def get_image_info(file_path):
        """
        Get basic information about an image file

        Args:
            file_path (str): Path to image file

        Returns:
            dict: Image information
        """
        try:
            with Image.open(file_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_bytes': os.path.getsize(file_path)
                }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def is_supported_format(file_path):
        """
        Check if file format is supported

        Args:
            file_path (str): Path to file

        Returns:
            bool: True if supported
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ImageUtils.SUPPORTED_FORMATS

    @staticmethod
    def convert_colorspace(image, source='RGB', target='RGB'):
        """
        Convert image between color spaces

        Args:
            image (numpy.ndarray): Input image
            source (str): Source color space
            target (str): Target color space

        Returns:
            numpy.ndarray: Converted image
        """
        if source == target:
            return image.copy()

        # Define conversion mappings
        conversions = {
            ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
            ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
            ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
            ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
            ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
            ('HSV', 'RGB'): cv2.COLOR_HSV2RGB,
            ('RGB', 'LAB'): cv2.COLOR_RGB2LAB,
            ('LAB', 'RGB'): cv2.COLOR_LAB2RGB,
        }

        conversion_key = (source, target)
        if conversion_key in conversions:
            return cv2.cvtColor(image, conversions[conversion_key])
        else:
            raise ValueError(f"Conversion from {source} to {target} not supported")

    @staticmethod
    def add_noise(image, noise_type='gaussian', amount=0.1):
        """
        Add noise to image for testing purposes

        Args:
            image (numpy.ndarray): Input image
            noise_type (str): Type of noise ('gaussian', 'salt_pepper', 'poisson')
            amount (float): Noise intensity

        Returns:
            numpy.ndarray: Noisy image
        """
        ImageUtils.validate_image(image)
        noisy_image = image.copy()

        if noise_type == 'gaussian':
            noise = np.random.normal(0, amount, image.shape)
            noisy_image = image + noise

        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            coords = np.random.random(image.shape) < amount
            noisy_image[coords] = np.random.choice([0, 1], size=np.sum(coords))

        elif noise_type == 'poisson':
            # Poisson noise
            noisy_image = np.random.poisson(image * amount) / amount

        return np.clip(noisy_image, 0, 1)