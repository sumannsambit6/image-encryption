# Image Anti-Photography Protection System

A defensive security tool that protects images from unauthorized camera capture by adding imperceptible modifications that become visible when photographed. The protected images look identical to the original when viewed on screen, but show severe distortion when captured by cameras (including modern smartphones like iPhone 16).

## How It Works

The system uses **five complementary protection techniques** that exploit fundamental differences between human vision and camera sensors:

### 1. Bayer Interference Protection
- Creates patterns aligned with camera sensor Bayer filters (RGGB)
- Causes severe aliasing and color distortion during demosaicing
- Exploits the 2x2 pixel grid structure of camera sensors

### 2. Screen Moiré Protection
- Exploits LCD/OLED subpixel structure (RGB stripes, PenTile)
- Creates moiré patterns when screen is photographed
- Uses irrational frequencies (golden ratio, √5) to maximize interference

### 3. High-Frequency Noise Injection
- Adds subtle high-frequency patterns in the frequency domain
- Human eyes naturally filter out these patterns
- Camera sensors capture them at Nyquist frequency, causing artifacts

### 4. Infrared Spectrum Manipulation
- Exploits differences between human vision (400-700nm) and camera sensors (350-1000nm)
- Simulates near-infrared characteristics that cameras detect differently
- Creates metameric color shifts invisible to human eyes

### 5. Adversarial Camera Interference
- Generates calculated noise patterns that disrupt camera AI systems:
  - Confuses autofocus mechanisms (phase-detection and contrast-detection)
  - Interferes with exposure metering (spot and matrix)
  - Disrupts white balance algorithms (AWB)
  - Creates false edge patterns for image processing confusion

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd image-encryption
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Simple Folder-Based Interface

1. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

2. **Run the application:**
```bash
python3 image_protector.py
```

3. **Follow the prompts:**
   - The program creates `input_images/` and `protected_images/` folders
   - Add your images to the `input_images/` folder
   - Select which image to protect (or choose 'all' for batch processing)
   - Choose protection level (Light/Medium/Heavy/Extreme)
   - Protected images are saved to `protected_images/` folder

### Protection Levels

- **Light (0.3-0.4)**: Moderate camera distortion, imperceptible on screen
  - Good for older cameras and basic smartphones
  - May be filtered by advanced AI processing (iPhone 16, Pixel 9, etc.)

- **Medium (0.6-0.7)**: Strong camera distortion, still imperceptible on screen
  - **Recommended for modern smartphones** (iPhone 16, Samsung S24, etc.)
  - Resists computational photography and AI noise reduction

- **Heavy (0.85-0.9)**: Maximum camera distortion with minimal visibility
  - For high-end cameras with aggressive processing
  - May show very slight texture on screen upon close inspection

- **Extreme (1.0)**: Experimental maximum strength
  - Guaranteed camera distortion
  - May show minor visible patterns on screen

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Example Workflow

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the application
python3 image_protector.py

# 3. Follow the prompts:
#    - Add images to input_images/ folder (if not already present)
#    - Select image(s) to protect (or 'all' for batch processing)
#    - Choose protection level:
#      1 = Light (basic cameras)
#      2 = Medium (modern smartphones) ← RECOMMENDED
#      3 = Heavy (iPhone 16, high-end cameras)
#      4 = Extreme (maximum protection)

# 4. Find protected images in protected_images/ folder
#    Files are named: original_name_protected_[level].ext
```

### Quick Test Workflow:

```bash
# Process a test image
python3 image_protector.py
# Choose: image → Medium protection

# Test on iPhone 16:
# 1. Display protected image full screen on Mac
# 2. Take photo with iPhone (fill frame, get close)
# 3. View iPhone photo → should show rainbow/grid distortion
```

## Testing Effectiveness

### For Modern Smartphones (iPhone 16, Pixel 9, Samsung S24, etc.):

1. **Process image with MEDIUM or HEAVY protection** (not Light)
   - Modern phones have aggressive AI noise reduction

2. **Display protected image FULL SCREEN** on your monitor/laptop
   - Use maximum screen brightness
   - Verify with naked eye that image looks identical to original

3. **Take photo with smartphone:**
   - Get CLOSE to the screen (fill the frame)
   - Let camera autofocus naturally (don't tap to focus)
   - Try slight angles (not perfectly perpendicular)
   - Take multiple shots at different distances

4. **View the captured photo:**
   - Should show color fringing, rainbow patterns, moiré
   - Grid-like artifacts and pixelation
   - Unexpected color shifts and distortion

5. **If no distortion visible:**
   - Try HEAVY or EXTREME protection level
   - Get closer to the screen when photographing
   - Try different screen brightness levels

### For DSLR/Mirrorless Cameras:

1. Start with LIGHT or MEDIUM protection
2. Take photos at various focal lengths
3. Disable in-camera noise reduction if possible
4. Distortion should be immediately visible

## Project Structure

```
├── image_protector.py          # Main application (terminal interface)
├── requirements.txt            # Python dependencies
├── protection/                 # Protection algorithms (5 techniques)
│   ├── bayer_interference.py  # Bayer sensor pattern exploitation
│   ├── screen_moire.py        # Screen pixel grid interference
│   ├── frequency_domain.py    # High-frequency noise injection
│   ├── infrared_patterns.py   # IR spectrum manipulation
│   └── adversarial_noise.py   # Camera AI interference
├── utils/                      # Utility functions
│   └── image_utils.py         # Image I/O and validation
├── input_images/              # Place your images here
└── protected_images/          # Protected images output here
```

## Protection Algorithms (Technical Details)

1. **BayerInterferenceProtection**: Exploits camera sensor demosaicing
   - Creates Nyquist-frequency patterns aligned with RGGB Bayer filter
   - Generates false correlation between adjacent pixels
   - Exploits chroma subsampling (4:2:0, 4:2:2)

2. **ScreenMoireProtection**: Screen pixel structure exploitation
   - LCD subpixel patterns (RGB horizontal stripes)
   - OLED PenTile confusion (RGBG diamond arrangement)
   - Irrational frequency patterns for maximum moiré

3. **FrequencyDomainProtection**: Frequency domain manipulation using FFT
   - High-frequency emphasis filters
   - Camera-visible noise patterns at Nyquist limit
   - Imperceptible to human visual system (CSF filtering)

4. **InfraredPatternProtection**: Near-IR simulation
   - Channel-specific modifications (Red: high IR, Blue: low IR)
   - Metameric color shifts (same to humans, different to cameras)
   - Material IR property simulation

5. **AdversarialNoiseProtection**: Camera AI system interference
   - Autofocus confusion (false edges, contrast noise)
   - Exposure metering disruption
   - White balance algorithm interference
   - Edge enhancement confusion patterns

## Limitations

### Camera-Specific:
- **Modern smartphones** (iPhone 16, Pixel 9, Samsung S24+):
  - Require MEDIUM or HEAVY protection due to AI processing
  - Computational photography can filter Light protection
  - Multi-frame processing may reduce effectiveness

- **DSLR/Mirrorless cameras**:
  - Generally more susceptible (less AI processing)
  - LIGHT or MEDIUM protection typically sufficient

- **Webcams and basic cameras**:
  - Very effective even at LIGHT protection
  - Limited processing capabilities

### Environmental Factors:
- Very bright lighting conditions may reduce moiré visibility
- Screen type matters (OLED better than LCD for protection)
- Viewing angle affects moiré pattern strength

### Technical Limitations:
- Protection is **defensive only** - designed for legitimate privacy
- Patterns exploit fundamental sensor physics (can't be fully eliminated)
- Trade-off between imperceptibility and protection strength

## Legal and Ethical Usage

This tool is intended for legitimate privacy protection purposes only:
- Protecting personal photos from unauthorized capture
- Preventing unauthorized documentation in private settings
- Research and educational purposes

**Do not use this tool for:**
- Circumventing security systems
- Hiding illegal activities
- Interfering with legitimate photography rights

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- Pillow
- SciPy

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### No Images Found
- Make sure images are in the `input_images/` folder
- Check that images have supported extensions (.jpg, .png, .bmp, .tiff)

### Processing Errors
- Ensure sufficient memory for large images
- Verify file permissions for input/output directories
- Check that image files are not corrupted