# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

This is an Image Anti-Photography Protection System - a defensive security tool that protects images from unauthorized camera capture by adding imperceptible modifications that become visible when photographed. The system uses three complementary protection algorithms that exploit differences between human vision and camera sensors.

## Core Architecture

### Main Application Flow
- **Entry Point**: `image_protector.py` - Simple terminal interface with folder-based workflow
- **Input**: Images placed in `input_images/` folder
- **Output**: Protected images saved to `protected_images/` folder
- **Interaction**: Interactive prompts for image selection and protection level

### Protection Pipeline
1. **Image Loading**: `utils/image_utils.py` handles all file I/O with format validation
2. **Protection Application**: Three algorithms applied sequentially with configurable strength
3. **Image Saving**: Output with filename indicating protection level

### Protection Algorithms (in `protection/` directory)
- **BayerInterferenceProtection**: Creates patterns aligned with camera sensor Bayer filter (RGGB) causing severe aliasing and color distortion during demosaicing
- **ScreenMoireProtection**: Exploits LCD/OLED subpixel structure to create moiré patterns when screen is photographed
- **FrequencyDomainProtection**: Uses FFT to inject high-frequency noise patterns imperceptible to humans but visible to camera sensors
- **InfraredPatternProtection**: Modifies RGB channels to simulate near-infrared characteristics that cameras detect differently than human eyes
- **AdversarialNoiseProtection**: Generates noise patterns that confuse camera autofocus, exposure, and white balance systems

## Development Environment Setup

```bash
# Environment setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run application
python3 image_protector.py
```

## Key Technical Details

### Image Processing Pipeline
- All images normalized to float32 [0,1] range for processing
- RGB format used throughout (BGR converted internally for OpenCV operations)
- Protection algorithms preserve visual imperceptibility (mean difference < 0.05)

### Protection Strength Levels
- **Light (0.3-0.4)**: Moderate camera distortion, imperceptible on screen
  - Good for basic cameras and older smartphones
  - May be filtered by modern AI processing (iPhone 16, Pixel 9)

- **Medium (0.6-0.7)**: Strong camera distortion, still imperceptible on screen
  - **RECOMMENDED for modern smartphones** (iPhone 16, Samsung S24, etc.)
  - Resists computational photography and AI noise reduction
  - Balanced imperceptibility vs camera distortion

- **Heavy (0.85-0.9)**: Maximum camera distortion with minimal visibility
  - For high-end cameras with aggressive processing
  - May show very slight texture on screen upon close inspection
  - Complete image destruction when photographed

- **Extreme (1.0)**: Experimental maximum strength
  - Guaranteed severe camera distortion
  - May show minor visible patterns on screen

### Algorithm Implementation Notes

**Strength Multipliers (Balanced for iPhone 16 Resistance):**

1. **Bayer Interference**: 0.015-0.022 range
   - Creates Nyquist-frequency patterns aligned with 2x2 RGGB sensor grid
   - Exploits demosaicing algorithms with anti-correlated channel modifications
   - Chroma subsampling exploitation (4:2:0, 4:2:2)

2. **Screen Moiré**: 0.016-0.020 range
   - Generates patterns at irrational frequencies (golden ratio √1.618, √5 = 2.236)
   - LCD subpixel (RGB stripes) and OLED PenTile (RGBG) exploitation
   - Maximizes moiré with camera sensor grid interference

3. **Frequency Domain**: 0.012-0.025 range
   - Uses scipy.fftpack for FFT operations
   - High-frequency emphasis filters with sigmoid rolloff
   - Patterns at Nyquist limit (imperceptible to human CSF)

4. **Infrared Patterns**: 0.008-0.018 range
   - Simulates near-IR material properties (Red: high IR, Blue: low IR)
   - Metameric color shifts and spectral confusion
   - Channel-specific modifications exploit camera sensor IR sensitivity

5. **Adversarial Noise**: 0.008-0.012 range
   - Targets camera AI subsystems (Autofocus, Auto-Exposure, Auto-White-Balance)
   - False edge patterns for AF confusion
   - Brightness and color cast patterns for metering disruption

**Key Implementation Details:**
- All patterns normalized to mean=0, std=1 before application
- Cumulative effect across 5 algorithms provides robust protection
- Strength multipliers calibrated to resist iPhone 16 computational photography
- Trade-off: imperceptibility to humans vs camera AI resistance

## Supported Image Formats
JPEG, PNG, BMP, TIFF - handled through both OpenCV and PIL for maximum compatibility

## Security and Ethics
This is a defensive security tool designed for legitimate privacy protection. The codebase should only be enhanced for defensive purposes - never for circumventing security systems or hiding illegal activities.

## Common Development Tasks

### Testing Protection Effectiveness
**For Modern Smartphones (iPhone 16, etc.):**
1. Use MEDIUM or HEAVY protection level
2. Display protected image full screen on monitor (max brightness)
3. Photograph with smartphone at close range (fill frame)
4. View captured photo - should show rainbow/grid/moiré distortion
5. If no distortion: try HEAVY/EXTREME or get closer to screen

**For Older Cameras:**
1. LIGHT or MEDIUM protection sufficient
2. Distortion should be immediately visible

### Adding New Protection Algorithms
1. Create new class in `protection/` directory
2. Implement `apply(image, strength)` method
3. Add to protection pipeline in `image_protector.py`
4. Follow existing pattern: validate input, normalize strength, return clipped output
5. **Calibrate strength multipliers**:
   - Start with 0.01-0.02 range for imperceptibility
   - Test with iPhone 16 to verify distortion
   - Adjust multipliers to balance visibility vs protection

### Modifying Protection Levels
Update the preset dictionaries in `select_protection_level()` function in `image_protector.py`:120-96

### Adjusting Strength Multipliers
Strength multipliers are in each algorithm's `apply()` methods:
- `bayer_interference.py`: Lines 90, 94, 98, 192-194, 236-237, 307-309
- `screen_moire.py`: Lines 93-96, 152-157, 210-213, 282-285
- `frequency_domain.py`: Lines 93, 194
- `infrared_patterns.py`: Lines 74, 79, 84, 156, 159, 162, 229-231, 237-238, 331
- `adversarial_noise.py`: Lines 69, 176, 247, 322, 397

**Calibration guideline:**
- Human imperceptibility threshold: < 0.005 per pixel (cumulative)
- iPhone 16 distortion threshold: > 0.015 per pixel (cumulative)
- Current settings: ~0.01-0.02 per pixel at Medium strength (0.6-0.7)