# Road Pavement Condition Analysis with AI

An advanced deep learning system that combines Language Segment-Anything and specialized pavement distress detection models to automatically assess road conditions according to ASTM D6433 standards.

![example_detection.png](/assets/outputs/example_detection.png)

## Features

### Core AI Models
- **Language-SAM Integration**: Zero-shot text-to-bbox approach for initial distress detection
- **Specialized Detection Models**:
 - Enhanced texture analysis for surface degradation  
 - Depth estimation for 3D distresses
 - Temporal analysis for consistent detection
- **ASTM D6433 Compliance**: Automated calculation of Pavement Condition Index (PCI)

### Technical Capabilities
- Multi-scale feature extraction
- Cross-modality attention mechanisms 
- Uncertainty estimation
- Batch processing support
- Real-time analysis capabilities
- Easy deployment using Lightning AI platform

## System Architecture

### Data Processing Pipeline
Input Image → Language-SAM Detection → Specialized Analysis → PCI Calculation
↓                    ↓                        ↓                    ↓
4K Video     Text-Guided Segmentation     Texture Analysis     Distress Rating
↓                    ↓                  Depth Analysis           ↓
GPS Data        Initial Masks              Temporal Analysis    Final Report

### Distress Types Coverage
Comprehensive detection of 19 ASTM D6433 distress types:
- Alligator/Fatigue Cracking
- Bleeding
- Block Cracking  
- Rutting
- Potholes
- And more...

## Getting Started

### Prerequisites
- Python 3.11 or higher
- CUDA-capable GPU

### Model Architecture
The system uses a multi-headed neural network architecture:

Vision Transformer backbone
Specialized heads for texture/depth analysis
Cross-modality attention mechanisms
Uncertainty estimation

### Acknowledgments
This project builds upon:

Language Segment-Anything
GroundingDINO
Segment-Anything-2
LitServe

### License
This project is licensed under the Apache 2.0 License

### Citations

@misc{language-sam,
    title={Language Segment-Anything},
    author={Luca Medeiros},
    year={2024},
    publisher={Github},
    url={https://github.com/luca-medeiros/lang-segment-anything}
}

@article{astm-d6433,
    title={Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys},
    author={ASTM International},
    year={2007},
    journal={ASTM D6433-07}
}