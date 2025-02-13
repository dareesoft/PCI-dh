import torch
from torch.utils.data import Dataset
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

"""
dataset_structure = {
    'root_dir/': {
        'images/': {
            'train/',
            'val/',
            'test/'
        },
        'annotations/': {  # JSON format
            'train/',
            'val/',
            'test/'
        },
        'masks/': {  # Binary masks
            'train/',
            'val/',
            'test/'
        },
        'depth_maps/': {  # Numpy arrays
            'train/',
            'val/',
            'test/'
        },
        'metadata/': {
            'camera_calibration.json',
            'environmental_conditions.json'
        }
    }
}

metadata = {
    'image_info': {
        'capture_time': 'YYYY-MM-DD HH:MM:SS',
        'weather_condition': 'sunny/rainy/cloudy',
        'illumination_level': 'lux_value',
        'road_surface_condition': 'dry/wet'
    },
    'camera_params': {
        'focal_length': 'mm',
        'sensor_size': 'mm',
        'mounting_height': 'm',
        'tilt_angle': 'degrees'
    },
    'location_info': {
        'gps_coordinates': [lat, lon],
        'road_type': 'highway/local/etc',
        'pavement_type': 'asphalt/concrete'
    }
}
"""

# 데이터셋 구조 주석 추가
"""
Dataset Types and Formats:

1. Images/
    - Format: JPG/PNG
    - Resolution: 1920x1080 (FHD)
    - Color Space: RGB
    - Bit Depth: 8-bit
    - Naming: {timestamp}_{location_id}_{camera_id}.jpg

2. Annotations/
    - Format: JSON
    - Structure per file:
        {
            "image_id": str,
            "timestamp": str,
            "distresses": [
                {
                    "type": str,  # from distress_types
                    "severity": str,  # L/M/H
                    "bbox": [x1, y1, x2, y2],  # normalized coordinates
                    "segmentation": [[]],  # polygon coordinates
                    "area": float,
                    "length": float,  # for linear distresses
                    "depth": float,  # for 3D distresses
                    "measurements": {
                        "width": float,
                        "height": float,
                        "diameter": float  # for potholes
                    },
                    "confidence": float
                }
            ],
            "road_properties": {
                "surface_type": str,
                "age": int,
                "last_maintenance": str
            }
        }

3. Masks/
    - Format: PNG (1-channel)
    - Resolution: Same as input image
    - Value Range: 
        0: background
        1-19: distress type ID
        20-39: severity level per type
    - Naming: {image_id}_mask.png

4. Depth Maps/
    - Format: NPY
    - Type: float32
    - Range: 0-255 (normalized depth)
    - Resolution: Same as input image
    - Naming: {image_id}_depth.npy

5. Metadata/
    - Format: JSON
    - Contains:
        - Camera calibration
        - Environmental conditions
        - Road properties
        - Maintenance history
"""


class PavementDistressDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load dataset index
        self.samples = self._load_dataset_index()
        
        # Define complete distress types according to ASTM D6433
        self.distress_types = {
            'alligator_cracking': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'crack_width <= 3.175mm',  # 1/8 inch
                    'M': '3.175mm < crack_width <= 19.05mm',  # 3/4 inch
                    'H': 'crack_width > 19.05mm'
                }
            },
            'bleeding': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'slight_discoloration_few_days_per_year',
                    'M': 'sticky_surface_few_weeks_per_year',
                    'H': 'extensive_sticky_surface_several_weeks'
                }
            },
            'block_cracking': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'blocks_defined_by_low_severity_cracks',
                    'M': 'blocks_defined_by_medium_severity_cracks',
                    'H': 'blocks_defined_by_high_severity_cracks'
                }
            },
            'bumps_and_sags': {
                'measurement': 'length',
                'severity_criteria': {
                    'L': 'bump_causes_low_severity_ride_quality',
                    'M': 'bump_causes_medium_severity_ride_quality',
                    'H': 'bump_causes_high_severity_ride_quality'
                }
            },
            'corrugation': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'causes_low_severity_ride_quality',
                    'M': 'causes_medium_severity_ride_quality',
                    'H': 'causes_high_severity_ride_quality'
                }
            },
            'depression': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': '13mm to 25mm depth',
                    'M': '25mm to 50mm depth',
                    'H': 'more than 50mm depth'
                }
            },
            'edge_cracking': {
                'measurement': 'length',
                'severity_criteria': {
                    'L': 'low_or_medium_cracking_no_breakup',
                    'M': 'medium_cracks_with_some_breakup',
                    'H': 'considerable_breakup_or_raveling'
                }
            },
            'joint_reflection_cracking': {
                'measurement': 'length',
                'severity_criteria': {
                    'L': 'crack_width < 10mm',
                    'M': '10mm <= crack_width < 75mm',
                    'H': 'crack_width >= 75mm'
                }
            },
            'lane_shoulder_drop_off': {
                'measurement': 'length',
                'severity_criteria': {
                    'L': '25mm to 50mm difference',
                    'M': '50mm to 100mm difference',
                    'H': 'more than 100mm difference'
                }
            },
            'longitudinal_transverse_cracking': {
                'measurement': 'length',
                'severity_criteria': {
                    'L': 'crack_width < 10mm',
                    'M': '10mm <= crack_width < 75mm',
                    'H': 'crack_width >= 75mm'
                }
            },
            'patching_utility_cut': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'patch_in_good_condition',
                    'M': 'patch_moderately_deteriorated',
                    'H': 'patch_badly_deteriorated'
                }
            },
            'polished_aggregate': {
                'measurement': 'area',
                'severity_criteria': {
                    'N': 'no_severity_levels_defined'
                }
            },
            'potholes': {
                'measurement': 'count_and_area',
                'severity_criteria': {
                    'L': 'depth < 25mm',
                    'M': '25mm <= depth < 50mm',
                    'H': 'depth >= 50mm'
                },
                'diameter_criteria': {
                    'small': 'diameter < 100mm',
                    'medium': '100mm <= diameter < 450mm',
                    'large': '450mm <= diameter < 750mm'
                }
            },
            'railroad_crossing': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'causes_low_severity_ride_quality',
                    'M': 'causes_medium_severity_ride_quality',
                    'H': 'causes_high_severity_ride_quality'
                }
            },
            'rutting': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': '6mm to 13mm depth',
                    'M': '13mm to 25mm depth',
                    'H': 'more than 25mm depth'
                }
            },
            'shoving': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'causes_low_severity_ride_quality',
                    'M': 'causes_medium_severity_ride_quality',
                    'H': 'causes_high_severity_ride_quality'
                }
            },
            'slippage_cracking': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'crack_width < 10mm',
                    'M': '10mm <= crack_width < 40mm',
                    'H': 'crack_width >= 40mm'
                }
            },
            'swell': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'causes_low_severity_ride_quality',
                    'M': 'causes_medium_severity_ride_quality',
                    'H': 'causes_high_severity_ride_quality'
                }
            },
            'weathering_raveling': {
                'measurement': 'area',
                'severity_criteria': {
                    'L': 'aggregate_or_binder_worn_away_slightly',
                    'M': 'aggregate_or_binder_worn_away_significantly',
                    'H': 'aggregate_or_binder_worn_away_severely'
                }
            }
        }

    def validate_annotation(self, annotation: Dict) -> bool:
        """Validate annotation against defined criteria"""
        distress_type = annotation['distress_type']
        severity = annotation['severity']
        
        if distress_type not in self.distress_types:
            return False
            
        if severity not in self.distress_types[distress_type]['severity_criteria']:
            return False
            
        # Validate measurement type
        required_measurement = self.distress_types[distress_type]['measurement']
        if required_measurement not in annotation['measurements']:
            return False
            
        return True

    def calculate_distress_density(self, distress_type: str, measurements: Dict) -> float:
        """Calculate distress density according to ASTM D6433"""
        if self.distress_types[distress_type]['measurement'] == 'area':
            return measurements['area'] / measurements['sample_area'] * 100
        elif self.distress_types[distress_type]['measurement'] == 'length':
            return measurements['length'] / measurements['sample_length'] * 100
        elif self.distress_types[distress_type]['measurement'] == 'count_and_area':
            return measurements['count'] / measurements['sample_area'] * 100
        else:
            raise ValueError(f"Unknown measurement type for {distress_type}")

    def get_required_annotations(self, distress_type: str) -> List[str]:
        """Get required annotation types for a distress type"""
        measurement_type = self.distress_types[distress_type]['measurement']
        
        if measurement_type == 'area':
            return ['segmentation_mask']
        elif measurement_type == 'length':
            return ['polyline']
        elif measurement_type == 'count_and_area':
            return ['segmentation_mask', 'point']
        else:
            return ['bounding_box']

def balance_dataset(self):
    """Ensure balanced representation of distress types"""
    counts = defaultdict(int)
    for sample in self.samples:
        for distress in sample['distresses']:
            counts[distress['type']] += 1
            
    # Calculate sampling weights
    max_count = max(counts.values())
    sampling_weights = {
        dtype: max_count/count 
        for dtype, count in counts.items()
    }
    
    return sampling_weights

def validate_sample(self, sample: Dict) -> bool:
    """Validate sample quality and annotations"""
    checks = [
        self._check_image_quality(sample['image']),
        self._check_annotation_completeness(sample['annotations']),
        self._check_mask_validity(sample['masks']),
        self._validate_measurements(sample['measurements']),
        self._check_metadata_completeness(sample['metadata'])
    ]
    return all(checks)