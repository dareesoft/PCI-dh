import numpy as np

DISTRESS_DATA_TYPES = {
    'alligator_cracking': {  # 악어균열
        'measurements': {
            'area': np.float32,  # m²
            'crack_width': np.float32,  # mm
            'pattern_density': np.float32  # cracks/m²
        },
        'mask_type': 'binary',
        'required_views': ['top']
    },
    'bleeding': {  # 블리딩
        'measurements': {
            'area': np.float32,  # m²
            'gloss_level': np.float32,  # 0-1 normalized
            'surface_temperature': np.float32,  # °C
            'stickiness_level': np.int32  # 0: none, 1: slight, 2: moderate, 3: severe
        },
        'mask_type': 'reflectance_map',
        'required_views': ['top', 'angled']
    },
    'block_cracking': {  # 블록균열
        'measurements': {
            'block_size': np.float32,  # m²
            'crack_width': np.float32,  # mm
            'block_count': np.int32,
            'total_area': np.float32  # m²
        },
        'mask_type': 'binary',
        'required_views': ['top']
    },
    'bumps_and_sags': {  # 범프와 처짐
        'measurements': {
            'height': np.float32,  # mm (for bumps)
            'depth': np.float32,  # mm (for sags)
            'length': np.float32,  # m
            'width': np.float32,  # m
            'ride_quality_impact': np.int32  # 1: low, 2: medium, 3: high
        },
        'mask_type': 'depth_map',
        'required_views': ['side', 'front']
    },
    'corrugation': {  # 파상로
        'measurements': {
            'wavelength': np.float32,  # m
            'amplitude': np.float32,  # mm
            'area': np.float32,  # m²
            'ride_quality_impact': np.int32  # 1: low, 2: medium, 3: high
        },
        'mask_type': 'depth_map',
        'required_views': ['front', 'side']
    },
    'depression': {  # 침하
        'measurements': {
            'depth': np.float32,  # mm
            'area': np.float32,  # m²
            'water_retention': np.bool_,  # True if water pools
            'shape_regularity': np.float32  # 0-1 normalized
        },
        'mask_type': 'depth_map',
        'required_views': ['top', 'angled']
    },
    'edge_cracking': {  # 모서리 균열
        'measurements': {
            'length': np.float32,  # m
            'width': np.float32,  # mm
            'distance_from_edge': np.float32,  # m
            'raveling_present': np.bool_
        },
        'mask_type': 'binary',
        'required_views': ['top']
    },
    'joint_reflection_cracking': {  # 조인트 반사 균열
        'measurements': {
            'length': np.float32,  # m
            'width': np.float32,  # mm
            'spalling_width': np.float32,  # mm
            'vertical_displacement': np.float32  # mm
        },
        'mask_type': 'binary',
        'required_views': ['top']
    },
    'lane_shoulder_drop_off': {  # 차로/길어깨 단차
        'measurements': {
            'height_difference': np.float32,  # mm
            'length': np.float32,  # m
            'slope': np.float32,  # degrees
            'edge_condition': np.int32  # 0: intact, 1: minor damage, 2: severe damage
        },
        'mask_type': 'depth_map',
        'required_views': ['side', 'angled']
    },
    'longitudinal_transverse_cracking': {  # 종횡 균열
        'measurements': {
            'length': np.float32,  # m
            'width': np.float32,  # mm
            'orientation': np.float32,  # degrees
            'crack_type': np.int32  # 0: longitudinal, 1: transverse
        },
        'mask_type': 'binary',
        'required_views': ['top']
    },
    'patching_utility_cut': {  # 패칭
        'measurements': {
            'area': np.float32,  # m²
            'perimeter': np.float32,  # m
            'roughness': np.float32,  # IRI value
            'deterioration_level': np.int32  # 0: good, 1: moderate, 2: poor
        },
        'mask_type': 'binary',
        'required_views': ['top']
    },
    'polished_aggregate': {  # 골재 광택
        'measurements': {
            'area': np.float32,  # m²
            'skid_resistance': np.float32,  # coefficient
            'texture_depth': np.float32,  # mm
            'reflectance': np.float32  # 0-1 normalized
        },
        'mask_type': 'reflectance_map',
        'required_views': ['top', 'angled']
    },
    'potholes': {  # 포트홀
        'measurements': {
            'depth': np.float32,  # mm
            'diameter': np.float32,  # mm
            'area': np.float32,  # m²
            'volume': np.float32  # m³
        },
        'mask_type': '3d_point_cloud',
        'required_views': ['top', 'angled']
    },
    'railroad_crossing': {  # 철도 건널목
        'measurements': {
            'bump_height': np.float32,  # mm
            'depression_depth': np.float32,  # mm
            'crossing_width': np.float32,  # m
            'ride_quality_impact': np.int32  # 1: low, 2: medium, 3: high
        },
        'mask_type': 'depth_map',
        'required_views': ['front', 'top']
    },
    'rutting': {  # 러팅
        'measurements': {
            'depth': np.float32,  # mm
            'width': np.float32,  # mm
            'length': np.float32,  # m
            'cross_section_area': np.float32  # mm²
        },
        'mask_type': 'depth_map',
        'required_views': ['front', 'top']
    },
    'shoving': {  # 밀림
        'measurements': {
            'height': np.float32,  # mm
            'length': np.float32,  # m
            'width': np.float32,  # m
            'direction': np.float32  # degrees
        },
        'mask_type': 'depth_map',
        'required_views': ['front', 'top']
    },
    'slippage_cracking': {  # 슬립 균열
        'measurements': {
            'length': np.float32,  # m
            'width': np.float32,  # mm
            'curvature': np.float32,  # 1/m
            'orientation': np.float32  # degrees
        },
        'mask_type': 'binary',
        'required_views': ['top']
    },
    'swell': {  # 융기
        'measurements': {
            'height': np.float32,  # mm
            'area': np.float32,  # m²
            'length': np.float32,  # m
            'ride_quality_impact': np.int32  # 1: low, 2: medium, 3: high
        },
        'mask_type': 'depth_map',
        'required_views': ['front', 'side']
    },
    'weathering_raveling': {  # 풍화/마모
        'measurements': {
            'area': np.float32,  # m²
            'depth_loss': np.float32,  # mm
            'texture_loss': np.float32,  # 0-1 normalized
            'aggregate_loss': np.float32  # kg/m²
        },
        'mask_type': 'depth_texture_map',
        'required_views': ['top', 'angled']
    }
}