"""
Exercise Pose Dataset Loader for CNN Keypoint Regression
Loads filtered exercise images, extracts joint coordinates, normalizes them,
and constructs high-performance tf.data pipelines with augmentation.
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# 16 MPII Joint Names
JOINT_NAMES = [
    'r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
    'pelvis', 'thorax', 'upper_neck', 'head_top',
    'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'
]


def load_annotations(json_path, images_dir, cache_dir='cache_224'):
    """
    Parses exercise_annotations.json and filters valid samples with keypoints.
    If cache_dir exists, points image paths to the fast 224x224 cache.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file not found at: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    use_cache = os.path.exists(cache_dir)
    
    samples = []
    for item in data['annotations']:
        img_name = item['image_name']
        src_path = os.path.join(images_dir, img_name)
        if not os.path.exists(src_path):
            continue
            
        cached_path = os.path.join(cache_dir, img_name) if use_cache else src_path
        if use_cache and not os.path.exists(cached_path):
            continue
            
        persons = item.get('persons', [])
        if not persons:
            continue
            
        best_person = max(persons, key=lambda p: len(p.get('points', [])))
        points = best_person.get('points', [])
        if len(points) == 0:
            continue
            
        try:
            with Image.open(src_path) as im:
                width, height = im.size
        except Exception:
            continue
            
        if width <= 0 or height <= 0:
            continue
            
        kps = np.zeros((16, 2), dtype=np.float32)
        mask = np.zeros((16, 2), dtype=np.float32)
        
        for pt in points:
            jid = pt.get('id', -1)
            if 0 <= jid < 16:
                x = pt.get('x', 0.0)
                y = pt.get('y', 0.0)
                if x > 0 and y > 0:
                    norm_x = np.clip(x / width, 0.0, 1.0)
                    norm_y = np.clip(y / height, 0.0, 1.0)
                    kps[jid, 0] = norm_x
                    kps[jid, 1] = norm_y
                    mask[jid, 0] = 1.0
                    mask[jid, 1] = 1.0
                    
        samples.append({
            'image_path': cached_path,
            'image_name': img_name,
            'activity': item.get('activity_caption', ''),
            'keypoints': kps.flatten(),
            'mask': mask.flatten()
        })
        
    return samples


def create_tf_dataset(samples, img_size=(224, 224), batch_size=32, is_training=True):
    """
    Creates an optimized tf.data.Dataset pipeline.
    """
    image_paths = [s['image_path'] for s in samples]
    keypoints = np.array([s['keypoints'] for s in samples], dtype=np.float32)
    masks = np.array([s['mask'] for s in samples], dtype=np.float32)

    def parse_fn(path, kps, msk):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        
        combined_target = tf.concat([kps, msk], axis=0)
        return img, combined_target

    ds = tf.data.Dataset.from_tensor_slices((image_paths, keypoints, masks))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(samples), seed=42)
        
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def parse_fn(path, kps, msk):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0

    if is_training:  # wrap this in a closure or pass flag
        # Safe augmentations that don't require keypoint remapping
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)

    combined_target = tf.concat([kps, msk], axis=0)
    return img, combined_target