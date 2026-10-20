"""
Preprocesses and caches exercise images at 224x224 resolution for ultra-fast training.
"""

import os
import json
import time
import numpy as np
from PIL import Image

def prepare_cache(
    json_path='../filtered_exercise_dataset/exercise_annotations.json',
    images_dir='../filtered_exercise_dataset/images',
    cache_dir='cache_224'
):
    os.makedirs(cache_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Preparing 224x224 cached images from {images_dir} to {cache_dir}...")
    start_time = time.time()
    
    count = 0
    for item in data['annotations']:
        img_name = item['image_name']
        src_path = os.path.join(images_dir, img_name)
        dst_path = os.path.join(cache_dir, img_name)
        
        if os.path.exists(dst_path):
            count += 1
            continue
            
        if not os.path.exists(src_path):
            continue
            
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                resized = img.resize((224, 224), Image.Resampling.BILINEAR)
                resized.save(dst_path, format='JPEG', quality=85)
                count += 1
        except Exception:
            continue
            
        if count % 500 == 0:
            print(f"  Processed {count} images...")

    elapsed = time.time() - start_time
    print(f"Completed! Cached {count} images at 224x224 in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_cache(
        json_path=os.path.join(script_dir, '../filtered_exercise_dataset/exercise_annotations.json'),
        images_dir=os.path.join(script_dir, '../filtered_exercise_dataset/images'),
        cache_dir=os.path.join(script_dir, 'cache_224')
    )
