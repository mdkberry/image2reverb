#!/usr/bin/env python3
"""
Script to generate an impulse response from a single image using the Image2Reverb model.
"""

import os
import argparse
import shutil
import soundfile as sf
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Generate impulse response from a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for results")
    parser.add_argument("--encoder_path", type=str, default="models/resnet50_places365.pth.tar", help="Path to pre-trained Encoder ResNet50 model")
    parser.add_argument("--depthmodel_path", type=str, default="models/mono_640x192", help="Path to pre-trained depth (from monodepth2) encoder and decoder models")
    parser.add_argument("--model_path", type=str, default="models/model.ckpt", help="Path to pretrained Image2Reverb model")
    args = parser.parse_args()

    # Create temporary dataset structure
    temp_dataset = "./temp_dataset"
    test_A_dir = os.path.join(temp_dataset, "test_A")
    test_B_dir = os.path.join(temp_dataset, "test_B")
    
    os.makedirs(test_A_dir, exist_ok=True)
    os.makedirs(test_B_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy and resize image to test_A folder
    image_name = os.path.basename(args.image_path)
    temp_image_path = os.path.join(test_A_dir, image_name)
    
    # Resize image to 512x512 if needed
    img = Image.open(args.image_path)
    img_resized = img.resize((512, 512), Image.LANCZOS)
    img_resized.save(temp_image_path)
    
    # Create a dummy audio file in test_B folder (required by the dataset loader)
    dummy_audio_path = os.path.join(test_B_dir, os.path.splitext(image_name)[0] + ".wav")
    # Create 1 second of silence at 22050 Hz
    silence = np.zeros(22050)
    sf.write(dummy_audio_path, silence, 22050)
    
    print(f"Image copied to {temp_image_path}")
    print(f"Dummy audio created at {dummy_audio_path}")
    
    # Run the test script
    cmd = f"python test.py --dataset {temp_dataset} --model {args.model_path} --encoder_path {args.encoder_path} --depthmodel_path {args.depthmodel_path} --test_dir {args.output_dir}"
    print(f"Running command: {cmd}")
    os.system(cmd)
    
    # Clean up temporary dataset
    shutil.rmtree(temp_dataset)
    print("Temporary dataset cleaned up")

if __name__ == "__main__":
    main()
