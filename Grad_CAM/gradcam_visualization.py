"""
Grad-CAM Visualization Code for Paper Figure 3
==============================================

This script generates the attention visualizations needed for the paper.
It compares attention patterns between TrashNet-trained and RealWaste-trained models.

Requirements:
    pip install torch torchvision pytorch-grad-cam pillow matplotlib numpy pandas opencv-python --break-system-packages

Usage:
    python gradcam_visualization.py
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse
from pathlib import Path

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_TN_MODEL = os.path.join(PROJECT_ROOT, 'model', 'best_resnet18_ablation_trashnet.pth')
DEFAULT_RW_MODEL = os.path.join(PROJECT_ROOT, 'model', 'best_resnet18_ablation_realwaste.pth')
DEFAULT_SAMPLES_DIR = os.path.join(BASE_DIR, 'aligned_images')
DEFAULT_JSON_PATH = os.path.join(BASE_DIR, 'label', 'project-4-at-2026-01-16-15-58-932da740.json')
DEFAULT_MASKS_DIR = os.path.join(BASE_DIR, 'masks')
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'visualizations')


def parse_labelstudio_json(json_path, output_dir, images_dir):
    """
    Parse Label Studio JSON and generate binary masks for images.
    """
    print(f"Parsing annotations from {json_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    count = 0
    for task in tasks:
        # Get image filename
        img_url = task.get('data', {}).get('image') or task.get('data', {}).get('img')
        if not img_url:
            continue
            
        filename = os.path.basename(img_url)
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {filename} not found in {images_dir}")
            continue
            
        # Open image to get dimensions
        try:
            with Image.open(image_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"Error opening image {filename}: {e}")
            continue

        # Create mask
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        has_annotation = False
        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                # Handle Polygon Labels
                if result.get('type') == 'polygonlabels':
                    points = result.get('value', {}).get('points', [])
                    if not points:
                        continue
                        
                    # Convert percentage points to pixels
                    # Label Studio points are [x, y] in 0-100 range
                    pixels = []
                    for pt in points:
                        x = pt[0] * w / 100.0
                        y = pt[1] * h / 100.0
                        pixels.append((x, y))
                    
                    if len(pixels) > 2:
                        draw.polygon(pixels, outline=255, fill=255)
                        has_annotation = True
                
                # Handle Rectangle Labels
                elif result.get('type') == 'rectanglelabels':
                    value = result.get('value', {})
                    x = value.get('x', 0) * w / 100.0
                    y = value.get('y', 0) * h / 100.0
                    width = value.get('width', 0) * w / 100.0
                    height = value.get('height', 0) * h / 100.0
                    
                    draw.rectangle([x, y, x + width, y + height], outline=255, fill=255)
                    has_annotation = True

        if has_annotation:
            mask_filename = filename.rsplit('.', 1)[0] + '_mask.png'
            mask.save(os.path.join(output_dir, mask_filename))
            count += 1
            
    print(f"Generated {count} masks in {output_dir}")


class GradCAMVisualizer:
    """Generate Grad-CAM visualizations for waste classification models"""
    
    def __init__(self, model_path, num_classes=5, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            num_classes: Number of output classes
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path} on {self.device}...")
        
        # Load model
        self.model = models.resnet18(weights=None) # Changed from pretrained=False to weights=None for newer torch
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()
        
        # Setup Grad-CAM (target the final conv layer)
        self.target_layers = [self.model.layer4[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def generate_cam(self, image_path, target_class=None):
        """
        Generate Grad-CAM heatmap for an image
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image.resize((224, 224))) / 255.0
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            confidence, pred_class = torch.softmax(output, dim=1).max(dim=1)
            pred_class = pred_class.item()
            confidence = confidence.item()
        
        # Generate CAM
        targets = None if target_class is None else [ClassifierOutputTarget(target_class)]
        cam_map = self.cam(input_tensor=input_tensor, targets=targets)[0]
        
        return cam_map, pred_class, confidence, image_np
    
    def compute_attention_concentration(self, cam_map, object_mask):
        """
        Compute attention concentration metric (AC)
        Args:
            cam_map: Grad-CAM heatmap (H, W), values in [0, 1]
            object_mask: Binary mask (H, W), >0 = object, 0 = background
        """
        # Ensure same size
        if cam_map.shape != object_mask.shape:
            from scipy.ndimage import zoom
            scale_factors = (object_mask.shape[0] / cam_map.shape[0],
                           object_mask.shape[1] / cam_map.shape[1])
            cam_map = zoom(cam_map, scale_factors, order=1)
        
        # Normalize CAM
        cam_normalized = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        
        # Binarize mask if needed (ensure it's 0 or 1)
        binary_mask = (object_mask > 0).astype(float)
        
        # Compute AC
        attention_on_object = (cam_normalized * binary_mask).sum()
        total_attention = cam_normalized.sum()
        ac_score = attention_on_object / (total_attention + 1e-8)
        
        return ac_score


def batch_compute_ac_statistics(samples_dir, masks_dir, tn_model_path, 
                                rw_model_path, output_csv='ac_statistics.csv'):
    """
    Compute AC statistics for all samples (for Table 5 in the paper)
    """
    import pandas as pd
    
    tn_viz = GradCAMVisualizer(tn_model_path)
    rw_viz = GradCAMVisualizer(rw_model_path)
    
    results = []
    
    files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for image_file in files:
        image_path = os.path.join(samples_dir, image_file)
        
        # Look for mask (png or npy)
        mask_name_base = os.path.splitext(image_file)[0] + '_mask'
        mask_path = os.path.join(masks_dir, mask_name_base + '.png')
        
        if not os.path.exists(mask_path):
             mask_path = os.path.join(masks_dir, mask_name_base + '.npy')
             if not os.path.exists(mask_path):
                print(f"⚠️  Mask not found for {image_file}, skipping...")
                continue
        
        # Load mask
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            # Load image mask and resize to 224x224 to match CAM output roughly (or keep original)
            # Actually CAM output is 224x224 usually because of resize in generate_cam, 
            # but generate_cam returns cam_map of size 224x224 (input size)
            # So we should resize mask to 224x224
            mask_img = Image.open(mask_path).convert('L').resize((224, 224), Image.NEAREST)
            mask = np.array(mask_img)
        
        # Generate CAMs and compute AC
        try:
            tn_cam, tn_pred, tn_conf, _ = tn_viz.generate_cam(image_path)
            rw_cam, rw_pred, rw_conf, _ = rw_viz.generate_cam(image_path)
            
            tn_ac = tn_viz.compute_attention_concentration(tn_cam, mask)
            rw_ac = rw_viz.compute_attention_concentration(rw_cam, mask)
            
            # Determine dataset origin
            dataset = 'TrashNet' if 'tn_' in image_file.lower() else 'RealWaste'
            
            results.append({
                'image': image_file,
                'dataset': dataset,
                'tn_pred': tn_pred,
                'tn_conf': tn_conf,
                'tn_ac': tn_ac,
                'rw_pred': rw_pred,
                'rw_conf': rw_conf,
                'rw_ac': rw_ac
            })
            
            print(f"✓ Processed {image_file}: TN_AC={tn_ac:.3f}, RW_AC={rw_ac:.3f}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    if not results:
        print("No results generated.")
        return None

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved statistics: {output_csv}")
    
    # Compute summary statistics (for Table 5)
    print("\n" + "="*60)
    print("SUMMARY STATISTICS FOR TABLE 5")
    print("="*60)
    
    for dataset in ['TrashNet', 'RealWaste']:
        subset = df[df['dataset'] == dataset]
        if len(subset) == 0: continue
        
        print(f"\n{dataset} samples ({len(subset)}):")
        print(f"  TN Model - Mean AC: {subset['tn_ac'].mean():.3f} ± {subset['tn_ac'].std():.3f}")
        print(f"  RW Model - Mean AC: {subset['rw_ac'].mean():.3f} ± {subset['rw_ac'].std():.3f}")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--tn_model', default=DEFAULT_TN_MODEL, help='Path to TrashNet model')
    parser.add_argument('--rw_model', default=DEFAULT_RW_MODEL, help='Path to RealWaste model')
    parser.add_argument('--samples_dir', default=DEFAULT_SAMPLES_DIR, help='Directory with sample images')
    parser.add_argument('--json_path', default=DEFAULT_JSON_PATH, help='Path to Label Studio JSON')
    parser.add_argument('--masks_dir', default=DEFAULT_MASKS_DIR, help='Directory to save/load masks')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--mode', default='batch', choices=['batch', 'figure'], help='Operation mode')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.masks_dir, exist_ok=True)
    
    # Step 1: Generate masks from JSON if JSON exists
    if os.path.exists(args.json_path):
        parse_labelstudio_json(args.json_path, args.masks_dir, args.samples_dir)
    else:
        print(f"Info: JSON file not found at {args.json_path}. Assuming masks already exist in {args.masks_dir}")

    # Step 2: Run Analysis
    if args.mode == 'batch':
        batch_compute_ac_statistics(
            samples_dir=args.samples_dir,
            masks_dir=args.masks_dir,
            tn_model_path=args.tn_model,
            rw_model_path=args.rw_model,
            output_csv=os.path.join(args.output_dir, 'ac_statistics.csv')
        )
    
    elif args.mode == 'figure':
        print("Figure generation mode is simplified here. Check batch mode for statistics.")
        # Re-implement figure generation if needed, but batch is more important for AC/mAP analysis context.

    print("\n✨ Done!")
