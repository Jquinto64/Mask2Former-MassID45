"""
Standalone inference script for Mask2Former models.
Runs on a single image or a directory of images.
Visualizes predictions and saves them to an output directory.
"""
import logging
import argparse
import os
import glob
import sys
import cv2
import numpy as np
import torch

# --- 1. SETUP PATHS (Adjust if necessary) ---
# We keep this from your snippet to ensure Mask2Former imports work
sys.path.insert(0, "mask2former")

# --- 2. IMPORTS ---
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances

# Mask2Former imports
from mask2former import add_maskformer2_config
# from mask2former.modeling import * # Imported implicitly by config/factory usually, but kept if needed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_predictor(args):
    """
    Sets up the Detectron2/Mask2Former config and predictor.
    """
    logger.info("Setting up configuration...")
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    
    # Add specific project configs
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Merge file config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    cfg.merge_from_file(args.config)
    
    # Load Weights
    cfg.MODEL.WEIGHTS = args.model_path
    
    # Set Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set Inference Image Size (Per your snippet)
    # DefaultPredictor handles resizing the input to this size, 
    # and then resizing predictions BACK to original size automatically.
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    
    cfg.freeze()

    logger.info(f"Loading model from {args.model_path}...")
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def predict_on_single_image(predictor, img_path, debug=False):
    """
    Runs inference on a single image.
    DefaultPredictor handles loading, resizing, and post-processing.
    """
    # Read image using OpenCV
    original_image = cv2.imread(img_path)
    if original_image is None:
        logger.error(f"Could not read image: {img_path}")
        return None, None

    # Run Inference
    # DefaultPredictor.__call__ handles:
    # 1. Resizing input to 1024 (based on cfg)
    # 2. Passing (image, orig_height, orig_width) to model
    # 3. Resizing output masks back to orig_height/orig_width
    outputs = predictor(original_image)

    if debug:
        print("\n--- DEBUG INFO ---")
        instances = outputs['instances']
        print(f"Detected {len(instances)} objects")
        if len(instances) > 0:
            if instances.has("pred_masks"):
                # These masks should already be resized back to original image size
                sample_preds = instances.pred_masks.cpu().detach().numpy()
                print(f"Mask Shape: {sample_preds.shape} (Should match original image {original_image.shape[:2]})")

    return outputs, original_image

def display_predictions(image, predictions, output_path, metadata):
    """
    Visualizes predictions and saves the result to disk.
    """
    if predictions is None:
        return

    # Convert BGR (OpenCV) to RGB for Visualizer
    visualizer = Visualizer(
        image[:, :, ::-1], 
        metadata=metadata, 
        scale=1.0, 
        instance_mode=ColorMode.IMAGE
    )
    
    # Draw predictions
    # We move instances to CPU before visualization
    instances = predictions["instances"].to("cpu")
    instances = predictions["instances"].to("cpu")
    instances_ = Instances(instances.image_size)# <class 'detectron2.structures.instances.Instances'>
    flag = False
    for index in range(len(instances)):
        # print(instances[index].scores)
        score = instances[index].scores[0]
        if score > 0.25: # confidence score
            if flag == False:
                instances_ = instances[index]
                flag = True
            else:
                instances_ = Instances.cat([instances_, instances[index]])
    vis_output = visualizer.draw_instance_predictions(predictions=instances_)

    # Get the result image (in RGB) and convert back to BGR for OpenCV saving
    result_image = vis_output.get_image()[:, :, ::-1]

    # Save
    cv2.imwrite(output_path, result_image)
    logger.info(f"Saved visualization to: {output_path}")

def main(args):
    # 1. Setup Predictor
    predictor, cfg = setup_predictor(args)

    # 2. Setup Metadata for Visualization
    # Define your specific mapping here
    category_mapping = {"1": "b"} 
    
    # We create a custom metadata object to ensure class names match your mapping
    metadata = MetadataCatalog.get("__unused_name_for_inference__") 
    # Ensure the list order corresponds exactly to the class IDs output by the model
    metadata.thing_classes = list(category_mapping.values())
    
    # 3. Handle Input Directory or File
    input_path = args.imgs_directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    if os.path.isdir(input_path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
    elif os.path.isfile(input_path):
        image_files = [input_path]
    else:
        logger.error(f"Input path {input_path} is not valid.")
        return

    logger.info(f"Found {len(image_files)} images to process.")

    # 4. Run Inference Loop
    for i, img_file in enumerate(image_files):
        filename = os.path.basename(img_file)
        save_path = os.path.join(output_dir, f"pred_{filename}")

        logger.info(f"[{i+1}/{len(image_files)}] Processing {filename}...")
        
        predictions, original_img = predict_on_single_image(
            predictor, 
            img_file, 
            debug=args.debug
        )

        if predictions is not None:
            display_predictions(original_img, predictions, save_path, metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone inference script for Mask2Former")
    
    # Path Arguments
    parser.add_argument("--model_path", required=True, help="Path to the model .pkl or .pth file")
    parser.add_argument("--config", required=True, help="Path to the Mask2Former .yaml config file")
    parser.add_argument("--imgs_directory", required=True, help="Path to input image or directory of images")
    parser.add_argument("--output_dir", default="output_predictions_m2f", help="Directory to save visualized outputs")
    
    # Optional Arguments
    parser.add_argument("--debug", action="store_true", help="Print debug information about masks/boxes")
    
    args = parser.parse_args()
    
    main(args)