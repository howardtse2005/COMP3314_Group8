from data.dataset import Dataset
from model.unet import UNet 
from training.trainer_unet import UNetTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import datetime
from config import Config as cfg
import torch.nn.functional as F

# Add tif writer
try:
    import tifffile as _tifffile
    def _save_tiff(path, arr):
        # arr shape: (N, H, W) or (N, H, W, C)
        _tifffile.imwrite(path, arr, photometric='minisblack')
except Exception:
    import imageio
    def _save_tiff(path, arr):
        # imageio expects a list of images or a stacked array
        imageio.mimwrite(path, arr, format='TIFF')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

test_img_path = cfg.dir_img_test
test_mask_path = cfg.dir_mask_test
checkpoint_path = cfg.pretrained_model


#--------------------- Main Test Function ---------------------

def test(test_data_path='data/test_example.txt',
         save_path='results/images',
         pretrained_model=checkpoint_path,
         test_img_path = test_img_path,
         test_mask_path = test_mask_path,
         threshold=0.65):
    
    # Create timestamp for folder names
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    result_folder_name = f"evaluation_{timestamp}"
    timestamped_save_path = os.path.join(save_path, result_folder_name)
    os.makedirs(timestamped_save_path, exist_ok=True)
    
    print(f"Results will be saved to: {timestamped_save_path}")
    
    # Load dataset
    test_dataset = Dataset(
        dataset_img_path=test_img_path,
        dataset_mask_path=test_mask_path,
        temp_dir=cfg.dir_temp_ts,
    )

    # Build model and trainer
    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()
    
    model = UNet()
    print("Using UNet architecture")
    print("Using Attention UNet architecture") 
        
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)
    
    trainer = UNetTrainer(
        model=model,
        optimizer=None,  
        criterions=None,
        train_loader=None,
        val_loader=None,
        log_dir=None,  
        chkp_dir=None
        ).to(device)

    model.load_state_dict(trainer.checkpointer.load(pretrained_model, multi_gpu=True))
    model.eval()
    
    # Store predictions and ground truths
    all_predictions = []
    all_groundtruths = []
    
    print("Processing full-resolution images...")
    
    # Process each image
    for idx in tqdm(range(len(test_dataset))):
        # Get image and mask from dataset
        img_tensor, gt_tensor = test_dataset[idx]
        
        # Prepare input batch - add batch dimension
        img_batch = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
        
        # Get ground truth
        gt_np = gt_tensor.numpy().astype(np.float32)
        
        # Perform inference
        with torch.no_grad():
            pred = torch.sigmoid(model(img_batch))
    
            # Remove batch dimension and move to CPU
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
        
        # Store results
        all_predictions.append(pred_np)
        all_groundtruths.append(gt_np)
        
        # Create visualization - convert image tensor back to numpy for visualization
        img_vis = img_tensor.permute(1, 2, 0).numpy() * 255.0
        img_vis = img_vis.astype(np.uint8)
    
    # Save all predictions and ground truths to multi-page TIFFs (matching order)
    if len(all_predictions) > 0 and len(all_groundtruths) > 0:
        # Stack as (N, H, W)
        preds_stack = np.stack(all_predictions, axis=0)

        # Convert predictions to uint8. If threshold is provided, create binary masks using it.
        if threshold is not None:
            # threshold is in [0,1], produce binary masks (0 or 255)
            preds_bool = (preds_stack >= float(threshold))
            preds_uint8 = (preds_bool.astype(np.uint8) * 255)
        else:
            # Normalize/clip predictions to [0,1] then convert to uint8 for compact storage
            preds_uint8 = (np.clip(preds_stack, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # Prepare ground-truth stack: ensure each gt is 2D (H, W)
        gts_processed = []
        for gt in all_groundtruths:
            gt_arr = np.array(gt)
            # If gt has channel or batch dims like (1, H, W) or (C, H, W), squeeze them
            while gt_arr.ndim > 2:
                gt_arr = np.squeeze(gt_arr, axis=0)
            # If ground-truth is not in [0,1], try to normalize assuming it's {0,1} or small ints
            if gt_arr.max() > 1:
                # scale to [0,1]
                gt_norm = gt_arr.astype(np.float32)
                gt_norm = (gt_norm - gt_norm.min()) / (gt_norm.max() - gt_norm.min() + 1e-8)
            else:
                gt_norm = gt_arr.astype(np.float32)
            gts_processed.append(gt_norm)

        gts_stack = np.stack(gts_processed, axis=0)
        gts_uint8 = (np.clip(gts_stack, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # Save files
        out_preds = os.path.join(timestamped_save_path, "predictions.tif")
        out_gts = os.path.join(timestamped_save_path, "gt.tif")

        _save_tiff(out_preds, preds_uint8)
        _save_tiff(out_gts, gts_uint8)

        print(f"Saved predictions TIFF to: {out_preds}")
        print(f"Saved ground-truth TIFF to: {out_gts}")
    else:
        print("No predictions or ground-truths to save.")
    
    print(f"Images saved to {timestamped_save_path}")


if __name__ == '__main__':
    test()