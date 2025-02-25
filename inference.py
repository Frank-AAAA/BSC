import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from Bsc_edited import Unet

def load_model(model_path, device="cuda"):
    model = Unet()
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.to(device)
    model.eval()
    return model

def visualize_prediction(model, image_path, mask_path, device="cuda"):
    image = np.load(image_path)  # (H, W) / (C, H, W)
    mask = np.load(mask_path)

    if image.ndim == 2:
        # (H,W) -> (1,H,W)
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0)
    else:
        image = torch.tensor(image, dtype=torch.float)

    if mask.ndim == 2:
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
    else:
        mask = torch.tensor(mask, dtype=torch.float)

    # 下采样
    if image.shape == (1, 1024, 1024):
        image = image[:, ::2, ::2]  # -> (1,512,512)
        mask = mask[:, ::2, ::2]

    pil_img = transforms.ToPILImage()(image)
    pil_mask = transforms.ToPILImage()(mask.byte())

    pil_img = pil_img.resize((256, 256), resample=Image.BILINEAR)
    pil_mask = pil_mask.resize((256, 256), resample=Image.NEAREST)

    image = transforms.ToTensor()(pil_img)
    mask = transforms.ToTensor()(pil_mask)

    image = image - image.min()
    image = image / (image.max() + 1e-7)

    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image) # (B,1,H,W)
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).float()

    pred = (prob > 0.5).float()

    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title("Input Image")
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title("Predicted Mask")
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("output/best_model.pth", device=device)
    # 样本
    filename = "20231027-095207_Neasden Si(001)-H--STM_AtomManipulation--9_2_0.npy"
    test_image_path = rf"C:\Users\XYH\OneDrive - University College London\Desktop\infer_test_image\{filename}"
    test_mask_path = rf"C:\Users\XYH\OneDrive - University College London\Desktop\infer_test_mask\{filename}"
    visualize_prediction(model, test_image_path, test_mask_path, device=device)
