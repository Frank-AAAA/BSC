import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torchvision import transforms
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ---------------------- data segmentation ----------------------
class AddNoise:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        noise = torch.randn(img.size()) * 0.43
        noisy_img = img + noise
        return noisy_img.clamp(0, 1)

class KernelFilter:
    def __call__(self, img):
        return img.filter(ImageFilter.BLUR)

class RandomRotationFromSet(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        img_ = TF.rotate(img, angle)
        return img_

class RandomSizeCrop:
    def __init__(self, sizes, padding=None):
        self.sizes = sizes
        self.padding = padding

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Unsupported image type: {type(img)}")
        crop_size = random.choice(self.sizes)
        cropped_img = transforms.RandomCrop(crop_size)(img)
        return cropped_img

class RandomScanLineArtefact(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, scan):
        r1 = random.random()
        r2 = random.random()
        scan_maxmin = (torch.clone(scan) - torch.min(scan)) / (
            torch.max(scan) - torch.min(scan)
        )

        if r1 < self.p:
            rng = np.random.default_rng()
            res = scan.shape[1]
            num_lines = 15
            lines = rng.integers(0, res - 1, (num_lines,))
            columns = rng.integers(0, res - 1, (num_lines,))
            lengths = rng.integers(0, int(res * 0.8), (num_lines,))
            add_ons = rng.random(size=(num_lines,)) / 1.67
            for i in range(7):
                scan_maxmin[:, lines[i], columns[i] : columns[i] + lengths[i]] += (
                    add_ons[i]
                )
            for i in range(7, 9):
                scan_maxmin[
                    :, lines[i] : lines[i] + 2, columns[i] : columns[i] + lengths[i]
                ] += add_ons[i]
            for i in range(9, 13):
                end = rng.integers(200, 314) / 100
                lengths[i] = scan_maxmin[
                    :, lines[i], columns[i] : columns[i] + lengths[i]
                ].shape[0]
                cos = np.cos(np.linspace(0, end, num=lengths[i]))
                scan_maxmin[:, lines[i], columns[i] : columns[i] + lengths[i]] += (
                    cos * add_ons[i]
                )
            for i in range(13, 15):
                end = rng.integers(200, 314) / 100
                lengths[i] = scan_maxmin[
                    :, lines[i] : lines[i] + 2, columns[i] : columns[i] + lengths[i]
                ].shape[1]
                cos = np.cos(np.linspace(0, end, num=lengths[i]))
                scan_maxmin[
                    :, lines[i] : lines[i] + 2, columns[i] : columns[i] + lengths[i]
                ] += cos * add_ons[i]

        if r2 < self.p:
            rng = np.random.default_rng(12345)
            res = scan.shape[1]
            lines = rng.integers(0, res, (10,))
            columns = rng.integers(0, res, (10,))
            lengths = rng.integers(0, int(res * 0.1), (10,))
            add_ons = rng.random(size=(10,)) / 1.67
            for i in range(10):
                scan_maxmin[:, lines[i], columns[i] : columns[i] + lengths[i]] += (
                    add_ons[i]
                )

        return scan_maxmin

class AddSaltPepperNoise:
    def __init__(self, prob=0.01):
        self.prob = prob

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        noisy_img = img.copy()
        if len(noisy_img.shape) == 2:
            channels = False
        elif len(noisy_img.shape) == 3:
            channels = True
        else:
            raise ValueError(f"Unexpected image shape: {noisy_img.shape}")

        num_salt = int(self.prob * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
        if channels:
            noisy_img[coords[0], coords[1], :] = 255
        else:
            noisy_img[coords[0], coords[1]] = 255

        num_pepper = int(self.prob * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
        if channels:
            noisy_img[coords[0], coords[1], :] = 0
        else:
            noisy_img[coords[0], coords[1]] = 0

        return Image.fromarray(noisy_img.astype("uint8"))

class AddMixedNoise:
    def __init__(self, gaussian_std=0.2, sp_prob=0.02):
        self.gaussian_std = gaussian_std
        self.sp_prob = sp_prob

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        gaussian_noise = torch.randn(img.size()) * self.gaussian_std
        img = img + gaussian_noise
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        num_salt = int(self.sp_prob * img_np.size * 0.5)
        num_pepper = int(self.sp_prob * img_np.size * 0.5)
        coords_salt = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape[:2]]
        coords_pepper = [
            np.random.randint(0, i - 1, num_pepper) for i in img_np.shape[:2]
        ]
        img_np[coords_salt[0], coords_salt[1], :] = 255
        img_np[coords_pepper[0], coords_pepper[1], :] = 0
        img = torch.tensor(img_np).permute(2, 0, 1) / 255.0
        return img.clamp(0, 1)

def denormalize(image, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return image * std + mean

# ---------------------- Visualization ----------------------
def visualize_one_batch(model, dataloader, output, device="cuda", threshold=0.5):
    """
    Visualizes a batch of images along with their corresponding ground truth masks and model predictions.
    
    Parameters:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader providing image batches.
        output (Tensor): Output from the model.
        device (str): Device to run the model on (default: 'cuda').
        threshold (float): Threshold to binarize predictions.
    """
    # Fetch one batch of data
    sample_images, sample_masks = next(iter(dataloader))
    sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)

    # Put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(sample_images)  # Get model predictions
        prob = torch.sigmoid(output)  # Apply sigmoid to get probability map
        pred = (prob > threshold).float()  # Binarize predictions

    num_to_show = min(len(sample_images), 3)  # Limit number of images displayed
    fig, axes = plt.subplots(num_to_show, 3, figsize=(12, 4 * num_to_show))

    if num_to_show == 1:
        axes = [axes]  # Ensure axes are iterable

    for i in range(num_to_show):
        img_np = sample_images[i].squeeze().cpu().numpy()
        mask_np = sample_masks[i].squeeze().cpu().numpy()
        pred_np = pred[i].squeeze().cpu().numpy()

        axes[i][0].imshow(img_np, cmap="afmhut")
        axes[i][0].set_title("Input Image")

        axes[i][1].imshow(mask_np, cmap="afmhut")
        axes[i][1].set_title("Ground Truth Mask")
        print("Ground Truth Mask min value =", mask_np.min(), " max value =", mask_np.max())

        axes[i][2].imshow(pred_np, cmap="afmhut")
        axes[i][2].set_title("Predicted Mask")
        print("Prediction Mask min value =", pred_np.min(), " max value =", pred_np.max())

    plt.tight_layout()
    plt.show()

# ---------------------- Segmentation Dataset ----------------------
class SegmentationDataset(Dataset):
    """
    A PyTorch Dataset class for loading segmentation images and masks.
    
    Parameters:
        image_dir (str): Directory containing image files.
        mask_dir (str): Directory containing corresponding mask files.
        length (int): Number of samples in the dataset.
        is_train (bool): Whether the dataset is for training or validation.
    """
    def __init__(self, image_dir, mask_dir, length=500, is_train=True):
        self.image_dir = image_dir  # Path to image directory
        self.mask_dir = mask_dir  # Path to mask directory
        self.length = length  # Define dataset length
        self.is_train = is_train  # Define mode (training or validation)

        self.valid_extensions = (".npy",)  # Only allow .npy format
        self.images = [
            f for f in os.listdir(image_dir) if f.lower().endswith(self.valid_extensions)
        ]
        if len(self.images) == 0:
            raise ValueError(f"No valid .npy files found in directory: {image_dir}")

        # Define transformations based on training or validation mode
        if self.is_train:
            self.both_transform = transforms.Compose(
                [
                    RandomRotationFromSet([0, 90, 180, 270]),  # Apply random rotations
                    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
                    transforms.RandomCrop(32),  # Random crop of size 64x64
                ]
            )
        else:
            self.both_transform = None
            self.fixed_transform = transforms.Compose(
                [
                    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),  # Resize image to 256x256
                    transforms.ToTensor(),  # Convert image to PyTorch tensor
                ]
            )

        # Resize transformations for training images and masks
        self.train_resize_img = transforms.Resize(
            (256, 256), interpolation=InterpolationMode.BILINEAR
        )
        self.train_resize_mask = transforms.Resize(
            (256, 256), interpolation=InterpolationMode.NEAREST
        )

    def __len__(self):
        """Returns the length of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """
        Loads an image-mask pair, applies transformations, and returns them as tensors.
        
        Parameters:
            idx (int): Index of the sample.
        
        Returns:
            image (Tensor): Transformed image tensor.
            mask (Tensor): Corresponding transformed mask tensor.
        """
        idx = idx % len(self.images)  # Ensure index wraps around dataset size
        img_path = os.path.join(self.image_dir, self.images[idx])  # Get image file path
        mask_path = os.path.join(self.mask_dir, self.images[idx])  # Get mask file path

        image = np.load(img_path)  # Load image as numpy array
        mask = np.load(mask_path)  # Load mask as numpy array

        # Convert to PyTorch tensors and reshape if necessary
        if image.ndim == 2:
            image = torch.tensor(image).unsqueeze(0).float()  # Add channel dimension
        elif image.ndim == 3:
            image = torch.tensor(image).permute(2, 0, 1).float()  # Change to (C, H, W) format

        if mask.ndim == 2:
            mask = torch.tensor(mask).unsqueeze(0).float()  # Add channel dimension

        # Downsample large images
        if image.shape == (1, 1024, 1024):
            image = image[:, ::2, ::2]  # Reduce resolution by half
            mask = mask[:, ::2, ::2]  # Reduce mask resolution accordingly

        if self.is_train:
            # Convert tensors to PIL images for augmentation
            pil_img = transforms.ToPILImage()(image)
            pil_mask = transforms.ToPILImage()((mask * 255).byte())

            # Resize images and masks to 256x256
            pil_img = self.train_resize_img(pil_img)
            pil_mask = self.train_resize_mask(pil_mask)

            # Convert back to tensors
            image = transforms.ToTensor()(pil_img)
            mask = transforms.ToTensor()(pil_mask)

            # Apply data augmentations
            both = torch.cat([image, mask], dim=0)  # Stack image and mask together
            both = self.both_transform(both)  # Apply transformations

            image = both[0].unsqueeze(0)  # Extract transformed image
            mask = both[1].unsqueeze(0)  # Extract transformed mask

            # Binarize mask
            mask = (mask > 0.5).float()

            # Normalize image to range [0,1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-7)

        else:
            # Convert tensors to PIL images for resizing
            pil_img = transforms.ToPILImage()(image)
            pil_mask = transforms.ToPILImage()((mask * 255).byte())

            # Resize images and masks using different interpolation methods
            pil_img = pil_img.resize((256, 256), resample=Image.BILINEAR)
            pil_mask = pil_mask.resize((256, 256), resample=Image.NEAREST)

            # Convert back to tensors
            image = transforms.ToTensor()(pil_img)
            mask = transforms.ToTensor()(pil_mask)

            # Normalize image to range [0,1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-7)

        return image, mask

def plot_loss_curve(history, loss_name="Loss"):
    """
    Plots the training and validation loss curves.
    
    Parameters:
        history (dict): Dictionary containing 'train_loss' and 'val_loss' lists.
        loss_name (str): Name of the loss function to be displayed on the graph.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
        color="black",
        linestyle="-",
        linewidth=2,
    )
    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        color="blue",
        linestyle="-",
        linewidth=2,
    )
    plt.xlabel("Epochs")
    plt.ylabel(loss_name)
    plt.title(f"Training and Validation {loss_name}")
    
    # Highlight the best model point
    min_val_loss_idx = history["val_loss"].index(min(history["val_loss"])) + 1
    plt.scatter(
        min_val_loss_idx,
        history["val_loss"][min_val_loss_idx - 1],
        color="green",
        marker="*",
        s=150,
        label="Best Model",
    )
    
    # Highlight overfitting region
    overfit_start = min_val_loss_idx + 10 if min_val_loss_idx + 10 < len(epochs) else min_val_loss_idx
    plt.axvspan(overfit_start, len(epochs), color="red", alpha=0.2, label="Overfitting")
    if overfit_start < len(epochs):
        plt.text(
            overfit_start + (len(epochs) - overfit_start) / 4,
            max(history["val_loss"]) * 0.9,
            "Overfitting",
            color="red",
            fontsize=12,
        )
    
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

# ---------------------- U-Net ----------------------
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)


        self.up1 = self.upconv(1024, 512)
        self.dec1 = self.conv_block(1024, 512)
        self.up2 = self.upconv(512, 256)
        self.dec2 = self.conv_block(512, 256)
        self.up3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(256, 128)
        self.up4 = self.upconv(128, 64)
        self.dec4 = self.conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def upconv(self, in_c, out_c):
        return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(F.max_pool2d(c1, 2))
        c3 = self.enc3(F.max_pool2d(c2, 2))
        c4 = self.enc4(F.max_pool2d(c3, 2))
        c5 = self.enc5(F.max_pool2d(c4, 2))

        u1 = self.up1(c5)
        d1 = self.dec1(torch.cat([u1, c4], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, c3], dim=1))
        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([u3, c2], dim=1))
        u4 = self.up4(d3)
        d4 = self.dec4(torch.cat([u4, c1], dim=1))
        return self.out_conv(d4)

# ---------------------- train model ----------------------
def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    model_name,
    device="cuda",
    early_stopping_patience=10,
):
    output_dir = os.path.dirname(model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.to(device)
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Val
            model.eval()
            val_loss = 0.0
            val_iou_total = 0.0
            val_dice_total = 0.0
            val_samples = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

                    # SUM IoU & Dice
                    probs = torch.sigmoid(output)
                    for b in range(data.size(0)):
                        val_iou_total += compute_iou(probs[b], target[b])
                        val_dice_total += compute_dice(probs[b], target[b])
                        val_samples += 1

            val_loss /= len(val_loader)
            mean_val_iou = val_iou_total / (val_samples + 1e-7)
            mean_val_dice = val_dice_total / (val_samples + 1e-7)

            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_iou"].append(mean_val_iou)
            history["val_dice"].append(mean_val_dice)

            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val IoU: {mean_val_iou:.4f}, "
                f"Val Dice: {mean_val_dice:.4f}"
            )

            # 早停
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), model_name)
                print(f"  save as {model_name}")
            else:
                if val_loss > best_loss+0.1:
                    patience_counter += 1
                    print(f"  loss going up :{patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. \n"
                    f"Best epoch: {best_epoch + 1} with Val Loss: {best_loss:.4f}"
                )
                break

        return history

    except KeyboardInterrupt:
        interrupt_model_path = f"output/interrupt_model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), interrupt_model_path)
        print(f"\nKeyboardInterrupt detected. Model saved to {interrupt_model_path}")
        return history

# ---------------------- sample of the visualize ----------------------
def visualize_sample(image, mask, prediction, loss):
    """
    Visualizes a single image, its ground truth mask, and the model's predicted mask.
    
    Parameters:
        image (Tensor): Input image.
        mask (Tensor): Ground truth mask.
        prediction (Tensor): Model's predicted mask.
        loss (float): Loss value for this prediction.
    """
    filename = "20231027-095207_Neasden Si(001)-H--STM_AtomManipulation--9_2_0.npy"
    image = np.load(rf"C:\Users\XYH\OneDrive - University College London\Desktop\infer_test_image\{filename}")
    mask = np.load(rf"C:\Users\XYH\OneDrive - University College London\Desktop\infer_test_mask\{filename}")

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='afmhot')
    plt.title('Train Data')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().squeeze(), cmap="afmhot")
    plt.title("Ground Truth Mask")
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction.cpu().squeeze().detach().numpy(), cmap="afmhot")
    plt.title(f"Predicted Mask (Loss: {loss:.6f})")
    
    plt.show()

# ---------------------- IoU and Dice Computation ----------------------
def compute_iou(pred, target, threshold=0.5):
    """
    Computes the Intersection over Union (IoU) score between the predicted and target masks.
    
    Parameters:
        pred (Tensor): The predicted mask.
        target (Tensor): The ground truth mask.
        threshold (float): Threshold to binarize predictions.
    
    Returns:
        float: IoU score.
    """
    pred = (pred > threshold).float()  # Binarize prediction mask
    target = target.float()  # Ensure target is a float tensor
    intersection = (pred * target).sum()  # Compute intersection area
    union = pred.sum() + target.sum() - intersection  # Compute union area
    iou = (intersection + 1e-7) / (union + 1e-7)  # Compute IoU with smoothing term
    return iou.item()

def compute_dice(pred, target, threshold=0.5):
    """
    Computes the Dice Coefficient (F1 score) between the predicted and target masks.
    
    Parameters:
        pred (Tensor): The predicted mask.
        target (Tensor): The ground truth mask.
        threshold (float): Threshold to binarize predictions.
    
    Returns:
        float: Dice coefficient.
    """
    pred = (pred > threshold).float()  # Binarize prediction mask
    target = target.float()  # Ensure target is a float tensor
    intersection = (pred * target).sum()  # Compute intersection area
    dice = (2.0 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)  # Compute Dice score with smoothing term
    return dice.item()

# ---------------------- Model Testing ----------------------
def test_model(model, test_loader, device="cuda"):
    """
    Evaluates the trained model on the test dataset.
    
    Parameters:
        model (nn.Module): The trained segmentation model.
        test_loader (DataLoader): DataLoader for test data.
        device (str): Device to run the model on (default: 'cuda').
    
    Returns:
        tuple: Mean IoU and Dice scores across the test dataset.
    """
    model.load_state_dict(torch.load("output/best_model.pth", map_location=device,weights_only=True))  # Load trained model weights
    model.to(device)  # Move model to specified device
    model.eval()  # Set model to evaluation mode

    test_total = 0
    test_correct = 0
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    last_image, last_mask, last_output = None, None, None

    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Get model predictions
            probs = torch.sigmoid(outputs)  # Apply sigmoid activation
            predicted = (probs > 0.5).float()  # Binarize predictions
            
            test_total += labels.numel()  # Count total pixels
            test_correct += (predicted == labels).sum().item()  # Count correctly predicted pixels
            
            for j in range(images.size(0)):
                iou = compute_iou(probs[j], labels[j])  # Compute IoU for the sample
                dice = compute_dice(probs[j], labels[j])  # Compute Dice coefficient
                total_iou += iou
                total_dice += dice
                num_samples += 1
                last_image, last_mask, last_output = images[j], labels[j], predicted[j]

    print(f"Test Pixel Accuracy: {test_correct / test_total:.4f}")  # Print overall pixel accuracy

    # Visualize the last test sample
    if last_image is not None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(last_image.cpu().squeeze().numpy())
        plt.title("Input Image")
        
        plt.subplot(1, 3, 2)
        plt.imshow(last_mask.cpu().squeeze().numpy())
        plt.title("Ground Truth Mask")
        
        plt.subplot(1, 3, 3)
        plt.imshow(last_output.cpu().squeeze().numpy())
        plt.title("Predicted Mask")
        
        plt.show()

    return total_iou / num_samples, total_dice / num_samples
# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    try:
        train_image_dir = r'C:\Users\XYH\OneDrive - University College London\Desktop\train_image'  # 训练图像目录
        train_mask_dir = r'C:\Users\XYH\OneDrive - University College London\Desktop\train_mask'  # 训练标签目录
        test_image_dir = r'C:\Users\XYH\OneDrive - University College London\Desktop\test_image'  # 测试图像目录
        test_mask_dir = r'C:\Users\XYH\OneDrive - University College London\Desktop\test_mask'  # 测试标签目录

        train_dataset = SegmentationDataset(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            length=2500,
            is_train=True,
        )
        test_dataset = SegmentationDataset(
            image_dir=test_image_dir, mask_dir=test_mask_dir, length=300, is_train=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model = Unet().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        num_epochs = 1
        history = train(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            "output/best_model.pth",
            device,
        )
        plot_loss_curve(history)
        test_model(model, test_loader, device)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"An error occurred: {e}")
