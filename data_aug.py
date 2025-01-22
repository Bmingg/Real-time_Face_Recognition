import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import os
from PIL import Image
import numpy as np
import cv2

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """Adjust brightness and contrast of an image."""
    image = np.array(image)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def simulate_lighting_conditions(image):
    lighting_conditions = []
    alphas = [0.5, 1.0, 1.5]  # Contrast values
    betas = [-50, 0, 50]      # Brightness values

    for alpha in alphas:
        for beta in betas:
            if alpha == 0.5 and beta == -50:
                continue
            if alpha == 1.5 and beta == 50:
                continue
            adjusted = adjust_brightness_contrast(image, alpha=alpha, beta=beta)
            lighting_conditions.append((alpha, beta, adjusted))

    return lighting_conditions

def augment_and_process(image_path, output_dir, folder, current_index):
    try:
        pil_image = Image.open(image_path)
        if pil_image is None:
            print(f"Skipping {image_path} as face processing failed.")
            return current_index

        lighting_conditions = simulate_lighting_conditions(pil_image)

        transformations = {
            "Original": lambda x: x,
            # "Horizontal Flip": transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
            # "Rotation5": transforms.Compose([lambda x: F.rotate(x, -5)]),
            "Rotation10": transforms.Compose([lambda x: F.rotate(x, -10)]),
            "Rotation15": transforms.Compose([lambda x: F.rotate(x, -15)]),
            "Rotation30": transforms.Compose([lambda x: F.rotate(x, -30)]),
            "Rotation45": transforms.Compose([lambda x: F.rotate(x, -45)]),
            # "RotationInv5": transforms.Compose([lambda x: F.rotate(x, 5)]),
            "RotationInv10": transforms.Compose([lambda x: F.rotate(x, 10)]),
            "RotationInv15": transforms.Compose([lambda x: F.rotate(x, 15)]),
            "RotationInv30": transforms.Compose([lambda x: F.rotate(x, 30)]),
            "RotationInv45": transforms.Compose([lambda x: F.rotate(x, 45)]),
            # "Rotation75": transforms.Compose([lambda x: F.rotate(x, 75)])
            "Crop": transforms.Compose([
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
            ]),
            # "Affine": transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10)])
        }

        output_subfolder = os.path.join(output_dir, folder)
        os.makedirs(output_subfolder, exist_ok=True)

        original_augmentations = {}
        for name, transform in transformations.items():
            augmented = transform(pil_image)
            original_augmentations[name] = augmented
            augmented_array = np.array(augmented)

            height, width = augmented_array.shape[:2]
            x_start, y_start = 0, 0
            x_end, y_end = width, height

            off_set = 10
            eyebrow_y = int(y_end * 0.269) - off_set
            roi_gray = augmented_array[eyebrow_y:y_end, x_start:x_end]

            # Save the final image
            save_path = os.path.join(
                output_subfolder,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_Orig_{current_index}.jpg"
            )
            cv2.imwrite(save_path, cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB))
            print(f"Original transformation image saved: {save_path}")
            current_index += 1

        for idx, (alpha, beta, lighting_image) in enumerate(lighting_conditions):
            lighting_pil_image = Image.fromarray(lighting_image)
            for name, transform in transformations.items():
                augmented_image = transform(lighting_pil_image)

                if isinstance(augmented_image, Image.Image):
                    augmented_image = np.array(augmented_image)

                # Post-processing: crop ROI and save directly
                height, width = augmented_image.shape[:2]
                x_start, y_start = 0, 0
                x_end, y_end = width, height

                off_set = 10
                eyebrow_y = int(y_end * 0.269) - off_set
                roi_gray = augmented_image[eyebrow_y:y_end, x_start:x_end]

                roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB)

                # Save the final image
                save_path = os.path.join(
                    output_subfolder,
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_Aug_{current_index}.jpg"
                )
                cv2.imwrite(save_path, roi_rgb)
                print(f"Image saved: {save_path}")
                current_index += 1

        return current_index

    except (FileNotFoundError, IOError):
        print(f"Error: Could not open or read image file at {image_path}")
        return current_index

aug_input_dir = "dataset_crop_test"
aug_output_dir = "new_augmented_dataset_crop_test"
aug_target_folder = "2"
 
target_folder_path = os.path.join(aug_input_dir, aug_target_folder)
if not os.path.exists(target_folder_path):
                print(f"Target folder {aug_target_folder} does not exist. Exiting.")
else:
                current_index = 1
                for filename in sorted(os.listdir(target_folder_path)):
                    image_path = os.path.join(target_folder_path, filename)
                    if os.path.isfile(image_path):
                        print(f"Processing image: {image_path}")
                        current_index = augment_and_process(
                             image_path=image_path,
                             output_dir= aug_output_dir,
                             folder= aug_target_folder,
                             current_index=current_index
            )