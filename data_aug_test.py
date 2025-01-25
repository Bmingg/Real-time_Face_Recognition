import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import os
from PIL import Image
import numpy as np
import cv2
import random

def simulate_lighting_conditions(image):
    """Simulate different lighting conditions by changing brightness and contrast."""
    lighting_conditions = []
    
    low_brightness_high_contrast = adjust_brightness_contrast(image, alpha=1.5, beta=-50)
    lighting_conditions.append(('LowBrightness_HighContrast', np.array(low_brightness_high_contrast)))
    
    # Simulate high brightness, low contrast
    high_brightness_low_contrast = adjust_brightness_contrast(image, alpha=0.5, beta=50)
    lighting_conditions.append(('HighBrightness_LowContrast', np.array(high_brightness_low_contrast)))
    
    return lighting_conditions

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """Adjust brightness and contrast of an image."""
    image = np.array(image)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# def add_motion_blur(image, kernel_size=5):
#     """Apply motion blur to an image."""
#     kernel = np.zeros((kernel_size, kernel_size))
#     kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
#     kernel /= kernel_size
#     blurred = cv2.filter2D(np.array(image), -1, kernel)
#     return blurred

# def simulate_shadow(image):
#     """Simulate shadow effects on the image."""
#     image = np.array(image)
#     top_x, top_y = image.shape[1] // 3, image.shape[0] // 3
#     bottom_x, bottom_y = top_x * 2, top_y * 2
#     shadow_mask = np.zeros_like(image, dtype=np.uint8)
#     cv2.rectangle(shadow_mask, (top_x, top_y), (bottom_x, bottom_y), (50, 50, 50), -1)
#     shadowed = cv2.addWeighted(image, 1, shadow_mask, 0.5, 0)
#     return shadowed

# def apply_gamma_correction(image, gamma=1.0):
#     """Apply gamma correction to an image."""
#     inv_gamma = 1.0 / gamma
#     table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
#     return cv2.LUT(np.array(image), table)

def zoom_image(image, zoom_factor=1.2):
    """Zoom into the center of the image."""
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    
    # Resizing the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Cropping the zoomed area back to the original size
    left = (new_width - width) / 2
    top = (new_height - height) / 2
    right = (new_width + width) / 2
    bottom = (new_height + height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image

def translate_image(image, tx=0, ty=0):
    """Translate (shift) the image by tx pixels horizontally and ty pixels vertically."""
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(np.array(image), translation_matrix, (image.width, image.height))
    return Image.fromarray(translated_image)

def mean_filter(image, kernel_size=3):
    """Apply mean filtering to the image."""
    image_array = np.array(image)
    return Image.fromarray(cv2.blur(image_array, (kernel_size, kernel_size)))

def median_filter(image, kernel_size=3):
    """Apply median filtering to the image."""
    image_array = np.array(image)
    return Image.fromarray(cv2.medianBlur(image_array, kernel_size))

def augment_and_process(image_path, output_dir, folder, current_index):
    try:
        pil_image = Image.open(image_path)
        if pil_image is None:
            print(f"Skipping {image_path} as face processing failed.")
            return current_index

        lighting_conditions = simulate_lighting_conditions(pil_image)

        transformations = {
            "Original": lambda x: x,
            # "Blur": lambda x: Image.fromarray(add_motion_blur(x)),
            # "Shadow": lambda x: Image.fromarray(simulate_shadow(x)),
            # "GammaCorrection": lambda x: Image.fromarray(apply_gamma_correction(x, gamma=random.uniform(0.8, 1.5))),
            "Rotation5": transforms.Compose([lambda x: F.rotate(x, -5)]),
            "Rotation10": transforms.Compose([lambda x: F.rotate(x, -10)]),
            "Rotation15": transforms.Compose([lambda x: F.rotate(x, -15)]),
            "Rotation20": transforms.Compose([lambda x: F.rotate(x, -20)]),
            "Rotation25": transforms.Compose([lambda x: F.rotate(x, -25)]),
            "Rotation30": transforms.Compose([lambda x: F.rotate(x, -30)]),
            "Rotation35": transforms.Compose([lambda x: F.rotate(x, -35)]),
            "Rotation40": transforms.Compose([lambda x: F.rotate(x, -40)]),
            "RotationInv5": transforms.Compose([lambda x: F.rotate(x, 5)]),
            "RotationInv10": transforms.Compose([lambda x: F.rotate(x, 10)]),
            "RotationInv15": transforms.Compose([lambda x: F.rotate(x, 15)]),
            "RotationInv20": transforms.Compose([lambda x: F.rotate(x, 20)]),
            "RotationInv25": transforms.Compose([lambda x: F.rotate(x, 25)]),
            "Rotation30": transforms.Compose([lambda x: F.rotate(x, 30)]),
            "Rotation35": transforms.Compose([lambda x: F.rotate(x, 35)]),
            "Rotation40": transforms.Compose([lambda x: F.rotate(x, 40)]),
            "Crop": transforms.Compose([transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))]),
            "Zoom": lambda x: zoom_image(x, zoom_factor=random.uniform(1.1, 1.5)),
            "Translate": lambda x: translate_image(x, tx=random.randint(-20, 20), ty=random.randint(-20, 20)),
            "MeanFilter": lambda x: mean_filter(x, kernel_size=3),
            "MedianFilter": lambda x: median_filter(x, kernel_size=3),
        }

        output_subfolder = os.path.join(output_dir, folder)
        os.makedirs(output_subfolder, exist_ok=True)

        for name, transform in transformations.items():
            augmented = transform(pil_image)
            augmented_array = np.array(augmented)

            height, width = augmented_array.shape[:2]
            x_start, y_start = 0, 0
            x_end, y_end = width, height

            off_set = 15
            eyebrow_y = int(y_end * 0.269) + off_set
            roi_gray = augmented_array[eyebrow_y:y_end, x_start:x_end]

            # Save the final image
            save_path = os.path.join(
                output_subfolder,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_{name}_{current_index}.jpg"
            )
            cv2.imwrite(save_path, cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB))
            print(f"Transformation {name} image saved: {save_path}")
            current_index += 1

        for idx, (lighting_name, lighting_image) in enumerate(lighting_conditions):

            lighting_pil_image = Image.fromarray(lighting_image)
            for name, transform in transformations.items():
                augmented_image = transform(lighting_pil_image)

                if isinstance(augmented_image, Image.Image):
                    augmented_image = np.array(augmented_image)

                # Post-processing: crop ROI and save directly
                height, width = augmented_image.shape[:2]
                x_start, y_start = 0, 0
                x_end, y_end = width, height

                off_set = 15
                eyebrow_y = int(y_end * 0.269) + off_set
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

for i in range(17): 
    aug_input_dir = "dataset_crop_test"
    aug_output_dir = "test_augmented_dataset_crop"
    aug_target_folder = str(i)
 
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
