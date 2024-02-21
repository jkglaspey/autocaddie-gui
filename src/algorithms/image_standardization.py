import os
from image_augmentation import zero_pad_frame, crop_frame, create_augmented_directory
from bounding_box_annotation import find_bounding_box

def standardize_frames():
    img_dir = 'algorithms/cv/out/'
    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(img_dir, filename)

            # Create the augmented image
            augmented_img_path = os.path.join('algorithms/cv/annotated_frames/', filename)
            create_augmented_directory(img_path, augmented_img_path)

            # Find the bounding box of the user
            bbox = find_bounding_box(augmented_img_path)

            # Fail case: No bounding box for this image
            if bbox is None:
                print(f"Failed to determine a bounding box for '{filename}'")
                continue

            # Crop the image to the bounding box
            crop_frame(augmented_img_path, 0, 0, 80, 80)

            # Apply zero-padding
            zero_pad_frame(augmented_img_path, 160)


if __name__ == "__main__":
    standardize_frames()