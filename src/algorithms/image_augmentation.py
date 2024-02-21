import cv2

# Save an image to a directory
# Precondition: old_path has read a valid image
def create_augmented_directory(old_path, new_path):
    cv2.imwrite(new_path, cv2.imread(old_path))

# Crop a frame to a bounding box
def crop_frame(frame_path, x, y, width, height):
    frame = cv2.imread(frame_path)

    # Fail case
    if frame is None:
        print(f"Error: Failed to load image from '{frame_path}'")
        return
    
    # Ensure dimensions of cropping are valid
    img_height, img_width, _ = frame.shape
    if y + height > img_height or x + width > img_width:
        print(f"Error: cropping dimensions of [{y + height}, {x + width}] are larger than the image dimensions of [{img_height}, {img_width}]")
        return

    # Apply cropping
    cropped_frame = frame[y:y + height, x:x + width]

    # Override previous augmented image
    cv2.imwrite(frame_path, cropped_frame)

# Apply zero-padding to a frame (target size represents a square dimension)
def zero_pad_frame(frame_path, target_size):
    frame = cv2.imread(frame_path)

    # Fail case
    if frame is None:
        print(f"Error: Failed to load image from '{frame_path}'")
        return
    
    # Get padding dimensions
    height, width, _ = frame.shape
    pad_height = max(0, (target_size - height) // 2)
    pad_width = max(0, (target_size - width) // 2)

    # Pad the image
    padded_frame = cv2.copyMakeBorder(frame, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Override previous augmented image
    cv2.imwrite(frame_path, padded_frame)