import cv2

def find_bounding_box(frame_path):
    frame = cv2.imread(frame_path)

    # Fail case
    if frame is None:
        print(f"Error: Failed to load image from '{frame_path}'")
        return None
    
    # Find the bounding box
    # <INSERT LOGIC HERE>
    bbox = [0, 0, 80, 80]

    # We have our result!
    return bbox