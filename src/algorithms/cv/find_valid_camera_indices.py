import cv2
import sys

def find_camera_indices():
    # Try to find valid incides in the range of 0-9. May vary depending on system.
    camera_indices = []
    for idx in range(10):
        try:
            camera = cv2.VideoCapture(idx)
            if camera.isOpened():
                print(f"Camera found at index {idx}")
                camera_indices.append(idx)
                camera.release()
        except cv2.error as e:
            #print(f"Error while accessing camera at index {idx}: {e}")
            pass
        except:
            pass
            
    return camera_indices

if __name__ == "__main__":
    camera_indices = find_camera_indices()
