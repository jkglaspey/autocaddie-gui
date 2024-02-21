import os
import cv2
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

def process_all_frames():

    # Load MobileNetV2 model pre-trained on ImageNet
    model = MobileNetV2(weights='imagenet')

    # Get every frame within a directory
    img_dir = 'algorithms/cv/out/'
    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(img_dir, filename)

            # Format the image for MobileNetV2
            img = image.load_img(img_path, target_size = (224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Predict
            predictions = model.predict(x)

            print('Predicted:', decode_predictions(predictions, top=3)[0])

            # Extract bounding boxes and labels from predictions
            #bounding_boxes, labels = extract_bounding_boxes(preds)
            
            #person_idx = labels.index('person')
            #person_bbox = bounding_boxes[person_idx]

            # Draw bounding boxes on the frame
            #annotated_frame = draw_bounding_box(img, person_bbox)

            # Make new directory
            #tf.image.draw_bounding_box(x, )
            #success = cv2.imwrite(f"algorithms/cv/image_augmentation/annotated_frames/{filename}", annotated_frame)
            

def extract_bounding_boxes(preds):
    return 0, 0


def draw_bounding_box(img, bbox):
    return


if __name__ == "__main__":
    
    # run!
    process_all_frames()