import cv2
import mediapipe as mp
import csv
import time
import os
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent

def process_videos():

    # Create the file paths
    mp4_file_path1 = r'data_processing\video_data\trimmed_out_0.mp4'
    mp4_file_path2 = r'data_processing\video_data\trimmed_out_1.mp4'


    ### VIDEO 1 ###


    # Initialize MediaPipe Pose
    csv_file_name = 'amateur_swing_data1.csv'
    csv_file_path = os.path.join(OUTPUT_PATH, csv_file_name)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Create the CSV file and write the header row
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header_row = ['Frame']
        for i in range(33):  # 33 landmarks in MediaPipe Pose
            header_row.extend([f'Landmark_{i}_x', f'Landmark_{i}_y', f'Landmark_{i}_z', f'Landmark_{i}_visibility'])
        csv_writer.writerow(header_row)
    frame_count = 0

    # Process the edited MP4 file
    cap = cv2.VideoCapture(mp4_file_path1)

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(frame_rgb)

            # Extract the skeleton data
            if results.pose_landmarks:
                # Write the skeleton data to the CSV file
                with open(csv_file_path, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    row = [frame_count]
                    for landmark in results.pose_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    csv_writer.writerow(row)

            frame_count += 1

    # Release the video capture
    cap.release()
    #print(f"Amateur swing data has been extracted from the edited video and saved to '{csv_file_path}'.")


    ### VIDEO 2 ###


    # Initialize MediaPipe Pose
    csv_file_name = 'amateur_swing_data2.csv'
    csv_file_path = os.path.join(OUTPUT_PATH, csv_file_name)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Create the CSV file and write the header row
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header_row = ['Frame']
        for i in range(33):  # 33 landmarks in MediaPipe Pose
            header_row.extend([f'Landmark_{i}_x', f'Landmark_{i}_y', f'Landmark_{i}_z', f'Landmark_{i}_visibility'])
        csv_writer.writerow(header_row)
    frame_count = 0

    # Process the edited MP4 file
    cap = cv2.VideoCapture(mp4_file_path2)

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(frame_rgb)

            # Extract the skeleton data
            if results.pose_landmarks:
                # Write the skeleton data to the CSV file
                with open(csv_file_path, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    row = [frame_count]
                    for landmark in results.pose_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    csv_writer.writerow(row)

            frame_count += 1

    # Release the video capture
    cap.release()
    #print(f"Amateur swing data has been extracted from the edited video and saved to '{csv_file_path}'.")



### Functions for the code to work ###
    
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_rotation(point1, point2, reference_vector):
    vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    dot_product = np.dot(vector, reference_vector)
    magnitude = np.linalg.norm(vector) * np.linalg.norm(reference_vector)
    angle = np.arccos(dot_product / magnitude)
    return np.degrees(angle)

def preprocess_csv_data(csv_file):
    data = pd.read_csv(csv_file)
    
    angles_data = []
    reference_vector = np.array([1, 0])  # x-axis as the reference vector
    
    prev_video_number = None
    frame_counter = 0
    
    for _, row in data.iterrows():
        video_number = row['Video']
        
        if prev_video_number is None or video_number != prev_video_number:
            frame_counter = 0
            prev_video_number = video_number
        
        front_arm_points = [
            (row['Landmark_11_x'], row['Landmark_11_y']),
            (row['Landmark_13_x'], row['Landmark_13_y']),
            (row['Landmark_15_x'], row['Landmark_15_y'])
        ]
        hip_point1 = (row['Landmark_23_x'], row['Landmark_23_y'])
        hip_point2 = (row['Landmark_24_x'], row['Landmark_24_y'])
        shoulder_point1 = (row['Landmark_11_x'], row['Landmark_11_y'])
        shoulder_point2 = (row['Landmark_12_x'], row['Landmark_12_y'])
        
        front_arm_angle = calculate_angle(*front_arm_points)
        hip_rotation = calculate_rotation(hip_point1, hip_point2, reference_vector)
        shoulder_rotation = calculate_rotation(shoulder_point1, shoulder_point2, reference_vector)
        
        angles_data.append([video_number, frame_counter, front_arm_angle, hip_rotation, shoulder_rotation])
        
        frame_counter += 1
    
    angles_df = pd.DataFrame(angles_data, columns=['golfer_id', 'frame', 'front_arm_angle', 'hip_rotation', 'shoulder_rotation'])
    
    #inserting this here because the model was trained witha collom that said the video ID 
    angles_df.insert(0, 'Video', 0)
    
    return angles_df

def prepare_training_data(angles_df, sequence_length):
    golfer_ids = angles_df['golfer_id'].unique()
    
    input_sequences = []
    target_sequences = []
    
    for golfer_id in golfer_ids:
        golfer_data = angles_df[angles_df['golfer_id'] == golfer_id]
        golfer_angles = golfer_data[['front_arm_angle', 'hip_rotation', 'shoulder_rotation']].values
        
        num_sequences = len(golfer_angles) - sequence_length
        for i in range(num_sequences):
            input_seq = golfer_angles[i:i+sequence_length]
            target_seq = golfer_angles[i+sequence_length]
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
    
    input_sequences = np.array(input_sequences)
    target_sequences = np.array(target_sequences)
    
    return input_sequences, target_sequences

def compare_swings(model, amateur_sequences):
    predicted_sequences = model.predict(amateur_sequences)
    
    mse_values = []
    for amateur_seq, predicted_seq in zip(amateur_sequences, predicted_sequences):
        min_length = min(len(amateur_seq), len(predicted_seq))
        amateur_seq = amateur_seq[:min_length]
        predicted_seq = predicted_seq[:min_length]
        mse = np.mean(np.square(amateur_seq - predicted_seq))
        mse_values.append(mse)
    
    average_mse = np.mean(mse_values)
    
    if average_mse < 100:  # Adjust the threshold as needed
        print("Good swing!")
    else:
        print("Needs improvement.")


### PROCESS THE AMATEUR SWING DATA CSV WE RECORDED ###
        
def process_amateur_swing_data():


    ### DATA 1 ###


    csv_file_name = 'amateur_swing_data1.csv'
    file_path = os.path.join(OUTPUT_PATH, csv_file_name)
    sequence_length = 1

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

        num_rows = len(data)
        data[0].insert(0, "Video")
        for i in range(1, num_rows):
            data[i].insert(0, "0")

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    ##############
    amateur_angles_df1 = preprocess_csv_data(file_path)

    new_file_path = os.path.join(OUTPUT_PATH, r'angle_csv/amateur_angles1.csv')
    amateur_angles_df1.to_csv(new_file_path, index=False)
    # Prepare the amateur golfer's sequences
    amateur_sequences1 = prepare_training_data(amateur_angles_df1, sequence_length)


    ### DATA 2 ###

    csv_file_name = 'amateur_swing_data2.csv'
    file_path = os.path.join(OUTPUT_PATH, csv_file_name)
    sequence_length = 1

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

        num_rows = len(data)
        data[0].insert(0, "Video")
        for i in range(1, num_rows):
            data[i].insert(0, "0")

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    ##############
    amateur_angles_df2 = preprocess_csv_data(file_path)

    new_file_path = os.path.join(OUTPUT_PATH, r'angle_csv/amateur_angles2.csv')
    amateur_angles_df2.to_csv(new_file_path, index=False)
    # Prepare the amateur golfer's sequences
    amateur_sequences2 = prepare_training_data(amateur_angles_df2, sequence_length)




### FUNCTIONS FOR MAKING THE SKELETON POINT CLOUD

def visualize_body_positions(csv_file_path, output_video_path, background_image_path, fps=60, frame_size=(640, 640)):
    # Load the background image
    background_image = cv2.imread(background_image_path)
    # Resize the background image to match the frame size
    background_image = cv2.resize(background_image, (frame_size[0], frame_size[1]))

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Skip the header row

        # Extract the landmark names from the header
        landmark_names = header[2::4]  # Skip 'Video', 'Frame', and take every 4th element

        prev_video_number = None
        skeleton_frames = []

        for row in csv_reader:
            video_number = int(row[0])
            frame_number = int(row[1])

            if prev_video_number is None:
                prev_video_number = video_number

            if video_number != prev_video_number:
                # Visualize the skeleton frames for the previous video
                for frame in skeleton_frames:
                    # Use the background image
                    frame_background = background_image.copy()

                    # Draw the skeleton points on the background image
                    for i, landmark_name in enumerate(landmark_names):
                        if i * 4 + 1 < len(frame):  # Check if the required data is available
                            x = int(float(frame[i * 4]) * frame_size[0])
                            y = int(float(frame[i * 4 + 1]) * frame_size[1])
                            cv2.circle(frame_background, (x, y), 5, (0, 255, 0), -1)

                    # Write the frame with the background to the output video
                    video_writer.write(frame_background)

                # Reset the skeleton frames for the new video
                skeleton_frames = []
                prev_video_number = video_number

            # Extract the landmark coordinates for the current frame
            frame_landmarks = row[2:]
            skeleton_frames.append(frame_landmarks)

        # Visualize the skeleton frames for the last video
        for frame in skeleton_frames:
            # Use the background image
            frame_background = background_image.copy()

            # Draw the skeleton points on the background image
            for i, landmark_name in enumerate(landmark_names):
                if i * 4 + 1 < len(frame):  # Check if the required data is available
                    x = int(float(frame[i * 4]) * frame_size[0])
                    y = int(float(frame[i * 4 + 1]) * frame_size[1])
                    cv2.circle(frame_background, (x, y), 10, (0, 0, 255), -1)

            # Write the frame with the background to the output video
            video_writer.write(frame_background)

    # Release the VideoWriter object
    video_writer.release()

    #print("Skeleton video has been saved to", output_video_path)




### EXE WHERE WE LOAD THE MODEL ###
    
def process_amateur_video(video_path, angles_data, model, sequence_length, output_path, infile, out_folder):
    cap = cv2.VideoCapture(video_path)
     
    # Initialize variables for feedback and counter
    feedback_counter = 0
    feedback_frames = 5
    front_arm_angle_feedback_count = 0
    hip_rotation_feedback_count = 0
    shoulder_rotation_feedback_count = 0
    # Initialize feedback variables with default values
    front_arm_angle_feedback = ""
    hip_rotation_feedback = ""
    shoulder_rotation_feedback = ""
     # Initialize lists to store the actual and predicted values
    front_arm_angle_actual = []
    front_arm_angle_predicted = []
    hip_rotation_actual = []
    hip_rotation_predicted = []
    shoulder_rotation_actual = []
    shoulder_rotation_predicted = []
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 60
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_counter = 0
    input_sequence = []
    
    time_ai = time.time()
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform pose detection
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Draw skeleton on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get the angles and rotations for the current frame from the CSV data
            if frame_counter < len(angles_data):
                front_arm_angle = angles_data.loc[frame_counter, 'front_arm_angle']
                hip_rotation = angles_data.loc[frame_counter, 'hip_rotation']
                shoulder_rotation = angles_data.loc[frame_counter, 'shoulder_rotation']
                
                input_sequence.append([front_arm_angle, hip_rotation, shoulder_rotation])
                
                #print(f"{len(input_sequence)} == {sequence_length}")
                if len(input_sequence) == sequence_length:

                    # Prepare input sequence for prediction
                    input_sequence_array = np.array(input_sequence)
                    input_sequence_array = np.expand_dims(input_sequence_array, axis=0)
                    
                    # Make prediction using the model
                    predicted_angles_rotations = model.predict(input_sequence_array)
                    
                    # Extract the predicted angles and rotations
                    front_arm_angle_pred, hip_rotation_pred, shoulder_rotation_pred = predicted_angles_rotations[0]
                    
                    front_arm_angle_actual.append(front_arm_angle)
                    front_arm_angle_predicted.append(front_arm_angle_pred)
                    hip_rotation_actual.append(hip_rotation)
                    hip_rotation_predicted.append(hip_rotation_pred)
                    shoulder_rotation_actual.append(shoulder_rotation)
                    shoulder_rotation_predicted.append(shoulder_rotation_pred)
                    
                    
                    
                    
                    front_arm_angle_diff = abs(front_arm_angle - front_arm_angle_pred)
                    hip_rotation_diff = abs(hip_rotation - hip_rotation_pred)
                    shoulder_rotation_diff = abs(shoulder_rotation - shoulder_rotation_pred)
                    
                    
                    
                    # Set acceptable difference thresholds
                    front_arm_angle_threshold = 20
                    hip_rotation_threshold = 25
                    shoulder_rotation_threshold = 25
                    
                    # Determine text color based on the differences
                    front_arm_angle_color = (0, 255, 0) if front_arm_angle_diff < front_arm_angle_threshold else (0, 0, 255)
                    hip_rotation_color = (0, 255, 0) if hip_rotation_diff < hip_rotation_threshold else (0, 0, 255)
                    shoulder_rotation_color = (0, 255, 0) if shoulder_rotation_diff < shoulder_rotation_threshold else (0, 0, 255)
                    
                    # Display angle values on the frame with corresponding colors
                    cv2.putText(frame, f"Front Arm Angle: {front_arm_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, front_arm_angle_color, 2)
                    cv2.putText(frame, f"Hip Rotation: {hip_rotation:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hip_rotation_color, 2)
                    cv2.putText(frame, f"Shoulder Rotation: {shoulder_rotation:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, shoulder_rotation_color, 2)

                    # Remove the oldest entry from the input sequence
                    input_sequence = input_sequence[1:]
            
            # Write the frame to the output video
            out.write(frame)
            
            frame_counter += 1
    
    time_ai = time.time() - time_ai
    cap.release()
    out.release()

    # Plot the actual vs predicted values
    # Commented out, this is for displaying values
    """
    frames = range(len(front_arm_angle_actual))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(frames, front_arm_angle_actual, label='Actual')
    plt.plot(frames, front_arm_angle_predicted, label='Predicted')
    plt.xlabel('Frame')
    plt.ylabel('Front Arm Angle')
    plt.legend()
   
    plt.subplot(1, 3, 2)
    plt.plot(frames, hip_rotation_actual, label='Actual')
    plt.plot(frames, hip_rotation_predicted, label='Predicted')
    plt.xlabel('Frame')
    plt.ylabel('Hip Rotation')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(frames, shoulder_rotation_actual, label='Actual')
    plt.plot(frames, shoulder_rotation_predicted, label='Predicted')
    plt.xlabel('Frame')
    plt.ylabel('Shoulder Rotation')
    plt.legend()
  
    plt.tight_layout()
    plt.show()
    """

    # Define the directory
    graphs_dir = os.path.join(OUTPUT_PATH, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    graphs_with_video_path = os.path.join(graphs_dir, video_name)

    # Plot the actual vs predicted values
    frames = range(len(front_arm_angle_actual))
    
    plt.figure(figsize=(8, 4))
    plt.plot(frames, front_arm_angle_actual, label='Actual')
    plt.plot(frames, front_arm_angle_predicted, label='Predicted')
    plt.xlabel('Frame')
    plt.ylabel('Front Arm Angle')
    plt.legend()
    plot_filename = graphs_with_video_path + '_angle_arm_pred.png'
    plt.savefig(plot_filename)
    plt.clf()
    
    plt.figure(figsize=(8, 4))
    plt.plot(frames, hip_rotation_actual, label='Actual')
    plt.plot(frames, hip_rotation_predicted, label='Predicted')
    plt.xlabel('Frame')
    plt.ylabel('Hip Rotation')
    plt.legend()
    plot_filename = graphs_with_video_path + '_angle_hip_pred.png'
    plt.savefig(plot_filename)
    plt.clf()

    plt.figure(figsize=(8, 4))
    plt.plot(frames, shoulder_rotation_actual, label='Actual')
    plt.plot(frames, shoulder_rotation_predicted, label='Predicted')
    plt.xlabel('Frame')
    plt.ylabel('Shoulder Rotation')
    plt.legend()
    plot_filename = graphs_with_video_path + '_angle_shoulder_pred.png'
    plt.savefig(plot_filename)
    plt.clf()

    return time_ai


### PROCESS CAM VIDEO
def process_cam_video(video_path, angles_data, model, sequence_length, output_path, infile, out_folder):
    cap = cv2.VideoCapture(video_path)
     
    # Initialize variables for feedback and counter
    feedback_counter = 0
    feedback_frames = 5
    front_arm_angle_feedback_count = 0
    hip_rotation_feedback_count = 0
    shoulder_rotation_feedback_count = 0
    # Initialize feedback variables with default values
    front_arm_angle_feedback = ""
    hip_rotation_feedback = ""
    shoulder_rotation_feedback = ""
     # Initialize lists to store the actual and predicted values
    front_arm_angle_actual = []
    front_arm_angle_predicted = []
    hip_rotation_actual = []
    hip_rotation_predicted = []
    shoulder_rotation_actual = []
    shoulder_rotation_predicted = []
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    #fps = 60
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_counter = 0
    input_sequence = []
    
    
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform pose detection
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Draw skeleton on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get the angles and rotations for the current frame from the CSV data
            if frame_counter < len(angles_data):
                front_arm_angle = angles_data.loc[frame_counter, 'front_arm_angle']
                hip_rotation = angles_data.loc[frame_counter, 'hip_rotation']
                shoulder_rotation = angles_data.loc[frame_counter, 'shoulder_rotation']
                
                input_sequence.append([front_arm_angle, hip_rotation, shoulder_rotation])
                
                if len(input_sequence) == sequence_length:
                    # Prepare input sequence for prediction
                    input_sequence_array = np.array(input_sequence)
                    input_sequence_array = np.expand_dims(input_sequence_array, axis=0)
                    
                    # Make prediction using the model
                    
                    
    
                    front_arm_angle_actual.append(front_arm_angle)
                    
                    hip_rotation_actual.append(hip_rotation)
                    
                    shoulder_rotation_actual.append(shoulder_rotation)
                
                    
                    
                    front_arm_angle_diff = front_arm_angle 
                    hip_rotation_diff = hip_rotation 
                    shoulder_rotation_diff = shoulder_rotation 
                    
                    
                    
                    # Set acceptable difference thresholds
                    front_arm_angle_threshold = 20
                    hip_rotation_threshold = 25
                    shoulder_rotation_threshold = 25
                    
                    # Determine text color based on the differences
                    front_arm_angle_color = (0, 255, 0) 
                    hip_rotation_color = (0, 255, 0) 
                    shoulder_rotation_color = (0, 255, 0) 
                    # Display angle values on the frame with corresponding colors
                    cv2.putText(frame, f"Front Arm Angle: {front_arm_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, front_arm_angle_color, 2)
                    cv2.putText(frame, f"Hip Rotation: {hip_rotation:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hip_rotation_color, 2)
                    cv2.putText(frame, f"Shoulder Rotation: {shoulder_rotation:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, shoulder_rotation_color, 2)

                    cv2.putText(frame, f"Frame: {frame_counter}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    # Remove the oldest entry from the input sequence
                    input_sequence = input_sequence[1:]
            
            # Write the frame to the output video
            out.write(frame)
            
            frame_counter += 1
    
    cap.release()
    out.release()


    # Plot the actual vs predicted values
    # Commented out the popup graphs
    """
    frames = range(len(front_arm_angle_actual))

    plt.figure(figsize=(12, 4))

    ideal_front_arm_angle = 175
   
    plt.subplot(1, 3, 1)
    plt.plot(frames, front_arm_angle_actual, label='Actual')
    plt.axhline(y=ideal_front_arm_angle, color='r', linestyle='--', label='Ideal Impact Angle')
    plt.xlabel('Frame')
    plt.ylabel('Front Arm Angle')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(frames, hip_rotation_actual, label='Actual')
    plt.xlabel('Frame')
    plt.ylabel('Hip Rotation')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(frames, shoulder_rotation_actual, label='Actual')
    plt.xlabel('Frame')
    plt.ylabel('Shoulder Rotation')
    plt.legend()
    """

    # Define the directory
    graphs_dir = os.path.join(OUTPUT_PATH, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    graphs_with_video_path = os.path.join(graphs_dir, video_name)

    # Plot the actual vs predicted values
    frames = range(len(front_arm_angle_actual))
    ideal_front_arm_angle = 175
    
    plt.figure(figsize=(8, 4))
    plt.plot(frames, front_arm_angle_actual, label='Actual')
    plt.axhline(y=ideal_front_arm_angle, color='r', linestyle='--', label='Ideal Impact Angle')
    plt.xlabel('Frame')
    plt.ylabel('Front Arm Angle')
    plt.legend()
    plot_filename = graphs_with_video_path + '_angle_arm_act.png'
    plt.savefig(plot_filename)
    plt.clf()
    
    plt.figure(figsize=(8, 4))
    plt.plot(frames, hip_rotation_actual, label='Actual')
    plt.xlabel('Frame')
    plt.ylabel('Hip Rotation')
    plt.legend()
    plot_filename = graphs_with_video_path + '_angle_hip_act.png'
    plt.savefig(plot_filename)
    plt.clf()

    plt.figure(figsize=(8, 4))
    plt.plot(frames, shoulder_rotation_actual, label='Actual')
    plt.xlabel('Frame')
    plt.ylabel('Shoulder Rotation')
    plt.legend()
    plot_filename = graphs_with_video_path + '_angle_shoulder_act.png'
    plt.savefig(plot_filename)
    plt.clf()



### PROCESS VIDEOS
    
def process_videos_second_part():
    # Bookkeeping
    total_time_ai = 0
    start_time = time.time()

    ### MODEL LOADING
    file_path = os.path.join(OUTPUT_PATH, r'model.keras')
    model = load_model(file_path)


    ### VIDEO 1

    file_path1 = os.path.join(OUTPUT_PATH, r'amateur_swing_data1.csv')

    # Load the amateur angles data from the CSV file
    file_path = os.path.join(OUTPUT_PATH, r'angle_csv/amateur_angles1.csv')
    amateur_angles_data = pd.read_csv(file_path)

    # Process the amateur video and save the output
    amateur_video_path = r'data_processing\video_data\trimmed_out_0.mp4'
    output_video_path = os.path.join(OUTPUT_PATH, r'angle_videos\amateur_swing_analysis0.mp4')
    output_video_path_folder = os.path.join(OUTPUT_PATH, r'angle_videos\skeleton_video0.mp4')
    output_video_pathcam = os.path.join(OUTPUT_PATH, r'angle_videos\amateur_cam_analysis0.mp4')
    image_path = os.path.join(OUTPUT_PATH, r'bkg.png')

    # Make the green guy
    visualize_body_positions(file_path1, output_video_path_folder, image_path)

    # Skeleton video
    skell_video_path = os.path.join(OUTPUT_PATH, r'angle_videos\skeleton_video0.mp4')

    # Process green vid
    sequence_length = model.input_shape[1]
    total_time_ai += process_amateur_video(skell_video_path, amateur_angles_data, model, sequence_length, output_video_path, file_path1, output_video_path_folder)

    # Process Cam
    process_cam_video(amateur_video_path, amateur_angles_data, model, sequence_length, output_video_pathcam, file_path1, output_video_path_folder)


    ### VIDEO 2
    file_path1 = os.path.join(OUTPUT_PATH, r'amateur_swing_data2.csv')

    # Load the amateur angles data from the CSV file
    file_path = os.path.join(OUTPUT_PATH, r'angle_csv/amateur_angles2.csv')
    amateur_angles_data = pd.read_csv(file_path)

    # Process the amateur video and save the output
    amateur_video_path = r'data_processing\video_data\trimmed_out_1.mp4'
    output_video_path = os.path.join(OUTPUT_PATH, r'angle_videos\amateur_swing_analysis1.mp4')
    output_video_path_folder = os.path.join(OUTPUT_PATH, r'angle_videos\skeleton_video1.mp4')
    output_video_pathcam = os.path.join(OUTPUT_PATH, r'angle_videos\amateur_cam_analysis1.mp4')
    image_path = os.path.join(OUTPUT_PATH, r'bkg.png')

    # Make the green guy
    visualize_body_positions(file_path1, output_video_path_folder, image_path)

    # Skeleton video
    skell_video_path = os.path.join(OUTPUT_PATH, r'angle_videos\skeleton_video1.mp4')

    # Process green vid
    sequence_length = model.input_shape[1]
    total_time_ai += process_amateur_video(skell_video_path, amateur_angles_data, model, sequence_length, output_video_path, file_path1, output_video_path_folder)

    # Process Cam
    process_cam_video(amateur_video_path, amateur_angles_data, model, sequence_length, output_video_pathcam, file_path1, output_video_path_folder)

    # Print time
    print(f"Debug: Total time to process videos using Neural Network = {total_time_ai}")
    print(f"Debug: Total time to generate feedback = {time.time() - start_time - total_time_ai}")


def execute_process_video(callToGUI):
    process_videos()
    process_amateur_swing_data()
    process_videos_second_part()
    
    # Call back to the GUI
    if callToGUI:
        from gui_module.build.gui_generating_results import finish_processing_videos_from_AI
        finish_processing_videos_from_AI()

if __name__ == "__main__":
    execute_process_video(False)