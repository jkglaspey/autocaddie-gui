import time
import json
from data_processing.serial_bluetooth_communication import receive_packet
from gui_module.build import gui_recording
from algorithms.cv.convert_video_to_display import images

# Bluetooth container
def receive_imu_data(ser_out):
    print("Debug: Receiving IMU data.")

    # Store bluetooth state
    global terminate_bluetooth
    terminate_bluetooth = False

    # Initialize list to store received data
    prep_data_1 = []
    prep_data_2 = []
    swing_data_1 = []
    swing_data_2 = []

    # Record shoulder straightness (end signal sent by hardware)
    while terminate_bluetooth is False:

        # Poll infinitely
        line = receive_packet(ser_out)
        if line == None:
            continue

        # Record prep data
        if line == "PREP":
            images = []
            while terminate_bluetooth is False and line != "SWING":

                # Scan in 2 lines of input. Test each to see if order messed up
                line = receive_packet(ser_out)
                if line == "SWING":
                    continue
                line2 = receive_packet(ser_out)
                if line2 == "SWING":
                    line = "SWING"
                    continue

                # If the right lines are read, add them to the lists
                if line != None and line2 != None:
                    if line.split(",")[1] == "quat_1" and line2.split(",")[1] == "quat_2":
                        prep_data_1.append(line)
                        prep_data_2.append(line2)
        
        # Record swing data
        if line == "SWING":
            gui_recording.notify_start_videos()
            while terminate_bluetooth is False and line != "END":

                # Scan in 2 lines of input. Test each to see if order messed up
                line = receive_packet(ser_out)
                if line == "END":
                    continue
                line2 = receive_packet(ser_out)
                if line2 == "END":
                    line = "END"
                    continue

                # If the right lines are read, add them to the lists
                if line != None and line2 != None:
                    if line.split(",")[1] == "quat_1" and line2.split(",")[1] == "quat_2":
                        swing_data_1.append(line)
                        swing_data_2.append(line2)

        # End the recording
        if line == "END":

            # Are we good to dump the data?
            if terminate_bluetooth is True:
                return
            
            # Save the images
            gui_recording.notify_end_videos()

            # Write prep data for Q1
            with open(r"data_processing\imu_data\prep_data_q1.json", 'w') as f:
                json_data = []
                for line in prep_data_1:
                    data = line.split(",")
                    json_data.append({
                        "time": data[0],
                        "w": data[2],
                        "x": data[3],
                        "y": data[4],
                        "z": data[5]
                    })
                json.dump(json_data, f)

            # Write prep data for Q2
            with open(r"data_processing\imu_data\prep_data_q2.json", 'w') as f:
                json_data = []
                for line in prep_data_2:
                    data = line.split(",")
                    json_data.append({
                        "time": data[0],
                        "w": data[2],
                        "x": data[3],
                        "y": data[4],
                        "z": data[5]
                    })
                json.dump(json_data, f)

            # Write swing data for Q1
            with open(r"data_processing\imu_data\swing_data_q1.json", 'w') as f:
                json_data = []
                for line in swing_data_1:
                    data = line.split(",")
                    json_data.append({
                        "time": data[0],
                        "w": data[2],
                        "x": data[3],
                        "y": data[4],
                        "z": data[5]
                    })
                json.dump(json_data, f)

            # Write swing data for Q2
            with open(r"data_processing\imu_data\swing_data_q2.json", 'w') as f:
                json_data = []
                for line in swing_data_2:
                    data = line.split(",")
                    json_data.append({
                        "time": data[0],
                        "w": data[2],
                        "x": data[3],
                        "y": data[4],
                        "z": data[5]
                    })
                json.dump(json_data, f)
            
            # Notify the GUI
            gui_recording.notify_end_bluetooth()

            # We're done here
            return

def send_end_signal():
    global terminate_bluetooth
    terminate_bluetooth = True