import time
import json
from data_processing.serial_bluetooth_communication import receive_packet
from gui_module.build import gui_recording

# Bluetooth container
def receive_imu_data(ser_out):

    # Store bluetooth state
    global terminate_bluetooth
    terminate_bluetooth = False

    # Delay before recording (3 seconds)
    time.sleep(3)

    # Initialize list to store received data
    received_data = []

    # Record shoulder straightness (end signal sent by hardware)
    while terminate_bluetooth is False:

        # Poll infinitely
        line = receive_packet(ser_out)
        if line == None:
            continue

        # Start signal?
        if line == "END":
            connected_bluetooth = True

        # Append received data to list
        else:
            received_data.append(line)
        
    # Write the data
    with open(r"data_processing\imu_data\data.json", 'w') as f:
        # <Insert logic to format data into json style>

        # Dump the line
        json.dump(received_data, f)
    
    # Notify the GUI
    gui_recording.notify_end_bluetooth()

def send_end_signal():
    global terminate_bluetooth
    terminate_bluetooth = True