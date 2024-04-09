import serial
import time

# Define the serial port and baudrate globally
global OUTPORT, BAUDRATE
OUTPORT = 'COM3'  # Default Serial Port
BAUDRATE = 9600  # Default Baud Rate (will not be changed)

# Flag to indicate if the connection attempt is in progress
connection_in_progress = False

# Get the port and baudrate
def get_serial_info():
    return OUTPORT, BAUDRATE

# Open the serial port for receiving packets
def start_bluetooth(port, baud):
    global connection_in_progress
    try:
        print(f"Debug: Connecting to Serial Port {port} using baudrate {baud}.")
        connection_in_progress = True
        ser_out = serial.Serial(port, baud)
        connection_in_progress = False
        return ser_out
    except serial.SerialException as e:
        print(f"Error: {e}")
        connection_in_progress = False
        return None

# Receive a packet from the serial port
def receive_packet(ser_out):
    global connection_in_progress
    try:
        if ser_out:
            connection_in_progress = True
            line_out = ser_out.readline().decode().strip()
            connection_in_progress = False
            return line_out
        else:
            print("Error: Serial port is not open, and tried to receive packet.")
            connection_in_progress = False
            return None
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None
    
def clear_buffer(ser_out):
    if ser_out is not None:
        ser_out.reset_input_buffer()

def receive_packet_async(ser_out):
    global connection_in_progress
    try:
        if ser_out:
            connection_in_progress = True
            start_time = time.time()
            timeout = 0.001  # 1 millisecond timeout
            while True:
                if ser_out.in_waiting > 0:
                    try:
                        line_out = ser_out.readline().decode().strip()
                        connection_in_progress = False
                        return line_out
                    except serial.SerialException as e:
                        print(f"Error: {e}")
                        connection_in_progress = False
                        return None
                if time.time() - start_time > timeout:
                    # Timeout exceeded, return None
                    connection_in_progress = False
                    return None
        else:
            print("Error: Serial port is not open, and tried to receive packet.")
            connection_in_progress = False
            return None
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

# Close the serial port to terminate the connection
def stop_bluetooth(ser_out):
    global connection_in_progress
    try:
        if ser_out:
            # Wait until the connection attempt finishes
            while connection_in_progress:
                pass
            print("Debug: Closing serial connection.")
            ser_out.close()
        else:
            print("Error: Serial port is not open, and tried to close serial connection.")
    except serial.SerialException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Start the Bluetooth connection on a separate thread
    ser_out = start_bluetooth(OUTPORT, BAUDRATE)
    
    # Wait for the user input to stop Bluetooth
    input("Press Enter to stop Bluetooth...")
    
    # Close the Bluetooth connection
    stop_bluetooth(ser_out)
