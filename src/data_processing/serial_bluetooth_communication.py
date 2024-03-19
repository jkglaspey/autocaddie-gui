import serial

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
        print(f"Connecting to Serial Port {port} using baudrate {baud}.")
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
            print("Serial port is not open.")
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
            print("Closing connection.")
            ser_out.close()
        else:
            print("Serial port is not open.")
    except serial.SerialException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Start the Bluetooth connection on a separate thread
    ser_out = start_bluetooth(OUTPORT, BAUDRATE)
    
    # Wait for the user input to stop Bluetooth
    input("Press Enter to stop Bluetooth...")
    
    # Close the Bluetooth connection
    stop_bluetooth(ser_out)
