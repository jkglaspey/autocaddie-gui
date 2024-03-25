import serial.tools.list_ports
import bluetooth
import serial

def find_bt_device_by_name(device_name):
    devices = bluetooth.discover_devices(lookup_names=True)
    for addr, name in devices:
        if name == device_name:
            return addr
    return None

def find_com_port(bt_address):
    com_ports = serial.tools.list_ports.comports()
    for port, desc, hwid in com_ports:
        if bt_address.replace(':', '').upper() in hwid.upper():
            return port
    return None

# Connect to the COM port. Wrapper function
def connect_to_device(device_name):
    bt_address = find_bt_device_by_name(device_name)
    if bt_address:
        com_port = find_com_port(bt_address)
        if com_port:
            try:
                #ser = serial.Serial(com_port)
                #print(f"Connected to {device_name} on COM port {com_port}")
                #return ser  # Return the Serial object for further use
                return com_port

            except serial.SerialException:
                print(f"Failed to connect to {device_name}")
                return None
        else:
            print(f"COM port for {device_name} not found")
            return None
    else:
        print(f"{device_name} not found")
        return None

# Done through other files
def read_data_from_device(ser):
    if ser:
        try:
            while True:
                # Read a line from the serial port
                line = ser.readline().decode().strip()

                # Print the received data
                print("Received:", line)
        except KeyboardInterrupt:
            # Close the serial port when Ctrl+C is pressed
            ser.close()
    else:
        print("No valid serial connection")

def find_com():
    device_name = "HC-06"
    com_port = connect_to_device(device_name)
    if com_port:
        print(f"Debug: Connected via {com_port}")
        return com_port
        #read_data_from_device(device_name)
    else:
        print("Debug: Could not connect to COM!")
        return None