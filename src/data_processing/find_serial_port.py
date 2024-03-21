def find_com():
    print("find_com needs to be implemented...")
    return 'COM3'

import bluetooth
import serial.tools.list_ports

def find_device_com():
    # Method 1: PyBluez
    # Discover Bluetooth devices
    devices = bluetooth.discover_devices()

    # Iterate over discovered devices
    for device_address in devices:
        device_name_lookup = bluetooth.lookup_name(device_address)
        print(device_name_lookup)

        # This doesn't work... I think version error?
        #services = bluetooth.find_service(address=device_address)
        #for service in services:
        #    print("Service Name:", service["name"])
        #    print("Service Class:", service["service-classes"])
        #    print("Host:", service["host"])
        #    print("Port:", service["port"])
        #    print("Description:", service["description"])
        #    print("Protocol:", service["protocol"])
        #    print("")

    # Method 2: PySerial
    available_ports = serial.tools.list_ports.comports()
    bluetooth_ports = []
    for port in available_ports:
        print(f"{port.name} = {port.description}")
    return None

# Example usage
if __name__ == "__main__":
    device_name = "HC-06"
    device_uuid = ""
    com_port = find_device_com()
    if com_port:
        print(f"The COM port connected to '{device_name}' is {com_port}")
    else:
        print(f"No COM port found for device '{device_name}'")