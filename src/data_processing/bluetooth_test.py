import serial

# Define the serial port and baudrate
inport = 'COM4'
#outport = 'COM3'  # Change this to your serial port
baudrate = 9600  # Change this to match your device's baudrate

# Open the serial port
serIn = serial.Serial(inport, baudrate)
#serOut = serial.Serial(outport, baudrate)

try:
    print("Initializing...")
    while True:
        # Read a line from the serial port
        lineIn = serIn.readline().decode().strip()
        #lineOut = serOut.readline().decode().strip()
        
        # Print the received data
        print("In... Received:", lineIn)
        #print("Out... Received:", lineOut)
except KeyboardInterrupt:
    # Close the serial port when Ctrl+C is pressed
    serIn.close()
    #serOut.close()
