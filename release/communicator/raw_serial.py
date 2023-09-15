import serial

tria  = serial.Serial(port="/dev/ttyAMA0", baudrate=9600, timeout=.1)
while True:
   if(tria > 0):
     line = tria.readline()
     print(line)
