import serial
import time


tria = serial.Serial("/dev/ttyAMA0", 9600, timeout=1)
tria.flush()

while True:
  tria.write(b"Tria indoor climate control!\n")
  line = tria.readline().decode("utf-8").rstrip()
  print(">" + line)
  time.sleep(1)
