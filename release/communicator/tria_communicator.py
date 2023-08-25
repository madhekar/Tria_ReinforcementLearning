# -*- coding: utf-8 -*-
# lsusb to check device name
# dmesg | grep "tty" to find port name
import serial,time
if __name__ == '__main__':
    
    print('Running. Press CTRL-C to exit.')
    with serial.Serial("/dev/ttyACM0", 9600, timeout=1) as tria:
        time.sleep(0.1) #wait for serial to open
        if tria.isOpen():
            print("{} connected!".format(tria.port))
            try:
                while True:
                    cmd=input("Enter command : ")
                    tria.write(cmd.encode())
                    #time.sleep(0.1) #wait for tria to answer
                    while tria.inWaiting()==0: pass
                    if  tria.inWaiting()>0: 
                        answer=tria.readline()
                        print(answer)
                        tria.flushInput() #remove data after reading
            except KeyboardInterrupt:
                print("KeyboardInterrupt has been caught.")