# -*- coding: utf-8 -*-
# lsusb to check device name
# dmesg | grep "tty" to find port name
import serial, time
import urllib3
import json
from rpi_communicator import RPL


if __name__ == '__main__':
    baseUrl = 'http://127.0.0.1:8000/action?'
    http = urllib3.PoolManager()
    rpl = RPL(http, baseUrl)

    #print('Running. Press CTRL-C to exit.')
    with serial.Serial("/dev/ttyACM0", 9600, timeout=1) as tria:
        time.sleep(0.1) #wait for serial to open
        if tria.isOpen():
            print("{} connected!".format(tria.port))
            try:
                while True:
                    #cmd=input("Enter command : ")
                    #tria.write(cmd.encode())
                    #time.sleep(0.1) #wait for tria to answer
                    while tria.inWaiting()==0: pass
                    if  tria.inWaiting()>0: 
                        observations=tria.readline()
                        print('observation: ',observations)
                        tria.flushInput() #remove data after reading
                        obs = observations.split(':')
                        action = RPL.getAction(obs[0],obs[1],obs[2])
                        print('action: ', action)
                        tria.write(action)
            except KeyboardInterrupt:
                print("KeyboardInterrupt has been caught.")