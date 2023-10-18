# -*- coding: utf-8 -*-
# lsusb to check device name
# dmesg | grep "tty" to find port name
import serial, time
import urllib3
import json
from rpi_communicator import RPL

'''
Service hosted on raspberry pi to communicate with Tria device on serial interface.
This service also communicates with reinforcement model instance by providing
current sensor observations and receives an action.
'''
if __name__ == '__main__':
    baseUrl = 'http://127.0.0.1:8000/action?'
    http = urllib3.PoolManager()
    rpl = RPL(http, baseUrl)

    #print('Running. Press CTRL-C to exit.')
    with serial.Serial("/dev/ttyAMA0", 9600, timeout=1) as tria:
        time.sleep(0.1) #wait for serial to open
        if tria.isOpen():
            print("{} connected!".format(tria.port))
            try:
                while True:
                    while tria.inWaiting()==0: pass
                    if  tria.inWaiting()>0:
                        observations=tria.readline().decode('utf-8').rstrip()
                        tria.flushInput() #remove data after reading
                        if observations.startswith(">"):  #arduino sent debug message to display begins with >
                           print("Debug Msg: " + observations);
                        else:
                           obs = observations.split(':') # obserations are tokenized with : charactor
                           print("obs: ", obs)
                           action = rpl.getAction(t=float(obs[0]),h=float(obs[1]),a=float(obs[2]))
                           print('action: ', action)
                           tria.write(bytes(str(action), 'UTF-8'))
            except KeyboardInterrupt:
                print("KeyboardInterrupt has been caught.")
