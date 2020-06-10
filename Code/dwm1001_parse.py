import serial
import time
import re

ser=serial.Serial(
        port = 'COM6',
        baudrate = 115200,
        bytesize = serial.EIGHTBITS,
        parity = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
)
ser.write(b'\x0d'+b'\x0d'+b"lep\n")

while True:
        out=ser.readline()
        data = re.split('[,\r\n]', out.decode('utf-8'))
        try:
        	x= float(data[1])
        	y= float(data[2])
        	z= float(data[3])
        	percentage = int(data[4])
        except ValueError:
        	continue
        print(x," ",y," ",z," ",percentage,"\n")
        time.sleep(0.5) 
