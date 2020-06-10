from threading import Thread, Event

from time import sleep
import threading, queue
import serial
import re

event = Event()
x_dwm = 0
y_dwm = 0
z_dwm = 0

ser=serial.Serial(
      port = '/dev/ttyACM0',
      baudrate = 115200,
      bytesize = serial.EIGHTBITS,
      parity = serial.PARITY_NONE,
      stopbits = serial.STOPBITS_ONE,
)
ser.write(b'\x0d'+b'\x0d'+b"lep\n")

def modify_variable(var):
  global x_dwm
  global y_dwm
  global z_dwm
  
  print("\n\nSignal sent to DWM\n\n")

  while True:
    out = ser.readline()
    data = re.split('[,\r\n]', out.decode('utf-8'))
    try:
      x_dwm = float(data[1])
      y_dwm = float(data[2])
      z_dwm = float(data[3])
      print("\nz\n")
      #percentage = int(data[4])
    except ValueError:
      continue
    if event.is_set():
        break
  print('Stop printing')


my_var = [1, 2, 3]
t = Thread(target=modify_variable, args=(ser, ))
t.start()
while True:
    try:
        print("X: ", x_dwm, "Y: ", y_dwm, "Z: ", z_dwm, "\n")
    except KeyboardInterrupt:
        event.set()
        break
t.join()