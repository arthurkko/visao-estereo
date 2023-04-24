from pynmeagps import NMEAReader
import serial
from serial.tools.list_ports import comports
from beeprint import pp

# for port in comports():
#     print(port)

# s/ RTK:0
# c/ RTK:1
rover=0

# ROVER 1
# --------------------------------------------------------------------------------------------------- #
# Defining serial port to a variable

serial_port1 = '/dev/ttyACM0'
ser1 = serial.Serial(serial_port1, 57600)

# serial_port2 = '/dev/ttyACM1'
# ser2 = serial.Serial(serial_port2, 57600)

i=0
j=0
ll=0
igual=0
name_ll=[]
name=[]
count_base=0
count_rover=0
with open("./RTK/NMEAOutputs.txt", "a") as outputs:
    while 1:
        # Reading NMEA data
        msg1 = NMEAReader(ser1) 
        # msg2 = NMEAReader(ser2)   

        (_, parsed_data1) = msg1.read()
        pdict1 = parsed_data1.__dict__

        # (_, parsed_data2) = msg2.read()
        # pdict2 = parsed_data2.__dict__

        #print('######################################################################')
        # pp(pdict)
        i+=1

        if pdict1.get('lat') and pdict1.get('lon'):
            ll+=1
            count_base+=1
            if pdict1['_msgID'] not in name_ll:
                name_ll.append(pdict1['_msgID'])
        else:
            if pdict1['_msgID'] not in name:
                name.append(pdict1['_msgID'])

        if pdict1.get('lat') and pdict1.get('lon'):
            count_rover+=1

            
        # Getting only lat and lon as GLL
        # '_msgID' stands for the type of message
        if pdict1['_msgID'] == 'GLL':
            lat = pdict1['lat']
            lon = pdict1['lon']
            #print(lat,";",lon, file=outputs)
            j+=1

        # if pdict1['_msgID']==pdict2['_msgID']:
        #     igual+=1

        if j==100:
            break

print('i= ', i)
print('j= ', j)
print('ll= ', ll)
print('igual= ', igual)
print('name_ll: ', name_ll)
print('name: ', name)
print('count_base: ', count_base)
print('count_rover: ', count_rover)
