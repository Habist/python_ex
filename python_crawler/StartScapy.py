import sys
from scapy.layers.inet import *
from scapy.layers.inet6 import *
from scapy.layers.dns import *
from scapy.sendrecv import *
from scapy.supersocket import *
from scapy.layers.l2 import *
from scapy.layers.dot11 import *
from scapy.utils import *
from scapy.config import *



idx = 0

def sniffing(filter):
    sniff(prn=showPacket, count=1)

def showPacket(packet):
    global idx
    idx += 1
    print("===============================절취선====================================")
    print(idx)
    #, hexdump(packet)
    #  packet.show()
    #return hexdump(packet)
    return packet.show()


sniffing("1")
a=sr1(IP(dst="127.0.0.1"))
a.show()
