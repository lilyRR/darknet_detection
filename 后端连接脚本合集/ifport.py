import dpkt
import os
import socket
import struct

tcp_or_udp ={}
tcp_or_udp[6] = 'tcp'
tcp_or_udp[17] = 'udp'
tcp_or_udp[1] = 'icmp'
tcp_or_udp[2] = 'igmp'

def parseRadius(filepath,hostip):
    vect=[0 for y in range(5000)]
    j=0
    fpcap = open(filepath,'rb')
    source_pcap_name = os.path.basename(filepath).split('.')[0]
    string_data = fpcap.read()
    #pcap文件包头解析
    pcap_header = {}
    pcap_header['magic_number'] = string_data[0:4]
    pcap_header['version_major'] = string_data[4:6]
    pcap_header['version_minor'] = string_data[6:8]
    pcap_header['thiszone'] = string_data[8:12]
    pcap_header['sigfigs'] = string_data[12:16]
    pcap_header['snaplen'] = string_data[16:20]
    pcap_header['linktype'] = string_data[20:24]

    #pcap头部长度 24字节    
    pcap_header_str = string_data[:24]
    
    #pcap文件的数据包解析
    packet_num = 0
    packet_data = []
    
    pcap_packet_header = {}
    i =24
    
    write_pcap = {}
    
    while(i<len(string_data)):
         
        #数据包头各个字段
        pcap_packet_header['GMTtime'] = string_data[i:i+4]
        pcap_packet_header['MicroTime'] = string_data[i+4:i+8]
        pcap_packet_header['caplen'] = string_data[i+8:i+12]
        pcap_packet_header['len'] = string_data[i+12:i+16]
        #求出此包的包长len
        packet_len = struct.unpack('I',pcap_packet_header['len'])[0]
        #写入此包数据
        packet_data = ''
        packet_data = string_data[i+16:i+16+packet_len]
        
        #数据部分以太帧读取
        ether = dpkt.ethernet.Ethernet(packet_data)
        #判断为传输层为ip
        if ether.type == dpkt.ethernet.ETH_TYPE_IP:
            ip = ether.data                 #ip数据包
            src = socket.inet_ntoa(ip.src)  #源ip
            dst = socket.inet_ntoa(ip.dst)  #目的ip
            if (dst==hostip):
                vect[j]=-1
                j+=1
            if (src==hostip):
                vect[j]=1
                j+=1
            
            # print ('src ',src)
            # print ('dst ',dst)
            #传输层数据
            data = ip.data
            ip_pro =  tcp_or_udp[ip.p]
        
        i = i+ packet_len+16

        packet_num+=1
    return vect

filepath = 'D:\\b\\aimchat.pcap'
    
vect=parseRadius(filepath,"5.9.28.6")
print(vect)