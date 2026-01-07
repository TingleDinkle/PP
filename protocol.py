import threading
import queue
import random
import time
import math
from scapy.all import sniff, IP, TCP, UDP, ICMP, conf

# --- Constants ---
# Colors
COLOR_TCP = (0.0, 1.0, 1.0, 1.0)   # Cyan (normalized RGBA)
COLOR_UDP = (1.0, 153/255.0, 0.0, 1.0)   # Orange (normalized RGBA)
COLOR_ICMP = (1.0, 0.2, 0.8, 1.0) # Magenta/Pink for ICMP
COLOR_OTHER = (100/255.0, 100/255.0, 100/255.0, 1.0) # Grey for others (normalized RGBA)

# Packet Spawn Config
PACKET_ORBITAL_RADIUS_MIN = 5.0 # Min radius for packets to orbit outside the tunnel
PACKET_ORBITAL_RADIUS_MAX = 7.0 # Max radius for packets to orbit outside the tunnel
PACKET_ORBITAL_SPEED_MIN = 0.02 # Min orbital speed
PACKET_ORBITAL_SPEED_MAX = 0.05 # Max orbital speed

class PacketObject:
    """
    Represents a visualized packet in 3D space.
    """
    def __init__(self, src_ip, dst_ip, protocol, size, payload):
        self.src = src_ip
        self.dst = dst_ip
        self.protocol = protocol
        self.size = size
        self.payload = payload
        
        # 3D Position
        self.orbital_radius = random.uniform(PACKET_ORBITAL_RADIUS_MIN, PACKET_ORBITAL_RADIUS_MAX)
        self.angle = random.uniform(0, 2 * math.pi) # Random initial angle
        self.x = math.cos(self.angle) * self.orbital_radius
        self.y = math.sin(self.angle) * self.orbital_radius * 0.5 # Flattened Y-axis for elliptical orbit
        self.z = random.uniform(-5.0, -2.0) # Fixed Z-range

        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
        
        # Orbital speed
        self.orbital_speed = random.uniform(PACKET_ORBITAL_SPEED_MIN, PACKET_ORBITAL_SPEED_MAX)

        # Visual properties
        self.color = COLOR_OTHER
        if self.protocol == 'TCP':
            self.color = COLOR_TCP
        elif self.protocol == 'UDP':
            self.color = COLOR_UDP
        elif self.protocol == 'ICMP':
            self.color = COLOR_ICMP
            
        # self.speed is no longer used for Z-movement, but keep if for potential future use
        self.speed = max(0.05, 0.3 - (self.size / 3000.0))

    def update(self):
        """
        No longer moves the packet in Z, always returns True as movement is external.
        """
        # Movement is handled in main_gl.py
        return True

class ProtocolListener(threading.Thread):
    """
    Background thread to sniff network traffic using Scapy.
    """
    def __init__(self, packet_queue):
        super().__init__()
        self.packet_queue = packet_queue
        self.running = True
        self.daemon = True # Kill thread when main app exits

    def run(self):
        # Scapy sniff callback
        def process_packet(packet):
            if not self.running:
                return False # Stop sniffing
            
            if IP in packet:
                src = packet[IP].src
                dst = packet[IP].dst
                size = len(packet)
                proto = 'OTHER'
                
                if TCP in packet:
                    proto = 'TCP'
                elif UDP in packet:
                    proto = 'UDP'
                elif ICMP in packet:
                    proto = 'ICMP'
                
                # Extract payload summary
                payload_data = packet.summary()
                
                # Create packet object
                pkt_obj = PacketObject(src, dst, proto, size, payload_data)
                
                # Put in queue (non-blocking)
                try:
                    self.packet_queue.put(pkt_obj, block=False)
                except queue.Full:
                    pass # Drop packet if queue is full (backpressure)

        # Start sniffing
        # store=0 prevents Scapy from keeping all packets in memory
        # count=0 means infinity
        # timeout=1 allows checking self.running periodically if we were looping manually,
        # but with callback, we rely on the callback or stop_filter.
        # However, `sniff` blocks. We need a way to stop it cleanly or just let it die with the daemon.
        # Ideally, we use `stop_filter` lambda, but `daemon=True` is usually enough for simple apps.
        try:
            while self.running: # Loop to continuously sniff while thread is running
                sniff(prn=process_packet, store=0, iface=conf.iface, stop_filter=lambda x: not self.running)
        except Exception as e:
            print(f"Sniffer Error: {e}")
            print("Make sure Npcap is installed on Windows for Scapy.")

    def stop(self):
        self.running = False