import threading
import queue
import random
import time
from scapy.all import sniff, IP, TCP, UDP, conf, L3socket

# Force L3socket for Windows compatibility without Npcap driver issues
conf.L3socket = L3socket

# --- Constants ---
# Colors
COLOR_TCP = (0, 255, 255)   # Cyan
COLOR_UDP = (255, 153, 0)   # Orange
COLOR_OTHER = (100, 100, 100) # Grey for others

# Packet Spawn Config
SPAWN_Z_START = 10.0
SPAWN_RANGE_X = 5.0
SPAWN_RANGE_Y = 5.0

class PacketObject:
    """
    Represents a visualized packet in 3D space.
    """
    def __init__(self, src_ip, dst_ip, protocol, size):
        self.src = src_ip
        self.dst = dst_ip
        self.protocol = protocol
        self.size = size
        
        # 3D Position
        # Randomize X/Y within a range to spread them out
        self.x = random.uniform(-SPAWN_RANGE_X, SPAWN_RANGE_X)
        self.y = random.uniform(-SPAWN_RANGE_Y, SPAWN_RANGE_Y)
        self.z = SPAWN_Z_START
        
        # Visual properties
        self.color = COLOR_OTHER
        if self.protocol == 'TCP':
            self.color = COLOR_TCP
        elif self.protocol == 'UDP':
            self.color = COLOR_UDP
            
        # Calculate speed based on size (Heavier = Slower)
        # Small packet (e.g. 64 bytes) -> Fast
        # Large packet (e.g. 1500 bytes) -> Slow
        # Base speed 0.2, min speed 0.05
        # 1500 is max mtu usually.
        self.speed = max(0.05, 0.3 - (self.size / 3000.0))

    def update(self):
        """
        Moves the packet towards the viewer (decreasing Z).
        Returns False if the packet has passed the viewer.
        """
        self.z -= self.speed
        # Kill when it passes the camera (approx -8.0)
        return self.z > -8.0

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
                
                # Create packet object
                pkt_obj = PacketObject(src, dst, proto, size)
                
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
            # On Windows, need to make sure we have a valid interface.
            # conf.use_pcap = True (default) - requires Npcap.
            # If Npcap is missing, this might throw.
            sniff(prn=process_packet, store=0)
        except Exception as e:
            print(f"Sniffer Error: {e}")
            print("Make sure Npcap is installed on Windows for Scapy.")

    def stop(self):
        self.running = False
