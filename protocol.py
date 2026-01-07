import threading
import queue
import random
import math
from dataclasses import dataclass, field
from typing import Optional

from scapy.all import sniff, IP, TCP, UDP, ICMP, conf
import config

@dataclass
class PacketObject:
    """
    Represents a visualized packet in 3D space with physical properties.
    """
    src: str
    dst: str
    protocol: str
    size: int
    payload: str
    
    # 3D Position & Physics (Initialized in __post_init__)
    x: float = field(init=False)
    y: float = field(init=False)
    z: float = field(init=False)
    global_z: float = field(init=False, default=0.0) # Used by PacketSystem
    
    orbital_radius: float = field(init=False)
    angle: float = field(init=False)
    orbital_speed: float = field(init=False)
    
    # Visualization
    color: config.Color = field(init=False)
    lane_x: float = field(init=False, default=0.0)
    lane_y: float = field(init=False, default=0.0)

    def __post_init__(self):
        # Position Logic
        self.orbital_radius = random.uniform(config.PACKET_ORBITAL_RADIUS_MIN, config.PACKET_ORBITAL_RADIUS_MAX)
        self.angle = random.uniform(0, 2 * math.pi)
        self.orbital_speed = random.uniform(config.PACKET_ORBITAL_SPEED_MIN, config.PACKET_ORBITAL_SPEED_MAX)

        # Initial Orbit Position
        self.x = math.cos(self.angle) * self.orbital_radius
        self.y = math.sin(self.angle) * self.orbital_radius * 0.5
        self.z = random.uniform(-5.0, -2.0)

        # Color Mapping
        if self.protocol == 'TCP':
            self.color = config.COL_CYAN
        elif self.protocol == 'UDP':
            self.color = config.COL_ORANGE
        elif self.protocol == 'ICMP':
            self.color = config.COL_MAGENTA
        else:
            self.color = config.COL_GREY

class ProtocolListener(threading.Thread):
    """
    Background daemon for sniffing network traffic via Scapy.
    """
    def __init__(self, packet_queue: queue.Queue):
        super().__init__()
        self.packet_queue = packet_queue
        self.running = True
        self.daemon = True

    def run(self) -> None:
        def process_packet(packet) -> bool:
            if not self.running:
                return False
            
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
                
                pkt_obj = PacketObject(
                    src=src,
                    dst=dst,
                    protocol=proto,
                    size=size,
                    payload=packet.summary()
                )
                
                try:
                    self.packet_queue.put(pkt_obj, block=False)
                except queue.Full:
                    pass # Backpressure: Drop packet

        try:
            sniff(
                prn=process_packet,
                store=0,
                iface=conf.iface,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            print(f"Error: Scapy Sniffer failed: {e}")
            print("Ensure Npcap is installed in 'WinPcap API-compatible mode'.")

    def stop(self) -> None:
        self.running = False