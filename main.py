import pygame
import cv2
import mediapipe as mp
import queue
import protocol
import sounds
import time
import numpy as np
import math
import psutil
import collections

# --- Constants ---
SCALE = 120
EYE_DIST = 8.0
WIN_WIDTH = 800
WIN_HEIGHT = 600
GRID_SIZE = 4
GRID_SPACING = 2.0

# Colors (Lain Palette)
COL_CYAN = (0, 255, 255)
COL_RED = (255, 50, 50)
COL_DARK = (10, 10, 10)
COL_GRID = (0, 50, 100)

# --- Head Tracking Logic ---
def get_head(cap, face_mesh):
    ok, frame = cap.read()
    if not ok: return 0, 0, False
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks: return 0, 0, False
    pt = res.multi_face_landmarks[0].landmark[168]
    return (pt.x - 0.5) * 2, (pt.y - 0.5) * 2, True

# --- Projection Logic ---
def project(x, y, z, hx, hy, WIDTH, HEIGHT):
    d = EYE_DIST + z
    if d == 0: d = 0.001
    f = EYE_DIST / d
    sx = hx + (x - hx) * f
    sy = hy + (y - hy) * f
    px = int(WIDTH/2 + sx * SCALE)
    py = int(HEIGHT/2 + sy * SCALE)
    return px, py

class PsycheChip:
    def __init__(self):
        self.angle = 0.0
        self.vertices = [
            (0, -1, 0), (0, 1, 0),
            (-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)
        ]
        self.edges = [
            (0,2), (0,3), (0,4), (0,5),
            (1,2), (1,3), (1,4), (1,5),
            (2,3), (3,4), (4,5), (5,2)
        ]

    def update(self):
        self.angle += 0.02

    def draw(self, surface, center_x, center_y, scale=40):
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        projected_points = []
        for x, y, z in self.vertices:
            rx = x * cos_a - z * sin_a
            rz = x * sin_a + z * cos_a
            dist = 4.0 + rz 
            if dist == 0: dist = 0.1
            f = 300 / dist
            px = center_x + int(rx * scale * f * 0.01)
            py = center_y + int(y * scale * f * 0.01)
            projected_points.append((px, py))

        for start_idx, end_idx in self.edges:
            p1 = projected_points[start_idx]
            p2 = projected_points[end_idx]
            pygame.draw.line(surface, COL_RED, p1, p2, 2)

class WiredWindow:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Navi v7.0 // System Monitor")
        self.clock = pygame.time.Clock()
        self.running = True

        # Audio
        drone_data = sounds.generate_drone(duration=2.0)
        self.drone_sound = pygame.sndarray.make_sound(drone_data)
        self.drone_sound.play(loops=-1)
        self.drone_sound.set_volume(0.3)
        
        screech_data = sounds.generate_screech()
        self.screech_sound = pygame.sndarray.make_sound(screech_data)
        self.screech_sound.set_volume(0.4)
        self.burst_timer = 0

        # Fonts
        self.font = pygame.font.SysFont("Consolas", 12)
        self.font_large = pygame.font.SysFont("Consolas", 20, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 40, bold=True)

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)
        
        # Data
        self.packet_queue = queue.Queue(maxsize=100)
        self.packets = []
        self.last_packet_time = time.time()
        self.last_interaction_time = time.time() # For idle check
        
        self.listener = protocol.ProtocolListener(self.packet_queue)
        self.listener.start()

        self.hx, self.hy = 0, 0
        self.chip = PsycheChip()

        # --- NEW: System Stats History ---
        self.traffic_history = collections.deque([0]*100, maxlen=100)
        self.cpu_history = collections.deque([0]*50, maxlen=50)
        self.sys_stats = {"cpu": 0, "ram": 0, "net_sent": 0, "net_recv": 0}
        self.last_stat_update = 0

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def update_system_stats(self):
        # Update stats every 1 second (too expensive to do every frame)
        if time.time() - self.last_stat_update > 1.0:
            self.sys_stats["cpu"] = psutil.cpu_percent()
            self.sys_stats["ram"] = psutil.virtual_memory().percent
            
            # Net IO
            net = psutil.net_io_counters()
            self.sys_stats["net_sent"] = net.bytes_sent
            self.sys_stats["net_recv"] = net.bytes_recv
            
            self.last_stat_update = time.time()
        
        # Smoothly interpolate history for graphs
        self.cpu_history.append(self.sys_stats["cpu"])

    def update(self):
        # 1. Head Tracking
        new_hx, new_hy, found = get_head(self.cap, self.face_mesh)
        if found:
            # Check if head moved significantly (Wake up from idle)
            if abs(new_hx - self.hx) > 0.1 or abs(new_hy - self.hy) > 0.1:
                self.last_interaction_time = time.time()
            self.hx, self.hy = new_hx, new_hy

        # 2. System Stats
        self.update_system_stats()

        # 3. Packets
        packets_this_frame = 0
        try:
            while True:
                pkt = self.packet_queue.get_nowait()
                self.packets.append(pkt)
                packets_this_frame += 1
                self.last_packet_time = time.time()
                self.last_interaction_time = time.time() # Wake up on packet
        except queue.Empty:
            pass
        
        # Update Traffic Graph
        self.traffic_history.append(packets_this_frame * 5) # Scale up for visibility

        if packets_this_frame > 3 and time.time() > self.burst_timer:
            self.screech_sound.play()
            self.burst_timer = time.time() + 2.0

        self.packets = [p for p in self.packets if p.update()]

    def draw_hud_rings(self, center_x, center_y):
        """Draws concentric data rings around the Psyche chip (CPU/RAM)"""
        # RAM Ring (Outer, Cyan)
        radius_ram = 60 + (self.sys_stats['ram'] * 0.5) # Grows with usage
        rect_ram = (center_x - radius_ram, center_y - radius_ram, radius_ram*2, radius_ram*2)
        pygame.draw.arc(self.screen, COL_CYAN, rect_ram, 0, 3.14 * 2 * (self.sys_stats['ram']/100), 2)
        
        # Label RAM
        ram_txt = self.font.render(f"MEM: {self.sys_stats['ram']}%", True, COL_CYAN)
        self.screen.blit(ram_txt, (center_x + radius_ram + 5, center_y))

        # CPU Ring (Inner, Red)
        radius_cpu = 40 + (self.sys_stats['cpu'] * 0.4)
        rect_cpu = (center_x - radius_cpu, center_y - radius_cpu, radius_cpu*2, radius_cpu*2)
        pygame.draw.arc(self.screen, COL_RED, rect_cpu, 0, 3.14 * 2 * (self.sys_stats['cpu']/100), 2)
        
        # Label CPU
        cpu_txt = self.font.render(f"CPU: {self.sys_stats['cpu']}%", True, COL_RED)
        self.screen.blit(cpu_txt, (center_x - radius_cpu - 60, center_y))

    def draw_traffic_graph(self):
        """Draws a scrolling line graph of network traffic at the bottom"""
        graph_h = 100
        graph_y = WIN_HEIGHT - graph_h - 10
        margin_left = 50
        
        # Draw Border
        pygame.draw.rect(self.screen, COL_GRID, (margin_left, graph_y, 700, graph_h), 1)
        label = self.font.render("NET_IO // HEARTBEAT", True, COL_GRID)
        self.screen.blit(label, (margin_left, graph_y - 15))

        # Draw Points
        pts = []
        for i, val in enumerate(self.traffic_history):
            # Scale x to fit 700px width
            x = margin_left + (i * (700 / 100))
            # Clamp y
            val = min(val, graph_h)
            y = (graph_y + graph_h) - val
            pts.append((x, y))
        
        if len(pts) > 1:
            pygame.draw.lines(self.screen, COL_CYAN, False, pts, 2)

    def draw(self):
        self.chip.update()
        self.screen.fill(COL_DARK)

        # 1. 3D GRID
        for x in range(-GRID_SIZE, GRID_SIZE + 1, 2): 
            # Floor
            pts = []
            for z in range(10):
                px, py = project(x*GRID_SPACING, 2.0, z*2.0, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
                pts.append((px, py))
            if len(pts)>1: pygame.draw.lines(self.screen, COL_GRID, False, pts, 1)
            
            # Ceiling
            pts_c = []
            for z in range(10):
                px, py = project(x*GRID_SPACING, -2.0, z*2.0, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
                pts_c.append((px, py))
            if len(pts_c)>1: pygame.draw.lines(self.screen, COL_GRID, False, pts_c, 1)

        # 2. PACKETS
        for pkt in self.packets:
            brightness = max(0.2, 1.0 - (pkt.z + 5.0) / 15.0)
            color = (
                int(pkt.color[0] * brightness),
                int(pkt.color[1] * brightness),
                int(pkt.color[2] * brightness)
            )
            px, py = project(pkt.x, pkt.y, pkt.z, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
            tx, ty = project(pkt.x, pkt.y, pkt.z + 2.0, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
            
            pygame.draw.line(self.screen, color, (px, py), (tx, ty), 2)
            if brightness > 0.6:
                txt = self.font.render(f"{pkt.protocol}::{pkt.dst}", True, color)
                self.screen.blit(txt, (px+10, py))

        # 3. PSYCHE CHIP & HUD RINGS
        cx, cy = project(0, 0, 5, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
        self.chip.draw(self.screen, cx, cy)
        self.draw_hud_rings(cx, cy)
        
        # 4. TRAFFIC GRAPH
        self.draw_traffic_graph()

        # 5. IDLE SCREEN ("Close the world")
        # Only show if NO PACKETS for 8 seconds AND NO HEAD MOVEMENT for 5 seconds
        idle_pkt = time.time() - self.last_packet_time
        idle_head = time.time() - self.last_interaction_time
        
        if idle_pkt > 8.0 and idle_head > 5.0:
            # Pulse Alpha
            pulse = (math.sin(time.time() * 2) + 1) / 2 # 0 to 1
            alpha = int(pulse * 255)
            
            msg = self.title_font.render("CLOSE THE WORLD", True, (255, 255, 255))
            msg2 = self.font_large.render("OPEN THE NEXT", True, (255, 255, 255))
            
            # Create surf with alpha support
            s = pygame.Surface(msg.get_size(), pygame.SRCALPHA)
            s.blit(msg, (0,0))
            s.set_alpha(alpha)
            
            s2 = pygame.Surface(msg2.get_size(), pygame.SRCALPHA)
            s2.blit(msg2, (0,0))
            s2.set_alpha(alpha)
            
            self.screen.blit(s, (WIN_WIDTH//2 - s.get_width()//2, WIN_HEIGHT//2 - 20))
            self.screen.blit(s2, (WIN_WIDTH//2 - s2.get_width()//2, WIN_HEIGHT//2 + 30))

        pygame.display.flip()

    def run_loop(self):
        while self.running:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        self.listener.stop()
        self.cap.release()
        pygame.quit()
        self.face_mesh.close()

if __name__ == "__main__":
    app = WiredWindow()
    app.run_loop()