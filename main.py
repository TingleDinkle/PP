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

# Colors
COL_CYAN = (0, 255, 255)
COL_RED = (255, 50, 50)
COL_DARK = (10, 10, 10)
COL_GRID = (0, 50, 100)
COL_TEXT = (200, 200, 200)

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
        pygame.display.set_caption("Navi v7.0 // Process Monitor")
        self.clock = pygame.time.Clock()
        self.running = True

        # Audio
        self.drone_sound = pygame.sndarray.make_sound(sounds.generate_drone(duration=2.0))
        self.drone_sound.play(loops=-1)
        self.drone_sound.set_volume(0.3)
        self.screech_sound = pygame.sndarray.make_sound(sounds.generate_screech())
        self.screech_sound.set_volume(0.4)
        self.burst_timer = 0

        # Fonts
        self.font = pygame.font.SysFont("Consolas", 12)
        self.font_small = pygame.font.SysFont("Consolas", 10)
        self.font_large = pygame.font.SysFont("Arial", 40, bold=True)

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)
        
        # Data
        self.packet_queue = queue.Queue(maxsize=100)
        self.packets = []
        self.listener = protocol.ProtocolListener(self.packet_queue)
        self.listener.start()

        self.hx, self.hy = 0, 0
        self.chip = PsycheChip()

        # Stats & Processes
        self.traffic_history = collections.deque([0]*100, maxlen=100)
        self.sys_stats = {"cpu": 0, "ram": 0}
        self.active_processes = [] # List of (name, pid)
        self.last_stat_update = 0
        
        # Intro Logic
        self.start_time = time.time()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: self.running = False

    def update_system_stats(self):
        if time.time() - self.last_stat_update > 2.0: # Check every 2s
            self.sys_stats["cpu"] = psutil.cpu_percent()
            self.sys_stats["ram"] = psutil.virtual_memory().percent
            
            # Get processes with active network connections
            procs = {}
            try:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.status == 'ESTABLISHED' and conn.pid:
                        try:
                            p = psutil.Process(conn.pid)
                            procs[p.name()] = conn.pid
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except Exception:
                pass
            
            # Convert to list and keep top 10
            self.active_processes = list(procs.items())[:10]
            self.last_stat_update = time.time()

    def update(self):
        new_hx, new_hy, found = get_head(self.cap, self.face_mesh)
        if found: self.hx, self.hy = new_hx, new_hy

        self.update_system_stats()

        packets_this_frame = 0
        try:
            while True:
                pkt = self.packet_queue.get_nowait()
                self.packets.append(pkt)
                packets_this_frame += 1
        except queue.Empty:
            pass
        
        self.traffic_history.append(packets_this_frame * 5)
        if packets_this_frame > 3 and time.time() > self.burst_timer:
            self.screech_sound.play()
            self.burst_timer = time.time() + 2.0

        self.packets = [p for p in self.packets if p.update()]

    def draw_active_processes(self, chip_x, chip_y):
        """Draws list of apps on the right, connected to the chip."""
        start_x = WIN_WIDTH - 160
        start_y = 100
        
        # Title
        title = self.font.render("ACTIVE_THREADS //", True, COL_CYAN)
        self.screen.blit(title, (start_x, start_y - 20))

        for i, (name, pid) in enumerate(self.active_processes):
            y_pos = start_y + (i * 20)
            
            # Draw Text
            proc_txt = self.font_small.render(f"[{pid}] {name}", True, COL_TEXT)
            self.screen.blit(proc_txt, (start_x, y_pos))
            
            # Draw Connection Line to Chip
            # Only draw line if it's "active" (visual flair)
            if i < 5: 
                pygame.draw.line(self.screen, (0, 100, 100), (start_x - 5, y_pos + 5), (chip_x, chip_y), 1)

    def draw(self):
        self.chip.update()
        self.screen.fill(COL_DARK)

        # 1. 3D GRID
        for x in range(-GRID_SIZE, GRID_SIZE + 1, 2): 
            pts = []
            for z in range(10):
                px, py = project(x*GRID_SPACING, 2.0, z*2.0, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
                pts.append((px, py))
            if len(pts)>1: pygame.draw.lines(self.screen, COL_GRID, False, pts, 1)

        # 2. PACKETS
        for pkt in self.packets:
            brightness = max(0.2, 1.0 - (pkt.z + 5.0) / 15.0)
            color = (int(pkt.color[0]*brightness), int(pkt.color[1]*brightness), int(pkt.color[2]*brightness))
            px, py = project(pkt.x, pkt.y, pkt.z, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
            tx, ty = project(pkt.x, pkt.y, pkt.z + 2.0, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
            pygame.draw.line(self.screen, color, (px, py), (tx, ty), 2)

        # 3. CHIP & HUD
        cx, cy = project(0, 0, 5, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
        self.chip.draw(self.screen, cx, cy)
        
        # Draw Process List
        self.draw_active_processes(cx, cy)

        # 4. INTRO SEQUENCE (Fade out)
        elapsed = time.time() - self.start_time
        if elapsed < 5.0:
            alpha = int(255 * (1.0 - (elapsed / 5.0)))
            if alpha > 0:
                msg = self.font_large.render("CLOSE THE WORLD", True, (255, 255, 255))
                msg2 = self.font_large.render("OPEN THE NEXT", True, (255, 255, 255))
                
                s = pygame.Surface((WIN_WIDTH, WIN_HEIGHT), pygame.SRCALPHA)
                s.blit(msg, (WIN_WIDTH//2 - msg.get_width()//2, WIN_HEIGHT//2 - 40))
                s.blit(msg2, (WIN_WIDTH//2 - msg2.get_width()//2, WIN_HEIGHT//2 + 10))
                s.set_alpha(alpha)
                self.screen.blit(s, (0,0))

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
