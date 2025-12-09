import pygame
import cv2
import mediapipe as mp
import queue
import protocol  # Import our new network layer
import sounds    # Import sound generator
import time
import numpy as np
import math

# --- Constants ---
SCALE = 120
EYE_DIST = 8.0
WIN_WIDTH = 800
WIN_HEIGHT = 600
GRID_SIZE = 4
GRID_SPACING = 2.0
PACKET_SPEED = 0.05

# --- Head Tracking Logic ---
def get_head(cap, face_mesh):
    """
    Captures video frame, processes it with MediaPipe Face Mesh,
    and returns the relative X, Y coordinates of the head center (nose bridge).
    """
    ok, frame = cap.read()
    if not ok:
        return 0, 0, False
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    res = face_mesh.process(rgb)
    
    if not res.multi_face_landmarks:
        return 0, 0, False
    
    # Landmark 168 is often used for the nose bridge / center of face
    pt = res.multi_face_landmarks[0].landmark[168]
    
    # Normalize and center the coordinates
    return (pt.x - 0.5) * 2, (pt.y - 0.5) * 2, True

# --- Projection Logic ---
def project(x, y, z, hx, hy, WIDTH, HEIGHT):
    """
    Projects 3D coordinates (x, y, z) to 2D screen coordinates based on
    head position (hx, hy).
    """
    d = EYE_DIST + z
    if d == 0: d = 0.001 # Prevent division by zero
    f = EYE_DIST / d
    
    sx = hx + (x - hx) * f
    sy = hy + (y - hy) * f
    
    px = int(WIDTH/2 + sx * SCALE)
    py = int(HEIGHT/2 + sy * SCALE)
    return px, py

class PsycheChip:
    def __init__(self):
        self.angle = 0.0
        # Octahedron vertices (Diamond shape)
        self.vertices = [
            (0, -1, 0), (0, 1, 0),  # Top/Bottom
            (-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1) # Middle ring
        ]
        # Edges connecting vertices
        self.edges = [
            (0,2), (0,3), (0,4), (0,5), # Top to ring
            (1,2), (1,3), (1,4), (1,5), # Bottom to ring
            (2,3), (3,4), (4,5), (5,2)  # Ring loop
        ]

    def update(self):
        self.angle += 0.02

    def draw(self, surface, center_x, center_y, scale=40):
        # Rotation matrix (Y-axis)
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        projected_points = []
        for x, y, z in self.vertices:
            # Rotate
            rx = x * cos_a - z * sin_a
            rz = x * sin_a + z * cos_a
            # Project (simple perspective for the chip itself)
            # We add a slight Z-offset so it doesn't clip
            dist = 4.0 + rz 
            if dist == 0: dist = 0.1
            f = 300 / dist
            px = center_x + int(rx * scale * f * 0.01)
            py = center_y + int(y * scale * f * 0.01)
            projected_points.append((px, py))

        # Draw Wireframe
        for start_idx, end_idx in self.edges:
            p1 = projected_points[start_idx]
            p2 = projected_points[end_idx]
            pygame.draw.line(surface, (255, 0, 0), p1, p2, 2) # Red Chip

class WiredWindow:
    def __init__(self):
        # --- Initialization ---
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Window into the Wired - Protocol 7")
        self.clock = pygame.time.Clock()
        self.running = True

        # --- Audio Setup ---
        # Generate and play background drone
        drone_data = sounds.generate_drone(duration=2.0)
        self.drone_sound = pygame.sndarray.make_sound(drone_data)
        self.drone_sound.play(loops=-1)
        self.drone_sound.set_volume(0.3)
        
        # Generate screech sound for bursts
        screech_data = sounds.generate_screech()
        self.screech_sound = pygame.sndarray.make_sound(screech_data)
        self.screech_sound.set_volume(0.4)
        self.burst_timer = 0

        # --- Font Setup ---
        # Monospace font for that "hacker" aesthetic
        self.font = pygame.font.SysFont("Consolas", 14, bold=True)
        if not self.font:
             self.font = pygame.font.SysFont("Courier New", 14, bold=True)
        
        # Large font for the "Connection Lost" message
        self.title_font = pygame.font.SysFont("Arial", 40, bold=True)

        # --- MediaPipe Setup ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.running = False

        # --- Scene Data ---
        # Grid
        self.grid_points = []
        for x in range(-GRID_SIZE, GRID_SIZE + 1):
            for y in range(-GRID_SIZE, GRID_SIZE + 1):
                for z in range(0, 3): 
                    self.grid_points.append((x * GRID_SPACING, y * GRID_SPACING, z * GRID_SPACING))
        
        # Packets
        self.packet_queue = queue.Queue(maxsize=100) # Buffer
        self.packets = [] # Active packet objects being rendered
        self.last_packet_time = time.time()
        
        # Network Listener
        self.listener = protocol.ProtocolListener(self.packet_queue)
        self.listener.start()

        # Head Position
        self.hx, self.hy = 0, 0
        
        # Psyche Chip
        self.chip = PsycheChip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def update(self):
        # 1. Update Head Tracking
        new_hx, new_hy, found = get_head(self.cap, self.face_mesh)
        if found:
            self.hx, self.hy = new_hx, new_hy

        # 2. Fetch new packets from thread
        packets_this_frame = 0
        try:
            while True:
                pkt = self.packet_queue.get_nowait()
                self.packets.append(pkt)
                packets_this_frame += 1
                self.last_packet_time = time.time()
        except queue.Empty:
            pass
        
        # Audio: Burst Detection
        if packets_this_frame > 3 and time.time() > self.burst_timer:
            self.screech_sound.play()
            self.burst_timer = time.time() + 2.0 # Cooldown

        # 3. Update active packets
        # Filter list to keep only alive packets
        self.packets = [p for p in self.packets if p.update()]

    def draw(self):
        self.chip.update() # Spin the chip
        self.screen.fill((0, 0, 0)) # Clear to black

        # --- 1. Draw The Wired (Grid Lines instead of dots) ---
        # We assume grid_points is a list of (x,y,z). 
        # To draw lines, we need to know which points are neighbors.
        # This is a simplified "Speed Tunnel" approach:
        
        # Floor and Ceiling Lines
        for x in range(-GRID_SIZE, GRID_SIZE + 1, 2): 
            # Create a line of points stretching into Z depth
            line_points = []
            for z_step in range(0, 10): # Depth steps
                # Floor line
                wx, wy, wz = x * GRID_SPACING, 2.0, z_step * 2.0
                px, py = project(wx, wy, wz, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
                line_points.append((px, py))
            
            # Draw the connected line (Floor)
            if len(line_points) > 1:
                pygame.draw.lines(self.screen, (0, 100, 255), False, line_points, 1)

            # Ceiling line (flipped Y)
            line_points_ceil = []
            for z_step in range(0, 10):
                wx, wy, wz = x * GRID_SPACING, -2.0, z_step * 2.0
                px, py = project(wx, wy, wz, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
                line_points_ceil.append((px, py))
            
            if len(line_points_ceil) > 1:
                pygame.draw.lines(self.screen, (0, 100, 255), False, line_points_ceil, 1)

        # --- 2. Draw The Chip (Floating in Center) ---
        # We verify where the "center" is based on head tracking
        center_x, center_y = project(0, 0, 5, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
        self.chip.draw(self.screen, center_x, center_y)

        # --- 3. Draw Packets (Data Streams) ---
        for pkt in self.packets:
            # Calculate brightness/color
            brightness = 1.0 - (pkt.z + 5.0) / 15.0
            brightness = max(0.2, min(1.0, brightness))
            color = (
                int(pkt.color[0] * brightness),
                int(pkt.color[1] * brightness),
                int(pkt.color[2] * brightness)
            )

            # Project current position
            px, py = project(pkt.x, pkt.y, pkt.z, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
            
            # Draw a "Beam" trail behind the packet
            tail_z = pkt.z + 1.0 # The tail is further back
            tx, ty = project(pkt.x, pkt.y, tail_z, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
            pygame.draw.line(self.screen, color, (px, py), (tx, ty), 2)

            # Draw Text
            if brightness > 0.5: # Only draw text if close enough
                text_surf = self.font.render(pkt.dst, True, color)
                self.screen.blit(text_surf, (px, py))

        # --- 4. Draw Navi UI Overlay (Static) ---
        # These do NOT move with head tracking
        pygame.draw.rect(self.screen, (255, 255, 255), (20, 20, 200, 40), 1) # Box
        label = self.font.render("NAVI v7.0 // PROTOCOL: ON", True, (255, 255, 255))
        self.screen.blit(label, (25, 30))

        # Crosshair in center of screen
        cx, cy = WIN_WIDTH // 2, WIN_HEIGHT // 2
        pygame.draw.line(self.screen, (50, 255, 50), (cx-10, cy), (cx+10, cy), 1)
        pygame.draw.line(self.screen, (50, 255, 50), (cx, cy-10), (cx, cy+10), 1)

        # --- 5. Alerts ---
        elapsed = time.time() - self.last_packet_time
        if elapsed > 5.0:
            fade = min(1.0, (elapsed - 5.0) / 3.0)
            alpha = int(fade * 255)
            msg = self.title_font.render("CLOSE THE WORLD", True, (255, 255, 255))
            msg.set_alpha(alpha)
            self.screen.blit(msg, (WIN_WIDTH//2 - 150, WIN_HEIGHT//2 - 20))

        pygame.display.flip()

    def run_loop(self):
        while self.running:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)

        # Cleanup
        self.listener.stop()
        self.cap.release()
        pygame.quit()
        self.face_mesh.close()

if __name__ == "__main__":
    app = WiredWindow()
    app.run_loop()
