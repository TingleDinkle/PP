import pygame
import cv2
import mediapipe as mp
import queue
import protocol  # Import our new network layer
import sounds    # Import sound generator
import time
import numpy as np

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
        self.screen.fill((0, 0, 0))

        # Draw Grid
        for px, py, pz in self.grid_points:
            scene_z = pz + 2.0 
            screen_x, screen_y = project(px, py, scene_z, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
            
            color = (0, 255, 0)
            if pz == 0: color = (255, 0, 0)
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), 3)

        # Draw Packets
        for pkt in self.packets:
            # Calculate brightness based on Z
            brightness = 1.0 - (pkt.z + 5.0) / 15.0
            brightness = max(0.2, min(1.0, brightness))
            
            # Apply brightness to color
            r = int(pkt.color[0] * brightness)
            g = int(pkt.color[1] * brightness)
            b = int(pkt.color[2] * brightness)
            final_color = (r, g, b)
            
            # Render Text
            text_surf = self.font.render(pkt.dst, True, final_color)
            
            screen_x, screen_y = project(pkt.x, pkt.y, pkt.z, self.hx, self.hy, WIN_WIDTH, WIN_HEIGHT)
            
            # Center the text on the projected point
            rect = text_surf.get_rect(center=(screen_x, screen_y))
            self.screen.blit(text_surf, rect)

        # "Connection Lost" / "Open The Next" logic
        elapsed_since_packet = time.time() - self.last_packet_time
        if elapsed_since_packet > 5.0:
            # Fade in over 3 seconds
            fade = min(1.0, (elapsed_since_packet - 5.0) / 3.0)
            alpha = int(fade * 255)
            
            text_msg = "CLOSE THE WORLD, OPEN THE NEXT"
            msg_surf = self.title_font.render(text_msg, True, (255, 255, 255))
            msg_surf.set_alpha(alpha)
            
            center_rect = msg_surf.get_rect(center=(WIN_WIDTH//2, WIN_HEIGHT//2))
            self.screen.blit(msg_surf, center_rect)

        # Frame
        pygame.draw.rect(self.screen, (50, 50, 50), (0, 0, WIN_WIDTH, WIN_HEIGHT), 5)

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
