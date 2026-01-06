import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
import cv2
import numpy as np
import time
import math
import random
import queue
import psutil
import threading
import collections
import wifi_scanner
import protocol
import entities 
import sounds

# --- Constants ---
WIN_WIDTH = 1280
WIN_HEIGHT = 720
TUNNEL_DEPTH = 40.0 

# Colors
COL_CYAN = (0.0, 1.0, 1.0, 1.0)
COL_RED = (1.0, 0.2, 0.2, 1.0)
COL_GRID = (0.0, 0.15, 0.3, 0.4) 
COL_DARK = (0.02, 0.02, 0.04, 1.0)
COL_TEXT = (0.8, 0.8, 0.8, 1.0)
COL_GHOST = (0.2, 1.0, 0.2, 0.8) 
COL_WHITE = (1.0, 1.0, 1.0, 1.0)
COL_YELLOW = (1.0, 1.0, 0.0, 1.0)
COL_HEX = (0.0, 0.6, 0.0, 1.0) 





# --- GLSL Shaders ---
VS_BASE = """
#version 120
void main() {
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = ftransform();
}
"""

FS_COMPOSITE = """
#version 120
uniform sampler2D tex;
uniform float time;
uniform float glitch_intensity;
uniform vec3 tint_color;
uniform bool invert;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
    vec2 uv = gl_TexCoord[0].st;
    
    // Glitch Offset (Horizontal Tearing)
    if (rand(vec2(time, uv.y)) < glitch_intensity * 0.05) {
        uv.x += rand(vec2(time, uv.y)) * 0.05 - 0.025;
    }

    vec3 col = texture2D(tex, uv).rgb;
    
    // 1. Tint
    col *= tint_color;
    
    // 2. Scanline (Subtle)
    float scanline = sin(uv.y * 800.0) * 0.04;
    col -= scanline;
    
    // 3. Breach Inversion (Narrative Critical)
    if (invert) {
        col = 1.0 - col;
    }

    gl_FragColor = vec4(col, 1.0);
}
"""

class SimpleFramebuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fbo = glGenFramebuffers(1)
        self.tex = glGenTextures(1)
        self.rbo = glGenRenderbuffers(1)
        
        # Texture
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Depth Buffer (Needed for 3D)
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        
        # Attach
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.rbo)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

class PostProcess:
    def __init__(self, width, height):
        self.fbo = SimpleFramebuffer(width, height)
        self.width = width
        self.height = height
        try:
            self.program = compileProgram(
                compileShader(VS_BASE, GL_VERTEX_SHADER),
                compileShader(FS_COMPOSITE, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"Shader Fail: {e}")
            self.program = 0

    def begin(self):
        self.fbo.bind()

    def end(self, time_val, glitch, tint, invert):
        # Unbind to draw to screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if not self.program:
            # Fallback if shader failed
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.fbo.tex)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(-1, -1)
            glTexCoord2f(1, 0); glVertex2f(1, -1)
            glTexCoord2f(1, 1); glVertex2f(1, 1) 
            glTexCoord2f(0, 1); glVertex2f(-1, 1) 
            glEnd()
            return

        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "tex"), 0)
        glUniform1f(glGetUniformLocation(self.program, "time"), time_val)
        glUniform1f(glGetUniformLocation(self.program, "glitch_intensity"), glitch)
        glUniform3f(glGetUniformLocation(self.program, "tint_color"), tint[0], tint[1], tint[2])
        glUniform1i(glGetUniformLocation(self.program, "invert"), 1 if invert else 0)
        
        glDisable(GL_DEPTH_TEST)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.fbo.tex)
        
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1) 
        glTexCoord2f(0, 1); glVertex2f(-1, 1) 
        glEnd()
        
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)

class TextTexture:
    def __init__(self, font):
        self.font = font
        self.texture_id = glGenTextures(1)
        self.width = 0; self.height = 0
        self._current_text = None; self._current_color = None
        self.last_used = time.time()

    def update(self, text, color=(1.0, 1.0, 1.0, 1.0)): 
        if text == self._current_text and color == self._current_color: return 
        self._current_text = text; self._current_color = color

        # Handle both float (0-1) and int (0-255) inputs
        is_float = False
        for c in color[:3]:
            if isinstance(c, float) and c <= 1.0: 
                is_float = True # Assume float if any component is float <= 1.0? 
                # Risk: (0,0,0) int is <= 1.0. 
                # Better: check if ANY component > 1.0
        
        max_val = max(color[:3])
        if max_val > 1.0:
            # Assume 0-255
            c255 = list(color[:3])
            if len(color) == 4: c255.append(color[3])
            else: c255.append(255)
            # Ensure int
            c255 = [int(c) for c in c255]
        else:
            # Assume 0-1
            c255 = [int(c * 255) for c in color[:3]]
            if len(color) == 4: c255.append(int(color[3] * 255))
            else: c255.append(255)

        surf = self.font.render(text, True, c255)
        new_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        new_surf.fill((0, 0, 0, 0)) 
        new_surf.blit(surf, (0, 0)) 
        self.width = new_surf.get_width(); self.height = new_surf.get_height()

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        data = pygame.image.tostring(new_surf, "RGBA", True) 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

    def draw(self, x, y, z, scale=0.02, yaw=0):
        self.last_used = time.time()
        if self.width == 0: return 
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(yaw, 0, 1, 0)
        glScalef(scale, scale, 1)
        glTranslatef(-self.width / 2, -self.height / 2, 0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(0, 0, 0)
        glTexCoord2f(1, 0); glVertex3f(self.width, 0, 0)
        glTexCoord2f(1, 1); glVertex3f(self.width, self.height, 0)
        glTexCoord2f(0, 1); glVertex3f(0, self.height, 0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()

class WebcamTexture:
    def __init__(self):
        self.texture_id = glGenTextures(1)
        
    def update(self, frame):
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        rgba = np.zeros((240, 320, 4), dtype=np.uint8) 
        rgba[edges > 0] = [int(COL_GHOST[0]*255), int(COL_GHOST[1]*255), int(COL_GHOST[2]*255), int(COL_GHOST[3]*255)] 
        rgba = cv2.flip(rgba, 0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 320, 240, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba)

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

class SystemMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cpu = 0; self.ram = 0; self.disk_write = False
        self.processes = []
        self.running = True; self.daemon = True

    def get_port_color(self, pid):
        try:
            p = psutil.Process(pid)
            conns = p.net_connections(kind='inet')
            if not conns: return COL_CYAN 
            for c in conns:
                if c.status == 'ESTABLISHED' and c.raddr:
                    port = c.raddr.port
                    if port == 80 or port == 8080: return (0.0, 0.0, 1.0, 1.0) 
                    if port == 443: return (0.0, 1.0, 0.0, 1.0) 
                    if port in [21, 22]: return (1.0, 1.0, 0.0, 1.0) 
            return COL_CYAN 
        except: return COL_CYAN 

    def run(self):
        last_disk = 0
        while self.running:
            self.cpu = psutil.cpu_percent(interval=None)
            self.ram = psutil.virtual_memory().percent
            try:
                disk = psutil.disk_io_counters()
                curr = disk.write_bytes
                self.disk_write = (curr > last_disk)
                last_disk = curr
            except: pass
            
            try:
                conns = [p for p in psutil.net_connections() if p.status == 'ESTABLISHED']
                unique = {}
                for c in conns:
                    if c.pid and c.pid not in unique:
                        try:
                            p = psutil.Process(c.pid)
                            port = None
                            if c.raddr and c.raddr.port: port = c.raddr.port
                            unique[c.pid] = (p.name(), self.get_port_color(c.pid), port) 
                        except: pass
                
                # Atomic update: Build list first, then assign
                new_procs = []
                for pid, (name, color, port) in list(unique.items()):
                    new_procs.append((pid, name, color, port))
                
                # Sort by PID to ensure stable order (prevents satellites jumping/exploding)
                new_procs.sort(key=lambda x: x[0])
                
                # Limit to 12
                self.processes = new_procs[:12]
                
            except: pass
            time.sleep(1.0)

class WiredEngine:
    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Navi v22.0 - [BREACH PROTOCOL]")
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        # OpenGL Config
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_BLEND) 
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.5) 
        glEnable(GL_COLOR_MATERIAL) 
        glShadeModel(GL_FLAT) 
        
        # Camera
        glMatrixMode(GL_PROJECTION)
        gluPerspective(90, (WIN_WIDTH/WIN_HEIGHT), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        self.drone_sound = sounds.generate_drone(55, 10.0) # 10s loop
        self.screech_sound = sounds.generate_screech()
        self.explode_sound = sounds.generate_explosion()
        self.drone_channel = None
        if self.drone_sound:
            self.drone_channel = self.drone_sound.play(loops=-1, fade_ms=2000)
            self.drone_channel.set_volume(0.3)

        # Logic
        self.cap = cv2.VideoCapture(0)
        self.monitor = SystemMonitor(); self.monitor.start()
        self.packet_queue = queue.Queue()
        self.listener = protocol.ProtocolListener(self.packet_queue); self.listener.start()
        self.data_buffer = collections.deque(maxlen=2000) 
        for _ in range(200): self.data_buffer.append(random.randint(0, 255))
        self.wifi_scanner = wifi_scanner.WifiScanner(); self.wifi_scanner.start()

        # Resources
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.cam_tex = WebcamTexture()
        self.labels = {} 
        self.post_process = PostProcess(WIN_WIDTH, WIN_HEIGHT)
        
        # State
        self.cam_pos = [0.0, 0.0, 5.0] 
        self.cam_yaw = 0.0; self.cam_pitch = 0.0
        self.world_offset_z = 0.0 
        self.rotation = 0.0
        self.last_texture_cleanup = time.time()
        self.packets = [] 
        self.active_procs_objs = [] # {'pid', 'pos': (x,y,z), 'radius'}
        self.start_time = time.time()
        self.glitch_level = 0.0 
        self.running = True
        self.clock = pygame.time.Clock()
        
        # Breach Narrative
        self.breach_mode = False
        self.blackwall_timer = 0
        self.in_blackwall_zone = False

        # --- Entity Initialization ---
        self.entities = []
        self.entities.append(entities.InfiniteTunnel(self))
        self.entities.append(entities.CyberArch(self))
        self.entities.append(entities.GhostWall(self))
        self.entities.append(entities.Hypercore(self))
        self.entities.append(entities.SatelliteSystem(self))
        self.entities.append(entities.PacketSystem(self))
        self.entities.append(entities.CyberCity(self))
        self.entities.append(entities.Blackwall(self))
        self.entities.append(entities.AlienSwarm(self))
        self.entities.append(entities.StatsWall(self))
        self.entities.append(entities.WifiVisualizer(self))
        self.entities.append(entities.DigitalRain(self, side='left', color=entities.COL_HEX))
        self.entities.append(entities.DigitalRain(self, side='right', color=entities.COL_RED))
        self.entities.append(entities.IntroOverlay(self))
        
        self.particle_system = entities.ParticleSystem(self)
        self.entities.append(self.particle_system)

        # Pre-cache
        print("Pre-caching textures...")
        for i in range(256):
            h = f"{i:02X}"
            self.get_label(h, entities.COL_HEX)
        print("Ready.")

    def get_label(self, text, color):
        key = (text, color)
        if key not in self.labels:
            tex = TextTexture(self.font)
            tex.update(text, color)
            self.labels[key] = tex
        return self.labels[key]
        
    def get_ray_from_mouse(self, mx, my):
        # Viewport
        viewport = glGetIntegerv(GL_VIEWPORT)
        
        # Modelview
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        
        # Projection
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # WinY is inverted in Pygame vs OpenGL
        winY = float(viewport[3]) - float(my)
        winX = float(mx)
        
        # Unproject Near
        try:
            start = gluUnProject(winX, winY, 0.0, modelview, projection, viewport)
            end = gluUnProject(winX, winY, 1.0, modelview, projection, viewport)
        except: return (None, None)
        
        # Direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        direction = (dx/length, dy/length, dz/length)
        
        return start, direction

    def ray_sphere_intersect(self, r_origin, r_dir, s_center, s_radius):
        # Geometric solution
        # L = center - origin
        lx = s_center[0] - r_origin[0]
        ly = s_center[1] - r_origin[1]
        lz = s_center[2] - r_origin[2]
        
        # tca = L . D
        tca = lx*r_dir[0] + ly*r_dir[1] + lz*r_dir[2]
        
        if tca < 0: return False # Behind origin
        
        # d^2 = L.L - tca*tca
        d2 = (lx*lx + ly*ly + lz*lz) - tca*tca
        
        if d2 > s_radius * s_radius: return False
        
        return True # Hit

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == QUIT: self.running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE: self.running = False
            
            # Raycasting Interaction
            if event.type == MOUSEBUTTONDOWN and event.button == 1: # Left Click
                start, direction = self.get_ray_from_mouse(event.pos[0], event.pos[1])
                if start:
                    # Check intersection with SatelliteSystem processes
                    # We need the positions stored in SatelliteSystem (which are relative to world)
                    # SatelliteSystem updates 'active_procs_objs' in engine
                    
                    hit = False
                    for obj in self.active_procs_objs:
                        # obj['pos'] is set in SatelliteSystem.draw() which runs in render loop
                        # This might be one frame lag, but acceptable
                        if 'pos' not in obj: continue
                        
                        center = obj['pos']
                        # Apply Ray-Sphere intersection
                        if self.ray_sphere_intersect(start, direction, center, 2.0): # Approx radius 2
                            print(f"TERMINATING PROCESS: {obj['name']} ({obj['pid']})")
                            try:
                                p = psutil.Process(obj['pid'])
                                p.terminate()
                                self.particle_system.explode(center[0], center[1], center[2], color=(1,0,0,1))
                                if self.explode_sound: self.explode_sound.play()
                                hit = True
                            except: pass
                            break # One click, one kill

        # Continuous Input
        keys = pygame.key.get_pressed()
        speed = 2.0 
        
        mdx, mdy = pygame.mouse.get_rel()
        self.cam_yaw += mdx * 0.1
        self.cam_pitch -= mdy * 0.1 
        self.cam_pitch = max(-89, min(89, self.cam_pitch))
        
        rad_yaw = math.radians(self.cam_yaw)
        fx = math.sin(rad_yaw); fz = -math.cos(rad_yaw)
        rx = math.cos(rad_yaw); rz = math.sin(rad_yaw)
        
        dx = 0; dz = 0
        if keys[K_w]: dx += fx * speed; dz += fz * speed
        if keys[K_s]: dx -= fx * speed; dz -= fz * speed
        if keys[K_a]: dx -= rx * speed; dz -= rz * speed
        if keys[K_d]: dx += rx * speed; dz += rz * speed
            
        # Blackwall Collision
        next_z = self.cam_pos[2] + dz
        global_z = next_z + self.world_offset_z
        wall_z = -4500.0
        
        if not self.blackwall_state['breached'] and global_z < wall_z + 20:
             if dz < 0: dz = 0 
        
        self.cam_pos[0] += dx
        self.cam_pos[2] += dz
        self.cam_pos[0] = max(-3.5, min(3.5, self.cam_pos[0]))
        self.cam_pos[1] = max(-2.5, min(2.5, self.cam_pos[1]))
        self.cam_pos[2] = min(10.0, self.cam_pos[2])

    def update(self):
        ok, frame = self.cap.read()
        if ok: self.cam_tex.update(frame)

        for entity in self.entities: entity.update()
        self.rotation += 0.5
        
        global_z = self.cam_pos[2] + self.world_offset_z
        wall_z = -4500.0
        
        # Blackwall Logic
        if not self.blackwall_state['breached'] and global_z < -3500:
             dist = global_z - wall_z 
             if dist < 300 and self.blackwall_state['warnings'] == 0:
                 self.blackwall_state['warnings'] = 1; self.blackwall_state['message'] = "WARNING: CLASSIFIED DATA"
                 self.blackwall_state['last_warning_time'] = time.time(); self.cam_pos[2] += 20 
             elif dist < 150 and self.blackwall_state['warnings'] == 1:
                  if time.time() - self.blackwall_state['last_warning_time'] > 1.0:
                      self.blackwall_state['warnings'] = 2; self.blackwall_state['message'] = "DANGER: LETHAL COUNTERMEASURES"
                      self.blackwall_state['last_warning_time'] = time.time()
             elif dist < 50 and self.blackwall_state['warnings'] == 2:
                  if time.time() - self.blackwall_state['last_warning_time'] > 1.0:
                      self.blackwall_state['warnings'] = 3; self.blackwall_state['message'] = "CRITICAL: BREACH IMMINENT"
                      self.blackwall_state['last_warning_time'] = time.time()
             if self.blackwall_state['warnings'] == 3 and time.time() - self.blackwall_state['last_warning_time'] > 2.0:
                   self.blackwall_state['breached'] = True; self.blackwall_state['message'] = "SYSTEM FAILURE // BREACH DETECTED"

        # Zones & Breach Narrative
        self.in_blackwall_zone = False
        if global_z > -1000:
            self.zone_state = {'name': 'SURFACE', 'grid_color': COL_GRID, 'tint': (0.8, 1.1, 1.0), 'distortion': 0.0}
        elif global_z > -3000:
            self.zone_state = {'name': 'SPRAWL', 'grid_color': (0.6, 0.0, 0.8, 0.5), 'tint': (0.9, 0.8, 1.1), 'distortion': 0.1}
        elif global_z > -4500:
            self.zone_state = {'name': 'DEEP_WEB', 'grid_color': (0.0, 0.8, 0.2, 0.5), 'tint': (0.7, 1.0, 0.7), 'distortion': 1.5}
        elif global_z > -5500:
            self.zone_state = {'name': 'BLACKWALL', 'grid_color': (0.8, 0.0, 0.0, 0.5), 'tint': (1.2, 0.8, 0.8), 'distortion': 3.0}
            self.in_blackwall_zone = True
        else:
            self.zone_state = {'name': 'OLD_NET', 'grid_color': (0.8, 0.8, 0.9, 0.6), 'tint': (0.6, 0.6, 0.7), 'distortion': 4.0}

        # Breach Timer
        if self.in_blackwall_zone and not self.breach_mode:
            self.blackwall_timer += 1.0 / 60.0
            if self.blackwall_timer > 10.0:
                self.breach_mode = True
                self.blackwall_state['message'] = "SYSTEM COMPROMISED - HUNTERS DEPLOYED"
                if self.screech_sound: self.screech_sound.play()
                # Spawn Hunters
                for _ in range(5):
                    spawn_pos = (self.cam_pos[0] + random.uniform(-10, 10), 
                                 self.cam_pos[1] + random.uniform(-5, 5), 
                                 self.cam_pos[2] - 50) # In front of player
                    self.entities.append(entities.Hunter(self, spawn_pos))
        
        # Audio Modulation
        if self.drone_channel:
            # Volume based on packet count
            target_vol = 0.3 + min(0.7, len(self.packets) * 0.05)
            self.drone_channel.set_volume(target_vol)

        # Floating Origin
        if self.cam_pos[2] < -100.0:
            shift = self.cam_pos[2]; self.cam_pos[2] -= shift; self.world_offset_z += shift
        
        # GC
        if time.time() - self.last_texture_cleanup > 10.0:
            self.last_texture_cleanup = time.time()
            threshold = time.time() - 60.0
            keys_to_delete = [k for k, t in self.labels.items() if t.last_used < threshold]
            for k in keys_to_delete:
                try: glDeleteTextures([self.labels[k].texture_id])
                except: pass
                del self.labels[k]

    def draw(self):
        # Begin Post-Process
        self.post_process.begin()
        
        # Standard Rendering (to FBO)
        glLoadIdentity()
        rad_yaw = math.radians(self.cam_yaw)
        rad_pitch = math.radians(self.cam_pitch)
        lx = math.sin(rad_yaw) * math.cos(rad_pitch)
        ly = math.sin(rad_pitch)
        lz = -math.cos(rad_yaw) * math.cos(rad_pitch)
        gluLookAt(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
                  self.cam_pos[0] + lx, self.cam_pos[1] + ly, self.cam_pos[2] + lz,
                  0, 1, 0)
        
        for entity in self.entities: entity.draw()
        
        # End Post-Process -> Draw to Screen with Effects
        t = time.time() - self.start_time
        tint = self.zone_state.get('tint', (1,1,1))
        glitch = self.glitch_level + (2.0 if self.breach_mode else 0.0)
        self.post_process.end(t, glitch, tint, self.breach_mode)
        
        # UI Overlay (Drawn on top of effects)
        msg = self.blackwall_state.get('message')
        if msg:
             glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
             gluOrtho2D(0, self.screen.get_width(), self.screen.get_height(), 0)
             glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
             glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND)
             lbl = self.get_label(msg, (255, 50, 50))
             glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, lbl.texture_id)
             x = (self.screen.get_width() - lbl.width) / 2
             y = (self.screen.get_height() / 2) - 100
             alpha = 0.5 + math.sin(time.time()*15)*0.5
             glColor4f(1, 1, 1, alpha)
             glBegin(GL_QUADS)
             glTexCoord2f(0, 0); glVertex2f(x, y); glTexCoord2f(1, 0); glVertex2f(x + lbl.width, y)
             glTexCoord2f(1, 1); glVertex2f(x + lbl.width, y + lbl.height); glTexCoord2f(0, 1); glVertex2f(x, y + lbl.height)
             glEnd()
             glDisable(GL_TEXTURE_2D); glEnable(GL_DEPTH_TEST)
             glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
        
        pygame.display.flip()

    def loop(self):
        try:
            print("Entering main loop...")
            self.blackwall_state = {'warnings': 0, 'breached': False, 'last_warning_time': 0, 'message': None}
            frame_count = 0
            while self.running:
                self.handle_input()
                self.update()
                self.draw()
                self.clock.tick(60)
                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"FPS: {self.clock.get_fps():.2f}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            print("Cleaning up...")
            self.listener.stop(); self.monitor.running = False; self.wifi_scanner.stop()
            self.cap.release(); pygame.quit()

if __name__ == "__main__":
    WiredEngine().loop()