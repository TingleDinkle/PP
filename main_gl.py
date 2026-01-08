import math
import queue
import random
import threading
import time
import collections
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import psutil
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader

# Local Imports
import config
import entities
import protocol
import sounds
import wifi_scanner

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
    """Helper class for OpenGL Framebuffer Object (FBO) management."""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fbo = glGenFramebuffers(1)
        self.tex = glGenTextures(1)
        self.rbo = glGenRenderbuffers(1)
        
        # Texture Attachment
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Depth Buffer Attachment
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        
        # Link
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
    """Manages the full-screen post-processing shader pipeline."""
    def __init__(self, width: int, height: int):
        self.fbo = SimpleFramebuffer(width, height)
        self.width = width
        self.height = height
        try:
            self.program = compileProgram(
                compileShader(VS_BASE, GL_VERTEX_SHADER),
                compileShader(FS_COMPOSITE, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"Shader Compilation Failed: {e}")
            self.program = 0

    def begin(self):
        self.fbo.bind()

    def end(self, time_val: float, glitch: float, tint: Tuple[float, float, float], invert: bool):
        # Draw FBO texture to screen with shader
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if not self.program:
            # Fallback: Fixed Function
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.fbo.tex)
            self._draw_quad()
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
        
        self._draw_quad()
        
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)

    def _draw_quad(self):
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1) 
        glTexCoord2f(0, 1); glVertex2f(-1, 1) 
        glEnd()

class TextTexture:
    """Manages dynamic text rendering to OpenGL Textures."""
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.texture_id = glGenTextures(1)
        self.width = 0
        self.height = 0
        self._current_text = None
        self._current_color = None
        self.last_used = time.time()

    def update(self, text: str, color: config.Color = config.COL_WHITE): 
        if text == self._current_text and color == self._current_color:
            return 
        
        self._current_text = text
        self._current_color = color

        # Robust Float (0.0-1.0) to Int (0-255) conversion
        # Pygame requires 0-255 integers
        c255 = []
        for c in color[:3]:
            # Clamp and scale
            val = int(max(0.0, min(1.0, c)) * 255)
            c255.append(val)
        
        # Handle Alpha if present
        if len(color) == 4:
            c255.append(int(max(0.0, min(1.0, color[3])) * 255))
        else:
            c255.append(255)

        surf = self.font.render(text, True, c255)
        
        # Create RGBA Surface
        new_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        new_surf.fill((0, 0, 0, 0)) 
        new_surf.blit(surf, (0, 0)) 
        
        self.width = new_surf.get_width()
        self.height = new_surf.get_height()

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        data = pygame.image.tostring(new_surf, "RGBA", True) 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

    def draw(self, x: float, y: float, z: float, scale: float = 0.02, yaw: float = 0):
        self.last_used = time.time()
        if self.width == 0: return 
        
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(yaw, 0, 1, 0)
        glScalef(scale, scale, 1)
        glTranslatef(-self.width / 2, -self.height / 2, 0)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor4f(1.0, 1.0, 1.0, 1.0) # Tint white to keep original font color
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(0, 0, 0)
        glTexCoord2f(1, 0); glVertex3f(self.width, 0, 0)
        glTexCoord2f(1, 1); glVertex3f(self.width, self.height, 0)
        glTexCoord2f(0, 1); glVertex3f(0, self.height, 0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()

class WebcamTexture:
    """Manages webcam capture and Canny Edge Detection visualization."""
    def __init__(self):
        self.texture_id = glGenTextures(1)
        
    def update(self, frame):
        if frame is None: return
        # Downscale for performance & style
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create Ghostly Green Overlay
        rgba = np.zeros((240, 320, 4), dtype=np.uint8) 
        
        # Apply color to edges
        r, g, b, a = [int(c * 255) for c in config.COL_GHOST]
        rgba[edges > 0] = [r, g, b, a]
        
        rgba = cv2.flip(rgba, 0)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 320, 240, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba)

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

class SystemMonitor(threading.Thread):
    """Background thread for psutil metrics."""
    def __init__(self):
        super().__init__()
        self.cpu = 0
        self.ram = 0
        self.disk_write = False
        self.processes = []
        self.running = True
        self.daemon = True

    def get_port_color(self, pid: int) -> config.Color:
        try:
            p = psutil.Process(pid)
            conns = p.net_connections(kind='inet')
            if not conns: return config.COL_CYAN 
            
            for c in conns:
                if c.status == 'ESTABLISHED' and c.raddr:
                    port = c.raddr.port
                    if port in [80, 8080]: return (0.0, 0.0, 1.0, 1.0) # Blue (HTTP)
                    if port == 443: return (0.0, 1.0, 0.0, 1.0) # Green (HTTPS)
                    if port in [21, 22]: return (1.0, 1.0, 0.0, 1.0) # Yellow (SSH/FTP)
            return config.COL_CYAN 
        except:
            return config.COL_CYAN 

    def run(self):
        last_disk = 0
        while self.running:
            self.cpu = psutil.cpu_percent(interval=None)
            self.ram = psutil.virtual_memory().percent
            
            # Disk Activity
            try:
                disk = psutil.disk_io_counters()
                curr = disk.write_bytes
                self.disk_write = (curr > last_disk)
                last_disk = curr
            except: pass
            
            # Process Network Activity
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
                
                new_procs = []
                for pid, (name, color, port) in list(unique.items()):
                    new_procs.append((pid, name, color, port))
                
                # Sort by PID for stability
                new_procs.sort(key=lambda x: x[0])
                self.processes = new_procs[:12]
                
            except: pass
            time.sleep(1.0)

class WiredEngine:
    """Core Engine Class handling the Game Loop and OpenGL Context."""
    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode(
            (config.WIN_WIDTH, config.WIN_HEIGHT), 
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Navi v23.0 - [BREACH PROTOCOL]")
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
        gluPerspective(config.FOV, (config.WIN_WIDTH/config.WIN_HEIGHT), 0.1, config.RENDER_DISTANCE)
        glMatrixMode(GL_MODELVIEW)
        
        # Audio
        self.drone_sound = sounds.generate_drone(55, 10.0)
        self.screech_sound = sounds.generate_screech()
        self.explode_sound = sounds.generate_explosion()
        self.drone_channel = None
        if self.drone_sound:
            self.drone_channel = self.drone_sound.play(loops=-1, fade_ms=2000)
            self.drone_channel.set_volume(0.3)

        # Subsystems
        self.cap = cv2.VideoCapture(0)
        self.monitor = SystemMonitor()
        self.monitor.start()
        
        self.packet_queue = queue.Queue()
        self.listener = protocol.ProtocolListener(self.packet_queue)
        self.listener.start()
        
        self.data_buffer = collections.deque(maxlen=2000) 
        for _ in range(200): self.data_buffer.append(random.randint(0, 255))
        
        self.wifi_scanner = wifi_scanner.WifiScanner()
        self.wifi_scanner.start()

        # Rendering Resources
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.cam_tex = WebcamTexture()
        self.labels: Dict[Tuple[str, config.Color], TextTexture] = {} 
        self.post_process = PostProcess(config.WIN_WIDTH, config.WIN_HEIGHT)
        
        # State
        self.cam_pos = [0.0, 0.0, 5.0] 
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.world_offset_z = 0.0 
        self.rotation = 0.0
        self.last_texture_cleanup = time.time()
        self.packets: List[protocol.PacketObject] = [] 
        self.active_procs_objs = [] 
        self.start_time = time.time()
        self.glitch_level = 0.0 
        self.running = True
        self.clock = pygame.time.Clock()
        
        # Narrative State
        self.breach_mode = False
        self.ghost_room_reached = False
        self.blackwall_timer = 0
        self.in_blackwall_zone = False
        self.blackwall_state = {
            'warnings': 0, 
            'breached': False, 
            'last_warning_time': 0, 
            'message': None
        }
        self.zone_state = {}

        # --- Entity Initialization ---
        self.entities = [
            entities.InfiniteTunnel(self),
            entities.CyberArch(self),
            entities.GhostWall(self),
            entities.Hypercore(self),
            entities.SatelliteSystem(self),
            entities.PacketSystem(self),
            entities.CyberCity(self),
            entities.Blackwall(self),
            entities.AlienSwarm(self),
            entities.StatsWall(self),
            entities.WifiVisualizer(self),
            entities.DigitalRain(self, side='left', color=config.COL_HEX),
            entities.DigitalRain(self, side='right', color=config.COL_RED),
            entities.CustomModel(self),
            entities.GhostRoom(self),
            entities.IntroOverlay(self)
        ]
        
        self.particle_system = entities.ParticleSystem(self)
        self.entities.append(self.particle_system)

        # Pre-cache commonly used hex textures
        print("Pre-caching textures...")
        for i in range(256):
            h = f"{i:02X}"
            self.get_label(h, config.COL_HEX)
        print("Ready.")

    def get_label(self, text: str, color: config.Color) -> TextTexture:
        key = (text, color)
        if key not in self.labels:
            tex = TextTexture(self.font)
            tex.update(text, color)
            self.labels[key] = tex
        return self.labels[key]
        
    def get_ray_from_mouse(self, mx: int, my: int) -> Tuple[Any, Any]:
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        winY = float(viewport[3]) - float(my)
        winX = float(mx)
        
        try:
            start = gluUnProject(winX, winY, 0.0, modelview, projection, viewport)
            end = gluUnProject(winX, winY, 1.0, modelview, projection, viewport)
        except:
            return (None, None)
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length == 0: return (None, None)
        
        direction = (dx/length, dy/length, dz/length)
        return start, direction

    def ray_sphere_intersect(self, r_origin, r_dir, s_center, s_radius):
        lx = s_center[0] - r_origin[0]
        ly = s_center[1] - r_origin[1]
        lz = s_center[2] - r_origin[2]
        
        tca = lx*r_dir[0] + ly*r_dir[1] + lz*r_dir[2]
        if tca < 0: return False 
        
        d2 = (lx*lx + ly*ly + lz*lz) - tca*tca
        if d2 > s_radius * s_radius: return False
        
        return True

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == QUIT: self.running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE: self.running = False
            
            # Mouse Interaction
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                start, direction = self.get_ray_from_mouse(event.pos[0], event.pos[1])
                if start:
                    for obj in self.active_procs_objs:
                        if 'pos' not in obj: continue
                        center = obj['pos']
                        if self.ray_sphere_intersect(start, direction, center, 2.0):
                            print(f"TERMINATING PROCESS: {obj['name']} ({obj['pid']})")
                            try:
                                p = psutil.Process(obj['pid'])
                                p.terminate()
                                self.particle_system.explode(center[0], center[1], center[2], color=(1,0,0,1))
                                if self.explode_sound: self.explode_sound.play()
                            except: pass
                            break 

        # Continuous Movement
        keys = pygame.key.get_pressed()
        speed = 2.0 
        
        # Ghost Room Ending Logic
        ghost_room_z = config.ZONE_THRESHOLDS['GHOST_ROOM'] - 100.0
        current_global_z = self.cam_pos[2] + self.world_offset_z
        
        if self.ghost_room_reached:
            # Lock controls completely
            mdx, mdy = (0, 0)
            dx = 0; dz = 0
            # Force position to "sweet spot" (viewing the computer)
            # We want to be at ghost_room_z + 40 (looking back at it at -100)
            target_local_z = ghost_room_z - self.world_offset_z + 40.0
            
            # Smoothly interpolate to locked position
            self.cam_pos[0] = self.cam_pos[0] * 0.9 + 0.0 * 0.1
            self.cam_pos[2] = self.cam_pos[2] * 0.9 + target_local_z * 0.1
            
            # Smoothly face the center
            self.cam_yaw = self.cam_yaw * 0.9 + 0.0 * 0.1
            self.cam_pitch = self.cam_pitch * 0.9 + 0.0 * 0.1
            
        else:
            # Normal Controls
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
            
            # Check if we reached the room
            if abs(current_global_z - ghost_room_z) < 50.0:
                self.ghost_room_reached = True

        # Wall Collision
        next_z = self.cam_pos[2] + dz
        global_z = next_z + self.world_offset_z
        wall_z = config.ZONE_THRESHOLDS['DEEP_WEB'] 
        
        # Soft Wall / Resistance Logic
        if not self.blackwall_state['breached'] and global_z < wall_z + 100:
             dist = global_z - wall_z
             # The closer you get, the harder it pushes back
             # dist is positive (e.g. 100 -> 0)
             if dist < 50:
                 resistance = (50 - dist) * 0.05
                 # Push back
                 dz += resistance
                 
                 # Screen Shake / Feedback
                 self.cam_pos[0] += random.uniform(-0.05, 0.05)
                 self.cam_pos[1] += random.uniform(-0.05, 0.05)
        
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
        wall_z = config.ZONE_THRESHOLDS['DEEP_WEB'] 
        
        # Blackwall Narrative Logic
        if not self.blackwall_state['breached'] and global_z < (wall_z + 1000):
             dist = global_z - wall_z 
             
             # Force Breach (Pushing through)
             if dist < 10:
                 self.blackwall_state['breached'] = True
                 self.blackwall_state['message'] = "SYSTEM FAILURE // FORCED ENTRY"
                 if self.explode_sound: self.explode_sound.play()
                 # Push player forward into the void
                 self.cam_pos[2] -= 20.0 

             # Warning logic
             if dist < 300 and self.blackwall_state['warnings'] == 0:
                 self.blackwall_state['warnings'] = 1
                 self.blackwall_state['message'] = "WARNING: CLASSIFIED DATA"
                 self.blackwall_state['last_warning_time'] = time.time()
                 # Removed auto-push
             elif dist < 150 and self.blackwall_state['warnings'] == 1:
                  if time.time() - self.blackwall_state['last_warning_time'] > 1.0:
                      self.blackwall_state['warnings'] = 2
                      self.blackwall_state['message'] = "DANGER: LETHAL COUNTERMEASURES"
                      self.blackwall_state['last_warning_time'] = time.time()
             elif dist < 50 and self.blackwall_state['warnings'] == 2:
                  if time.time() - self.blackwall_state['last_warning_time'] > 1.0:
                      self.blackwall_state['warnings'] = 3
                      self.blackwall_state['message'] = "CRITICAL: BREACH IMMINENT"
                      self.blackwall_state['last_warning_time'] = time.time()
             
             if self.blackwall_state['warnings'] == 3 and time.time() - self.blackwall_state['last_warning_time'] > 2.0:
                   self.blackwall_state['breached'] = True
                   self.blackwall_state['message'] = "SYSTEM FAILURE // BREACH DETECTED"

        # Zone State
        self.in_blackwall_zone = False
        if global_z > config.ZONE_THRESHOLDS['SURFACE']:
            self.zone_state = {'name': 'SURFACE', 'grid_color': config.COL_GRID, 'tint': (0.8, 1.1, 1.0), 'distortion': 0.0}
        elif global_z > config.ZONE_THRESHOLDS['SPRAWL']:
            self.zone_state = {'name': 'SPRAWL', 'grid_color': (0.6, 0.0, 0.8, 0.5), 'tint': (0.9, 0.8, 1.1), 'distortion': 0.1}
        elif global_z > config.ZONE_THRESHOLDS['DEEP_WEB']:
            self.zone_state = {'name': 'DEEP_WEB', 'grid_color': (0.0, 0.8, 0.2, 0.5), 'tint': (0.7, 1.0, 0.7), 'distortion': 1.5}
        elif global_z > config.ZONE_THRESHOLDS['BLACKWALL']:
            self.zone_state = {'name': 'BLACKWALL', 'grid_color': (0.8, 0.0, 0.0, 0.5), 'tint': (1.2, 0.8, 0.8), 'distortion': 3.0}
            self.in_blackwall_zone = True
        elif global_z > config.ZONE_THRESHOLDS['GHOST_ROOM']:
            self.zone_state = {'name': 'OLD_NET', 'grid_color': (0.8, 0.8, 0.9, 0.6), 'tint': (0.6, 0.6, 0.7), 'distortion': 4.0}
        else:
            self.zone_state = {'name': 'GHOST_ROOM', 'grid_color': (0.0, 0.1, 0.0, 0.1), 'tint': (0.8, 1.0, 0.8), 'distortion': 1.0}

        # Breach Event
        if self.in_blackwall_zone and not self.breach_mode:
            self.blackwall_timer += 1.0 / 60.0
            if self.blackwall_timer > 10.0:
                self.breach_mode = True
                self.blackwall_state['message'] = "SYSTEM COMPROMISED - HUNTERS DEPLOYED"
                if self.screech_sound: self.screech_sound.play()
                for _ in range(5):
                    spawn_pos = (
                        self.cam_pos[0] + random.uniform(-10, 10), 
                        self.cam_pos[1] + random.uniform(-5, 5), 
                        self.cam_pos[2] - 50
                    )
                    self.entities.append(entities.Hunter(self, spawn_pos))
        
        # Audio
        if self.drone_channel:
            target_vol = 0.3 + min(0.7, len(self.packets) * 0.05)
            self.drone_channel.set_volume(target_vol)

        # Floating Origin
        if self.cam_pos[2] < -100.0:
            shift = self.cam_pos[2]
            self.cam_pos[2] -= shift
            self.world_offset_z += shift
        
        # Texture GC
        if time.time() - self.last_texture_cleanup > 10.0:
            self.last_texture_cleanup = time.time()
            threshold = time.time() - 60.0
            keys_to_delete = [k for k, t in self.labels.items() if t.last_used < threshold]
            for k in keys_to_delete:
                try: glDeleteTextures([self.labels[k].texture_id])
                except: pass
                del self.labels[k]

    def draw(self):
        self.post_process.begin()
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        rad_yaw = math.radians(self.cam_yaw)
        rad_pitch = math.radians(self.cam_pitch)
        lx = math.sin(rad_yaw) * math.cos(rad_pitch)
        ly = math.sin(rad_pitch)
        lz = -math.cos(rad_yaw) * math.cos(rad_pitch)
        gluLookAt(
            self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
            self.cam_pos[0] + lx, self.cam_pos[1] + ly, self.cam_pos[2] + lz,
            0, 1, 0
        )
        
        for entity in self.entities: entity.draw()
        
        # Finalize Frame
        t = time.time() - self.start_time
        tint = self.zone_state.get('tint', (1,1,1))
        glitch = self.glitch_level + (2.0 if self.breach_mode else 0.0)
        self.post_process.end(t, glitch, tint, self.breach_mode)
        
        # HUD Overlay
        msg = self.blackwall_state.get('message')
        if msg:
             self._draw_overlay_message(msg)
        
        pygame.display.flip()

    def _draw_overlay_message(self, msg: str):
         glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
         gluOrtho2D(0, self.screen.get_width(), self.screen.get_height(), 0)
         glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
         glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND)
         
         lbl = self.get_label(msg, (1.0, 0.2, 0.2, 1.0))
         glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, lbl.texture_id)
         
         x = (self.screen.get_width() - lbl.width) / 2
         y = (self.screen.get_height() / 2) - 100
         alpha = 0.5 + math.sin(time.time()*15)*0.5
         glColor4f(1, 1, 1, alpha)
         
         glBegin(GL_QUADS)
         glTexCoord2f(0, 0); glVertex2f(x, y)
         glTexCoord2f(1, 0); glVertex2f(x + lbl.width, y)
         glTexCoord2f(1, 1); glVertex2f(x + lbl.width, y + lbl.height)
         glTexCoord2f(0, 1); glVertex2f(x, y + lbl.height)
         glEnd()
         
         glDisable(GL_TEXTURE_2D); glEnable(GL_DEPTH_TEST)
         glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

    def loop(self):
        try:
            print("Entering main loop...")
            frame_count = 0
            while self.running:
                self.handle_input()
                self.update()
                self.draw()
                self.clock.tick(config.FPS_TARGET)
                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"FPS: {self.clock.get_fps():.2f}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            print("Cleaning up...")
            self.listener.stop()
            self.monitor.running = False
            self.wifi_scanner.stop()
            self.cap.release()
            pygame.quit()

if __name__ == "__main__":
    WiredEngine().loop()