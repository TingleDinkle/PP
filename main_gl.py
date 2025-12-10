import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
import queue
import psutil
import threading
import collections
import pyperclip
import wifi_scanner
import protocol

# --- Constants ---
WIN_WIDTH = 800
WIN_HEIGHT = 600
TUNNEL_DEPTH = 40.0
WIFI_Z_OFFSET = -35.0 # Z depth for WiFi entities
MAX_PACKETS_DISPLAYED = 40 # Limit the number of packets displayed for performance

# Colors (Normalized OpenGL RGBA)
COL_CYAN = (0.0, 1.0, 1.0, 1.0)
COL_RED = (1.0, 0.2, 0.2, 1.0)
COL_GRID = (0.0, 0.15, 0.3, 0.4) # Semi-transparent for solid walls
COL_DARK = (0.02, 0.02, 0.04, 1.0)
COL_TEXT = (0.8, 0.8, 0.8, 1.0)
COL_GHOST = (0.2, 1.0, 0.2, 0.8) # Ghost with alpha
COL_WHITE = (1.0, 1.0, 1.0, 1.0)
COL_YELLOW = (1.0, 1.0, 0.0, 1.0) # For original Yellow objects
COL_HEX = (0.0, 0.6, 0.0, 1.0) # Matrix green for hex

# --- GLSL Shaders ---
VERTEX_SHADER = """
#version 120
void main() {
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = ftransform();
}
"""

FRAGMENT_SHADER = """
#version 120
uniform sampler2D tex;
uniform float time;
uniform float glitch_intensity;

void main() {
    vec2 uv = gl_TexCoord[0].st;
    
    // Simple Scanline (Fast)
    float scanline = sin(uv.y * 800.0) * 0.05;
    
    // Sample Texture
    vec3 col = texture2D(tex, uv).rgb;
    
    // Simple Green/Cyan Tint (Fast)
    col *= vec3(0.8, 1.1, 1.0);
    
    // Apply scanline
    col -= scanline;

    gl_FragColor = vec4(col, 1.0);
}
"""

class ShaderPostProcess:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        try:
            self.program = compileProgram(
                compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"Shader Compilation Failed: {e}")
            self.program = 0
            
        self.fbo = glGenFramebuffers(1)
        self.tex = glGenTextures(1)
        self.rbo = glGenRenderbuffers(1)
        
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.rbo)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def begin(self):
        if self.program:
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glViewport(0, 0, self.width, self.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end(self, time_val, intensity):
        if not self.program: return
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "tex"), 0)
        glUniform1f(glGetUniformLocation(self.program, "time"), time_val)
        glUniform1f(glGetUniformLocation(self.program, "glitch_intensity"), intensity)
        
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glColor3f(1,1,1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1) # Changed from glVertex3f
        glTexCoord2f(0, 1); glVertex2f(-1, 1) # Changed from glVertex3f
        glEnd()
        glEnable(GL_DEPTH_TEST)
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
        glUseProgram(0)

class TextTexture:
    def __init__(self, font):
        self.font = font
        self.texture_id = glGenTextures(1)
        self.width = 0
        self.height = 0
        self._current_text = None # Store the last rendered text
        self._current_color = None # Store the last rendered color

    def update(self, text, color=(1.0, 1.0, 1.0, 1.0)): # Default to opaque white float
        # Check if text and color are identical to the last update
        if text == self._current_text and color == self._current_color:
            return # No change, skip expensive update

        self._current_text = text
        self._current_color = color

        # Normalize color components to 0-255 integers for pygame.font.render
        processed_color = [0, 0, 0, 255] # Default to opaque white
        
        if isinstance(color, tuple):
            if len(color) >= 3:
                # Handle RGB or RGBA input
                for i in range(min(len(color), 3)): # Process R, G, B
                    if isinstance(color[i], float):
                        processed_color[i] = int(color[i] * 255)
                    else: # Assume int
                        processed_color[i] = color[i]
                
                if len(color) == 4: # Process Alpha
                    if isinstance(color[3], float):
                        processed_color[3] = int(color[3] * 255)
                    else: # Assume int
                        processed_color[3] = color[3]
            elif len(color) == 1 and isinstance(color[0], (float, int)): # Grayscale
                val = int(color[0] * 255) if isinstance(color[0], float) else color[0]
                processed_color[0:3] = [val, val, val]
        
        # Ensure all components are within 0-255 range
        processed_color = tuple(max(0, min(255, c)) for c in processed_color)

        surf = self.font.render(text, True, processed_color)
        
        # Create a new surface with an alpha channel
        new_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        new_surf.fill((0, 0, 0, 0)) # Start with a transparent background
        new_surf.blit(surf, (0, 0)) # Blit the rendered text onto it
        
        # Update width and height from the new surface
        self.width = new_surf.get_width()
        self.height = new_surf.get_height()

        # Bind the texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Convert surface to string data
        data = pygame.image.tostring(new_surf, "RGBA", True) # Use True for flipped to match OpenGL

        # Upload texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

    def draw(self, x, y, z, scale=0.02, yaw=0):
        if self.width == 0 or self.height == 0: return # Don't draw empty textures
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(yaw, 0, 1, 0)
        glScalef(scale, scale, 1)
        glTranslatef(-self.width / 2, -self.height / 2, 0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # glColor4f(1, 1, 1, 1) # REMOVED: Allow caller to set color/alpha
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
        rgba = np.zeros((240, 320, 4), dtype=np.uint8) # Original transparent black
        rgba[edges > 0] = [int(COL_GHOST[0]*255), int(COL_GHOST[1]*255), int(COL_GHOST[2]*255), int(COL_GHOST[3]*255)] # Use COL_GHOST
        rgba = cv2.flip(rgba, 0)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 320, 240, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba)

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

class DigitalRain:
    def __init__(self, font, data_source, side='left', color=(0.0, 1.0, 0.0, 1.0), columns=20):
        self.font = font
        self.data_source = data_source
        self.side = side
        self.color = color
        self.columns = columns
        self.drops = []
        for i in range(columns):
            self.drops.append({
                'col_idx': i,
                'y': random.uniform(-10, 10),
                'speed': random.uniform(0.1, 0.3),
                'chars': [self._get_char() for _ in range(random.randint(5, 15))]
            })

    def _get_char(self):
        # Pull a byte from the data source if available
        if self.data_source:
            # We peek randomly or just take the last one? 
            # Let's take a random sample from the buffer to make it look active
            val = self.data_source[random.randint(0, len(self.data_source)-1)]
            return f'{val:02X}'
        else:
            return f'{random.randint(0,255):02X}'

    def update(self):
        for drop in self.drops:
            drop['y'] -= drop['speed']
            if drop['y'] < -5:
                drop['y'] = random.uniform(5, 10)
                drop['speed'] = random.uniform(0.1, 0.3)
                drop['chars'] = [self._get_char() for _ in range(random.randint(5, 15))]
            
            # Randomly change characters with real data
            if random.random() < 0.1: # Increased update rate
                drop['chars'][random.randint(0, len(drop['chars'])-1)] = self._get_char()

    def draw(self, wired_engine):
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_TEXTURE_BIT) # Save state
        glDisable(GL_LIGHTING) # Text is self-illuminated
        glDepthMask(GL_FALSE)  # Don't write to depth buffer (transparent overlay)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D) # Ensure texture is enabled
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE) # Critical for alpha texture blending
        
        # Batching: Group positions by character to reduce draw calls
        # batches[char_str] = [ (x, y, z, alpha), ... ]
        batches = {}
        
        for drop in self.drops:
            for i, char in enumerate(drop['chars']):
                # Spacing
                z = -10 + (drop['col_idx'] * 0.8) 
                y = drop['y'] + (i * 0.4)
                
                if y > 5 or y < -5: continue # Clip
                
                # Calculate X based on side
                if self.side == 'left':
                    x = -3.8
                    yaw = 90
                else: 
                    x = 3.8
                    yaw = -90
                
                # Apply nudge
                # We can't apply simple translation in batch easily without matrix ops,
                # so we pre-calculate world coordinates.
                # However, drawing quads aligned to axes is easier. 
                # The wall rain is rotated 90 deg around Y.
                # Left wall (-3.8): facing right (+X). Text plane is YZ.
                # Right wall (3.8): facing left (-X). Text plane is YZ.
                
                alpha = 1.0 - (i / len(drop['chars']))
                
                if char not in batches:
                    batches[char] = []
                batches[char].append((x, y, z, alpha))

        # Draw batches
        # We need to handle the rotation. 
        # Since all text on one wall shares the same rotation, we can set the matrix once per wall
        # if we treat coordinates as local to the wall.
        
        glPushMatrix()
        if self.side == 'left':
            glTranslatef(-3.8, 0, 0) # Move to wall
            glRotatef(90, 0, 1, 0)   # Rotate
        else:
            glTranslatef(3.8, 0, 0)
            glRotatef(-90, 0, 1, 0)
        
        glTranslatef(0, 0, 0.1) # Nudge out
        
        # Now we are in wall-local space. Text draws on XY plane? 
        # No, TextTexture.draw draws on XY plane (0..width, 0..height).
        # Our wall logic calculated 'z' as depth in tunnel, and 'y' as height.
        # In the rotated space:
        #   Original Global Y -> Local Y
        #   Original Global Z -> Local X (because of 90 deg rot)
        
        # Let's verify:
        # Left Wall: Rot 90 deg Y.
        # Local X+ points to Global Z- 
        # Local Z+ points to Global X+
        # Wait, standard rotation:
        # Rot 90 Y: (1,0,0) -> (0,0,-1). (0,0,1) -> (1,0,0).
        # So Global Z corresponds to Local X.
        
        scale = 0.02

        for char, instances in batches.items():
            tex = wired_engine.get_label(char, self.color)
            if tex.width == 0: continue
            
            glBindTexture(GL_TEXTURE_2D, tex.texture_id)
            
            w = tex.width * scale
            h = tex.height * scale
            
            # Center offset
            off_x = -w / 2
            off_y = -h / 2
            
            glBegin(GL_QUADS)
            for (gx, gy, gz, alpha) in instances:
                glColor4f(1, 1, 1, alpha)
                
                # Convert Global Y/Z to Local X/Y
                # In previous code: glTranslatef(-3.8, y, z) then Rotate(90, 0, 1, 0)
                # Matrix order: Translate(T) -> Rotate(R) * v
                # New logic: Rotate(R) -> Translate(T_wall) -> Translate(T_pos) * v ? No.
                
                # Let's stick to the local coordinates derived from y and z.
                # Code was: glTranslatef(-3.8, y, z); glRotatef(90, ...);
                # This means we move to (-3.8, y, z) THEN rotate. 
                # So the coordinate system is rotated AT that point.
                # Drawing (0,0,0) puts it at (-3.8, y, z).
                
                # If we Rotate FIRST (glRotatef(90...)), then X becomes Z, Z becomes -X.
                # We want to place items at 'y' (up) and 'z' (depth).
                
                # Let's simplify. We can use the previous logic but inside the loop only for translation?
                # No, too many matrix calls.
                
                # Optimization: 
                # We are rendering in a space rotated by 90/-90.
                # Local Y is Global Y.
                # Local X is -Global Z (Left Wall) or Global Z (Right Wall)?
                
                # Previous: Translate(WallX, y, z) -> Rotate(90) -> Draw(0,0).
                # This places the quad at WallX, y, z, facing correctly.
                
                # New: Translate(WallX, 0, 0) -> Rotate(90) -> Draw(local_x, local_y).
                # We need to map (y, z) to (local_x, local_y).
                
                # If we translate to (-3.8, 0, 0) and rotate 90 Y:
                # Local X+ is Global Z-.
                # Local Y+ is Global Y+.
                # So: local_x = -z. local_y = y.
                
                # Right wall (3.8, 0, 0), rotate -90 Y:
                # Local X+ is Global Z+.
                # So: local_x = z. local_y = y.
                
                lx = -gz if self.side == 'left' else gz
                ly = gy
                
                # Draw Quad
                glTexCoord2f(0, 0); glVertex3f(lx + off_x, ly + off_y, 0)
                glTexCoord2f(1, 0); glVertex3f(lx + off_x + w, ly + off_y, 0)
                glTexCoord2f(1, 1); glVertex3f(lx + off_x + w, ly + off_y + h, 0)
                glTexCoord2f(0, 1); glVertex3f(lx + off_x, ly + off_y + h, 0)

            glEnd()
            
        glPopMatrix()
        glPopAttrib() # Restore state

class SystemMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cpu = 0
        self.ram = 0
        self.disk_write = False
        self.processes = []
        self.daemon = True
        self.running = True
        self.last_disk = 0

    def get_port_color(self, pid):
        try:
            p = psutil.Process(pid)
            conns = p.net_connections(kind='inet')
            if not conns: return COL_CYAN # Return RGBA for glColor4f
            for c in conns:
                if c.status == 'ESTABLISHED' and c.raddr:
                    port = c.raddr.port
                    if port == 80 or port == 8080: return (0.0, 0.0, 1.0, 1.0) # Blue with Alpha
                    if port == 443: return (0.0, 1.0, 0.0, 1.0) # Green with Alpha
                    if port in [21, 22]: return (1.0, 1.0, 0.0, 1.0) # Yellow with Alpha
            return COL_CYAN # Return RGBA for glColor4f
        except:
            return COL_CYAN # Return RGBA for glColor4f

    def run(self):
        while self.running:
            self.cpu = psutil.cpu_percent(interval=None)
            self.ram = psutil.virtual_memory().percent
            
            try:
                disk = psutil.disk_io_counters()
                curr = disk.write_bytes
                self.disk_write = (curr > self.last_disk)
                self.last_disk = curr
            except: pass
            
            try:
                conns = [p for p in psutil.net_connections() if p.status == 'ESTABLISHED']
                unique = {}
                for c in conns:
                    if c.pid and c.pid not in unique:
                        try:
                            p = psutil.Process(c.pid)
                            # Find the port for the current connection 'c'
                            port = None
                            if c.raddr and c.raddr.port:
                                port = c.raddr.port
                            unique[c.pid] = (p.name(), self.get_port_color(c.pid), port) # Store name, color AND port
                        except: pass
                self.processes = []
                for pid, (name, color, port) in list(unique.items())[:12]:
                    self.processes.append((pid, name, color, port))
            except: pass
            
            time.sleep(1.0)

class WiredEngine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Navi")
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        # OpenGL Config
        glEnable(GL_DEPTH_TEST) # Re-enable depth test for proper drawing
        glEnable(GL_BLEND) # Re-enable blend for transparency
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.5) 
        glEnable(GL_COLOR_MATERIAL) 
        glShadeModel(GL_FLAT) # Ensure flat shading

        # Camera
        glMatrixMode(GL_PROJECTION)
        gluPerspective(60, (WIN_WIDTH/WIN_HEIGHT), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Logic
        self.cap = cv2.VideoCapture(0)
        self.monitor = SystemMonitor()
        self.monitor.start()
        
        self.packet_queue = queue.Queue()
        self.listener = protocol.ProtocolListener(self.packet_queue)
        self.listener.start()
        
        # Shared Data Buffer for Wall Rain (Real Sniffed Data)
        # deque with maxlen to keep a running history of bytes
        self.data_buffer = collections.deque(maxlen=2000) 
        # Pre-fill with some random data so it's not empty initially
        for _ in range(200): self.data_buffer.append(random.randint(0, 255))

        self.wifi_scanner = wifi_scanner.WifiScanner() # Init WiFi scanner
        self.wifi_scanner.start()

        # Resources
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.cam_tex = WebcamTexture()
        self.labels = {} 
        self.post_process = ShaderPostProcess(WIN_WIDTH, WIN_HEIGHT)
        
        # Digital Rain (Left: Green, Right: Red)
        self.rain_left = DigitalRain(self.font, self.data_buffer, side='left', color=COL_HEX)
        self.rain_right = DigitalRain(self.font, self.data_buffer, side='right', color=COL_RED)
        
        # Pre-cache all hex textures to prevent lag during runtime
        print("Pre-caching textures...")
        for i in range(256):
            h = f"{i:02X}"
            self.get_label(h, COL_HEX)
            self.get_label(h, COL_RED)
        print("Textures cached.")
        
        # Generate Glow Texture for Packet Heads
        self.glow_tex = self.create_glow_texture()

        # State
        self.cam_pos = [0.0, 0.0, 5.0] # Start slightly back
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        
        self.rotation = 0.0
        self.packets = [] 
        self.active_procs_objs = [] 
        self.start_time = time.time()
        self.glitch_level = 0.0 # This uniform existed in the shader this was based on

        self.running = True
        self.clock = pygame.time.Clock() # For smooth framerate

    def create_glow_texture(self):
        # Generate a radial gradient texture (32x32)
        size = 64
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        max_radius = size // 2
        
        # Manually draw radial gradient
        for r in range(max_radius, 0, -1):
            alpha = int(255 * (1.0 - (r / max_radius))**2) # Quadratic falloff for "hot" core
            color = (255, 255, 255, alpha)
            pygame.draw.circle(surface, color, (center, center), r)
            
        # Convert to OpenGL Texture
        tex_id = glGenTextures(1)
        data = pygame.image.tostring(surface, "RGBA", True)
        
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return tex_id

    def get_label(self, text, color=(200, 200, 200)):
        key = (text, color)
        if key not in self.labels:
            tex = TextTexture(self.font)
            tex.update(text, color)
            self.labels[key] = tex
        return self.labels[key]

    def handle_input_continuous(self):
        keys = pygame.key.get_pressed()
        speed = 0.5
        
        # Rotation
        mdx, mdy = pygame.mouse.get_rel()
        self.cam_yaw += mdx * 0.1
        self.cam_pitch -= mdy * 0.1 
        self.cam_pitch = max(-89, min(89, self.cam_pitch))
        
        # Movement
        rad_yaw = math.radians(self.cam_yaw)
        forward_x = math.sin(rad_yaw)
        forward_z = -math.cos(rad_yaw)
        right_x = math.cos(rad_yaw)
        right_z = math.sin(rad_yaw)
        
        if keys[K_w]:
            self.cam_pos[0] += forward_x * speed
            self.cam_pos[2] += forward_z * speed
        if keys[K_s]:
            self.cam_pos[0] -= forward_x * speed
            self.cam_pos[2] -= forward_z * speed
        if keys[K_a]:
            self.cam_pos[0] -= right_x * speed
            self.cam_pos[2] -= right_z * speed
        if keys[K_d]:
            self.cam_pos[0] += right_x * speed
            self.cam_pos[2] += right_z * speed
            
        # Clamp movement to tunnel
        self.cam_pos[0] = max(-3.5, min(3.5, self.cam_pos[0]))
        self.cam_pos[1] = max(-2.5, min(2.5, self.cam_pos[1]))
        self.cam_pos[2] = max(-TUNNEL_DEPTH + 2, min(10.0, self.cam_pos[2]))

    def handle_input_events(self, event):
        # No specific discrete events in this reverted version
        pass

    def update(self):
        # 1. Vision
        ok, frame = self.cap.read()
        if ok:
            self.cam_tex.update(frame)

        # 2. Packets
        try:
            while True:
                p = self.packet_queue.get_nowait()
                self.packets.append(p)
                
                # Extract bytes from payload string for the wall rain
                # Packet payload is a string summary, let's just take the ASCII values
                if p.payload:
                    for char in p.payload:
                        self.data_buffer.append(ord(char) % 256)
                
                if len(self.packets) > MAX_PACKETS_DISPLAYED:
                    self.packets.pop(0) # Remove oldest packet
                self.glitch_level = 0.3 # Reduced initial glitch level
        except queue.Empty: pass
        
        # Update and remove old packets
        # DATA HIGHWAY (Straight Lanes for Readability)
        updated_packets = []
        t = time.time()
        
        for i, p in enumerate(self.packets):
            # Initialize history and lane
            if not hasattr(p, 'history'):
                p.history = collections.deque(maxlen=20)
                p.base_z_offset = random.uniform(0, TUNNEL_DEPTH)
                
                # Assign Lane based on Protocol
                # Lane Width = 2.0
                if p.protocol == 'TCP':
                    p.lane_x = -2.5 # Left Lane
                elif p.protocol == 'UDP':
                    p.lane_x = 2.5 # Right Lane
                else:
                    p.lane_x = 0.0 # Center Lane
                
                # Add slight random Y offset to avoid perfect stacking
                p.lane_y = random.uniform(-1.5, 1.5)

            p.prev_x = p.x 
            p.prev_y = p.y
            p.prev_z = p.z
            p.history.append((p.x, p.y, p.z))

            # Continuous Z Movement (Slower for reading)
            scroll_speed = 2.0 # Reduced from 3.0
            range_len = TUNNEL_DEPTH + 10.0
            
            z_val = (p.base_z_offset + t * scroll_speed) % range_len
            p.z = -TUNNEL_DEPTH + z_val
            
            # Fixed X/Y (Highway)
            p.x = p.lane_x
            p.y = p.lane_y
            
            # No rotation/angle needed for straight lines
            p.angle = 0 

            updated_packets.append(p)
        self.packets = updated_packets

        # 3. Satellites
        current_procs = self.monitor.processes
        if len(self.active_procs_objs) != len(current_procs):
            self.active_procs_objs = []
            for i, (pid, name, color, port) in enumerate(current_procs): # Unpack pid, name, color, AND port
                self.active_procs_objs.append({
                    'name': name,
                    'pid': pid,
                    'angle': (i / len(current_procs)) * 2 * math.pi,
                    'radius': 3.0,
                    'color': color, # Store color
                    'port': port # Store port
                })
        
        self.rain_left.update()
        self.rain_right.update()
        self.rotation += 0.5
        self.glitch_level *= 0.9 # Decay glitch

    def draw_tunnel(self):
        # This draws a grid tunnel.
        
        if self.monitor.disk_write:
            glColor3f(COL_WHITE[0], COL_WHITE[1], COL_WHITE[2])
        else:
            glColor3f(COL_GRID[0], COL_GRID[1], COL_GRID[2])
        
        for z in range(0, int(-TUNNEL_DEPTH), -2):
            glBegin(GL_LINE_LOOP)
            glVertex3f(-4, -3, z)
            glVertex3f(4, -3, z)
            glVertex3f(4, 3, z)
            glVertex3f(-4, 3, z)
            glEnd()
        glBegin(GL_LINES)
        for x, y in [(-4,-3), (4,-3), (4,3), (-4,3)]: # Fixed typo: (-4,3) should be last
            glVertex3f(x, y, 0)
            glVertex3f(x, y, -TUNNEL_DEPTH)
        glEnd()

    def draw_packets(self):
        # DATA HIGHWAY VISUALIZATION
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_TEXTURE_BIT)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive for trails
        glDisable(GL_DEPTH_TEST) # X-Ray trails
        
        # 1. Draw Straight Trails (Laser Beams)
        glDisable(GL_TEXTURE_2D)
        glLineWidth(3.0) 
        
        for p in self.packets:
            if not hasattr(p, 'history') or len(p.history) < 2: continue
            
            glBegin(GL_LINE_STRIP)
            for i, (hx, hy, hz) in enumerate(p.history):
                # Fade trail
                alpha = (i / len(p.history)) * 0.5
                glColor4f(p.color[0], p.color[1], p.color[2], alpha)
                glVertex3f(hx, hy, hz)
            glEnd()

        # 2. Draw Packet Data (The "Payload")
        # Switch to standard blending for text readability
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        for p in self.packets:
            # Draw Glowing Head
            # Just a simple point or quad? Quad with glow texture is best.
            # Reuse billboard logic? 
            # Simplified: Just draw text.
            
            # TEXT LABEL
            # Color based on protocol
            label_col = p.color # Use packet color (Cyan/Orange)
            
            # Get text texture
            # Limit text length to avoid clutter
            display_text = p.payload[:20] + "..." if len(p.payload) > 20 else p.payload
            lbl = self.get_label(f"[{p.protocol}] {display_text}", label_col)
            
            glPushMatrix()
            glTranslatef(p.x, p.y + 0.3, p.z) # Float slightly above trail
            
            # BILLBOARDING: Always face camera
            # In simple FPS cam (cam is at 0,0,5 looking -Z), 
            # we just need to cancel out the modelview rotation?
            # Or just start with Identity (facing +Z) and... wait.
            # If we just draw in XY plane at Z depth, it faces the camera implicitly in this setup?
            # No, camera rotates.
            # We need to rotate OPPOSITE to self.cam_yaw, self.cam_pitch.
            glRotatef(-self.cam_yaw, 0, 1, 0)
            glRotatef(-self.cam_pitch, 1, 0, 0)
            
            # Scale for readability
            scale = 0.008 # Large enough to read
            glScalef(scale, scale, scale)
            glTranslatef(-lbl.width/2, 0, 0) # Center text
            
            lbl.draw(0, 0, 0, 1.0)
            glPopMatrix()
            
        glPopAttrib()



    def draw_satellites(self):
        # "System Bus" Visualization
        # A rotating ring of data points around the core
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING) 
        glEnable(GL_BLEND)
        glLineWidth(1.5) 
        
        center_z = -15.0 # Match Hypercore Z
        radius = 8.0 # Wider ring around the core
        
        glPushMatrix()
        glTranslatef(0, 0, center_z)
        
        # 1. Draw the Bus Ring (Wireframe Circle)
        # Counter-rotate against the core for contrast
        glRotatef(-self.rotation * 0.5, 0, 0, 1) 
        
        glColor4f(0.0, 0.5, 0.5, 0.3) 
        glBegin(GL_LINE_LOOP)
        for i in range(80):
            theta = 2.0 * math.pi * i / 80.0
            glVertex3f(math.cos(theta) * radius, math.sin(theta) * radius, 0)
        glEnd()
        
        # 2. Draw Process Nodes on the Ring
        # We use a fixed time offset for rotation to make them orbit
        orbit_speed = 0.2
        t = time.time()
        
        for i, obj in enumerate(self.active_procs_objs):
            # Smooth Orbit: Angle = Base Angle + Time
            # Spread them out evenly: i / count
            base_angle = (i / max(1, len(self.active_procs_objs))) * 2 * math.pi
            angle = base_angle + (t * orbit_speed)
            
            px = math.cos(angle) * radius
            py = math.sin(angle) * radius
            
            # Draw Node (Diamond)
            glPushMatrix()
            glTranslatef(px, py, 0)
            glRotatef(self.rotation * 5, 0, 0, 1) # Spin the node
            glScalef(0.2, 0.2, 0.2)
            
            glColor4f(*obj['color'])
            glBegin(GL_QUADS)
            glVertex3f(0, 1, 0); glVertex3f(1, 0, 0)
            glVertex3f(0, -1, 0); glVertex3f(-1, 0, 0)
            glEnd()
            glPopMatrix()
            
            # 3. Draw Connection Line (faint)
            glBegin(GL_LINES)
            glColor4f(obj['color'][0], obj['color'][1], obj['color'][2], 0.1)
            glVertex3f(px, py, 0)
            glVertex3f(0, 0, 0)
            glEnd()
            
            # 4. Draw Label (Readable!)
            # Only draw if it's in the top half (or just prevent overlapping)
            # Let's draw all but billboard them
            glPushMatrix()
            glTranslatef(px, py + 0.5, 0) # Above the node
            
            # Billboard: Inverse rotation of the ring group?
            # We rotated -self.rotation * 0.5 around Z.
            # So we rotate +self.rotation * 0.5 around Z to stay upright?
            glRotatef(self.rotation * 0.5, 0, 0, 1)
            
            # Scale down
            glScalef(0.015, 0.015, 0.015) # Readable size
            
            # Centering
            # We don't have text width here easily without get_label first
            # But get_label caches it.
            
            process_text = obj['name']
            lbl = self.get_label(process_text, COL_WHITE)
            glTranslatef(-lbl.width/2, 0, 0)
            
            glEnable(GL_TEXTURE_2D)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            lbl.draw(0, 0, 0, 1.0)
            glDisable(GL_TEXTURE_2D)
            
            glPopMatrix()

        glPopMatrix()
        glPopAttrib()


    def draw_chip(self):
        # "Utterly Complex" Hypercore
        # Force visibility: Disable depth test to draw ON TOP of everything (Hologram style)
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING) 
        glDisable(GL_DEPTH_TEST) # ALWAYS VISIBLE
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive glow
        
        glPushMatrix()
        glTranslatef(0, 0, -15.0) # Push back a bit more to be in the "center" of the tunnel view
        
        # 1. The Nucleus (Pulsing Energy Sphere)
        pulse = 1.0 + math.sin(time.time() * 3.0) * 0.2
        glColor4f(1.0, 0.1, 0.1, 1.0) # Full opacity red
        glPushMatrix()
        glScalef(pulse, pulse, pulse)
        sphere = gluNewQuadric()
        gluSphere(sphere, 0.8, 16, 16) # Larger
        glPopMatrix()
        
        # 2. Inner Rotating Cube (Wireframe)
        glPushMatrix()
        glRotatef(self.rotation * 3, 1, 1, 0)
        glColor4f(1.0, 0.6, 0.0, 0.8) # Brighter orange
        glLineWidth(3.0) # Thicker
        s = 2.0
        glScalef(s, s, s)
        self.draw_wire_cube()
        glPopMatrix()
        
        # 3. Outer Counter-Rotating Octahedron
        glPushMatrix()
        glRotatef(-self.rotation * 2, 0, 1, 1)
        glColor4f(0.0, 1.0, 1.0, 0.6)
        glLineWidth(2.0)
        s = 3.5 # Much larger
        glScalef(s, s, s)
        self.draw_wire_octahedron()
        glPopMatrix()
        
        # 4. Gyroscopic Rings
        for i in range(4): # More rings
            glPushMatrix()
            glRotatef(self.rotation * (1 + i*0.3), (i==0), (i==1), (i==2))
            glColor4f(0.2, 0.4, 1.0, 0.5)
            gluDisk(gluNewQuadric(), 4.0 + i*0.4, 4.1 + i*0.4, 64, 1)
            glPopMatrix()
            
        # 5. DATA RINGS (Text Orbiting the Core)
        # Draw a ring of rotating hex codes
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        chars = "0123456789ABCDEF"
        num_chars = 16
        ring_radius = 5.5
        
        glPushMatrix()
        glRotatef(self.rotation * 10, 0, 1, 0) # Rotate the ring of text
        
        for i in range(num_chars):
            angle = (i / num_chars) * 2 * math.pi
            x = math.cos(angle) * ring_radius
            z = math.sin(angle) * ring_radius
            
            # Pick a character based on time + index to make it "scramble"
            char_idx = int(time.time() * 10 + i) % 16
            char = chars[char_idx]
            
            lbl = self.get_label(char, COL_CYAN)
            
            glPushMatrix()
            glTranslatef(x, 0, z)
            # Billboard to face outward or camera? Outward is cooler for a ring.
            glRotatef(-math.degrees(angle) - 90, 0, 1, 0) 
            
            lbl.draw(0, 0, 0, 0.05) # Draw large char
            glPopMatrix()
            
        glPopMatrix()

        glPopMatrix()
        glPopAttrib()

    def draw_wire_cube(self):
        glBegin(GL_LINES)
        for x in [-1, 1]:
            for y in [-1, 1]:
                glVertex3f(x, y, -1); glVertex3f(x, y, 1)
                glVertex3f(x, -1, y); glVertex3f(x, 1, y)
                glVertex3f(-1, x, y); glVertex3f(1, x, y)
        glEnd()

    def draw_wire_octahedron(self):
        glBegin(GL_LINES)
        # Top pyramid
        glVertex3f(0,1,0); glVertex3f(1,0,0)
        glVertex3f(0,1,0); glVertex3f(-1,0,0)
        glVertex3f(0,1,0); glVertex3f(0,0,1)
        glVertex3f(0,1,0); glVertex3f(0,0,-1)
        # Bottom pyramid
        glVertex3f(0,-1,0); glVertex3f(1,0,0)
        glVertex3f(0,-1,0); glVertex3f(-1,0,0)
        glVertex3f(0,-1,0); glVertex3f(0,0,1)
        glVertex3f(0,-1,0); glVertex3f(0,0,-1)
        # Equator
        glVertex3f(1,0,0); glVertex3f(0,0,1)
        glVertex3f(0,0,1); glVertex3f(-1,0,0)
        glVertex3f(-1,0,0); glVertex3f(0,0,-1)
        glVertex3f(0,0,-1); glVertex3f(1,0,0)
        glEnd()

    def draw_stats_walls(self):
        glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        ram_lbl = self.get_label(f"RAM: {self.monitor.ram}%", (0.0, 1.0, 1.0, 1.0)) # Cyan
        glPushMatrix()
        glTranslatef(-3.9, 0, -5)
        glRotatef(90, 0, 1, 0) 
        glTranslatef(0, 0, 0.05) # Push slightly out from the wall
        ram_lbl.draw(0, 0, 0, 0.03) # Original scale
        glPopMatrix()
        
        cpu_lbl = self.get_label(f"CPU: {self.monitor.cpu}%", (1.0, 0.2, 0.2, 1.0)) # Red
        glPushMatrix()
        glTranslatef(3.9, 0, -5)
        glRotatef(-90, 0, 1, 0) 
        glTranslatef(0, 0, 0.05) # Push slightly out from the wall
        cpu_lbl.draw(0, 0, 0, 0.03) # Original scale
        glPopMatrix()
        
        glPopAttrib()

    def draw_wifi_networks(self):
        networks = self.wifi_scanner.networks
        if not networks:
            return
            
        glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        
        # Starting position for WiFi network labels
        start_x = 3.9  # Right wall
        start_z = WIFI_Z_OFFSET # Z depth for WiFi entities, defined in constants
        y_offset_step = 0.4 # Vertical spacing between networks

        for i, (ssid, signal) in enumerate(networks[:10]): # Limit to first 10 for display
            # Determine color based on signal strength
            if signal >= 70:
                color = (0.0, 1.0, 0.0, 1.0) # Green for strong
            elif signal >= 40:
                color = (1.0, 1.0, 0.0, 1.0) # Yellow for medium
            else:
                color = (1.0, 0.0, 0.0, 1.0) # Red for weak

            # 1. Draw Glowing Signal Dot (Clean UI)
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_LIGHTING)
            glPushMatrix()
            glTranslatef(start_x, 2.5 - (i * y_offset_step), start_z)
            glRotatef(-90, 0, 1, 0)
            glTranslatef(0, 0, 0.05) # Wall offset
            
            # Draw dot
            glPointSize(8.0)
            glBegin(GL_POINTS)
            glColor4f(*color)
            glVertex3f(0, 0, 0)
            glEnd()
            glPopMatrix()

            # 2. Draw SSID Text (To the RIGHT of dot)
            glEnable(GL_TEXTURE_2D)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            
            ssid_lbl = self.get_label(ssid, color)
            glPushMatrix()
            glTranslatef(start_x, 2.5 - (i * y_offset_step), start_z)
            glRotatef(-90, 0, 1, 0) # Rotate to face forward
            glTranslatef(0, 0, 0.05) # Push slightly out from the wall
            
            # Offset text to the right 
            glTranslatef(0.2, -0.05, 0) # Adjust vertically to center with dot
            
            ssid_lbl.draw(0, 0, 0, 0.015)
            glPopMatrix()
        
        glPopAttrib()

    def draw_ghost_wall(self):
        z = -TUNNEL_DEPTH
        glEnable(GL_TEXTURE_2D)
        self.cam_tex.bind()
        glColor4f(*COL_GHOST) # Original color with alpha from constant
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-4, -3, z)
        glTexCoord2f(1, 0); glVertex3f(4, -3, z)
        glTexCoord2f(1, 1); glVertex3f(4, 3, z)
        glTexCoord2f(0, 1); glVertex3f(-4, 3, z)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def draw_intro(self, t):
        if t < 5.0:
            alpha = max(0, min(1, 1.0 - (t / 5.0)))
            
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            gluOrtho2D(0, WIN_WIDTH, WIN_HEIGHT, 0) # Top-left is (0,0)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND) 
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # "CLOSE THE WORLD"
            lbl = self.get_label("CLOSE THE WORLD", (255, 255, 255))
            glColor4f(1, 1, 1, alpha) 
            
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, lbl.texture_id)
            
            x = (WIN_WIDTH - lbl.width) / 2
            y = (WIN_HEIGHT / 2) - 40
            w = lbl.width
            h = lbl.height
            
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(x, y)
            glTexCoord2f(1, 0); glVertex2f(x + w, y)
            glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
            glTexCoord2f(0, 1); glVertex2f(x, y + h)
            glEnd()
            
            # "OPEN THE NEXT"
            lbl2 = self.get_label("OPEN THE NEXT", (255, 255, 255))
            y += 50
            x = (WIN_WIDTH - lbl2.width) / 2
            
            glBindTexture(GL_TEXTURE_2D, lbl2.texture_id)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(x, y)
            glTexCoord2f(1, 0); glVertex2f(x + lbl2.width, y)
            glTexCoord2f(1, 1); glVertex2f(x + lbl2.width, y + lbl2.height)
            glTexCoord2f(0, 1); glVertex2f(x, y + lbl2.height)
            glEnd()
            
            glDisable(GL_TEXTURE_2D)
            
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            
            glEnable(GL_DEPTH_TEST) # Re-enable depth test

    def draw(self):
        self.post_process.begin() # Shader is active
        
        glClearColor(COL_DARK[0], COL_DARK[1], COL_DARK[2], COL_DARK[3]) # Original dark background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Re-enable standard OpenGL states
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_COLOR_MATERIAL) 
        glShadeModel(GL_FLAT) 
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # FPS Camera Calc
        rad_yaw = math.radians(self.cam_yaw)
        rad_pitch = math.radians(self.cam_pitch)
        
        lx = math.sin(rad_yaw) * math.cos(rad_pitch)
        ly = math.sin(rad_pitch)
        lz = -math.cos(rad_yaw) * math.cos(rad_pitch)
        
        gluLookAt(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
                  self.cam_pos[0] + lx, self.cam_pos[1] + ly, self.cam_pos[2] + lz,
                  0, 1, 0)
        
        # The tunnel is drawn first, so other objects appear on top
        self.draw_tunnel()
        self.draw_ghost_wall()
        self.draw_chip()
        self.draw_satellites()
        self.draw_packets()
        self.draw_stats_walls()
        self.rain_left.draw(self)
        self.rain_right.draw(self)
        self.draw_wifi_networks() 
        
        t = time.time() - self.start_time
        self.draw_intro(t) # Draw intro after 3D scene but before final post-process
        self.post_process.end(t, self.glitch_level) # Shader is active
        
        pygame.display.flip()

    def loop(self):
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        self.running = False
                
                self.handle_input_continuous()
                self.update()
                
                self.draw()
                self.clock.tick(60) # Limit to 60 FPS for smoothness
        except Exception as e:
            print(f"Error in loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.listener.stop()
            self.monitor.running = False
            self.wifi_scanner.running = False # Stop WiFi scanner thread
            
            # Explicitly join threads to ensure they finish before pygame quits
            self.listener.join(timeout=5.0) # Give it more time to clean up
            self.monitor.join(timeout=5.0)
            self.wifi_scanner.join(timeout=5.0)

            self.cap.release()
            pygame.quit()

if __name__ == "__main__":
    WiredEngine().loop()