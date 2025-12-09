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
MAX_PACKETS_DISPLAYED = 100 # Limit the number of packets displayed for performance

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

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// RGB to YIQ (NTSC)
vec3 rgb2yiq(vec3 c){
    return vec3(
        0.299*c.r + 0.587*c.g + 0.114*c.b,
        0.596*c.r - 0.275*c.g - 0.321*c.b,
        0.212*c.r - 0.523*c.g + 0.311*c.b
    );
}

// YIQ to RGB
vec3 yiq2rgb(vec3 c){
    return vec3(
        1.0*c.x + 0.956*c.y + 0.621*c.z,
        1.0*c.x - 0.272*c.y - 0.647*c.z,
        1.0*c.x - 1.105*c.y + 1.702*c.z
    );
}

void main() {
    vec2 uv = gl_TexCoord[0].st;
    
    // 1. Pixel Sort / Tear
    float tear = 0.0;
    if (rand(vec2(time, uv.y)) > 0.98) {
        tear = (rand(vec2(time, 0.0)) - 0.5) * glitch_intensity * 0.2;
    }
    uv.x += tear;

    // 2. Chromatic Aberration + VHS Color Bleed (YIQ)
    float r_offset = 0.003 * glitch_intensity;
    float b_offset = -0.003 * glitch_intensity;
    
    vec3 col;
    col.r = texture2D(tex, uv + vec2(r_offset, 0.0)).r;
    col.g = texture2D(tex, uv).g;
    col.b = texture2D(tex, uv + vec2(b_offset, 0.0)).b;
    
    // Convert to YIQ for saturation boost & color drift
    vec3 yiq = rgb2yiq(col);
    yiq.y *= 1.2; 
    yiq.z *= 1.2; 
    col = yiq2rgb(yiq);

    // 3. Simple Bloom (Glow)
    vec4 sum = vec4(0);
    for(int i= -2; i < 2; i++) {
        for(int j= -2; j < 2; j++) {
            sum += texture2D(tex, uv + vec2(i, j)*0.002) * 0.15;
        }
    }
    if (length(sum.rgb) > 1.5) {
       col += sum.rgb * 0.4;
    }

    // 4. Scanlines & Vignette
    float scanline = sin(uv.y * 600.0 + time * 10.0) * 0.05;
    float vig = 1.0 - length(uv - 0.5);
    
    col -= scanline;
    col *= vig;

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
        glColor4f(1, 1, 1, 1) # Textures use their own colors, modulate with white
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
        self.wifi_scanner = wifi_scanner.WifiScanner() # Init WiFi scanner
        self.wifi_scanner.start()

        # Resources
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.cam_tex = WebcamTexture()
        self.labels = {} 
        self.post_process = ShaderPostProcess(WIN_WIDTH, WIN_HEIGHT)
        
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
                if len(self.packets) > MAX_PACKETS_DISPLAYED:
                    self.packets.pop(0) # Remove oldest packet
                self.glitch_level = 0.3 # Reduced initial glitch level
        except queue.Empty: pass
        
        # Update and remove old packets
        # Instead of calling p.update(), handle movement here
        updated_packets = []
        for p in self.packets:
            p.prev_x = p.x # Store previous position for line drawing
            p.prev_y = p.y
            p.prev_z = p.z

            p.angle += p.orbital_speed # Update orbital angle
            
            # Update X and Y based on new angle and orbital radius
            p.x = math.cos(p.angle) * p.orbital_radius
            p.y = math.sin(p.angle) * p.orbital_radius * 0.5 # Flattened Y-axis for elliptical orbit

            # Z-oscillation to add subtle movement instead of static Z
            p.z = p.z + math.sin(time.time() * 2 + p.angle) * 0.005 # Small Z-oscillation

            # Remove packets if they go too far back (or perhaps too far forward in an oscillation)
            # For now, let's just make sure they stay in a visible range
            if p.z < self.cam_pos[2] - 10.0 or p.z > self.cam_pos[2] + 10.0: # Keep within ~20 units of camera Z
                # Re-initialize Z to keep it visible
                p.z = random.uniform(-5.0, -2.0)
            
            updated_packets.append(p)
        self.packets = updated_packets # No filtering by p.update() anymore

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
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        for p in self.packets:
            # Scale line width by packet size
            # Min size ~64 bytes, Max size ~1500 bytes (approx)
            # Map this to a line width from 0.5 to 5.0
            line_width = 0.5 + (p.size / 1500.0) * 4.5
            glLineWidth(line_width) 

            # Draw line behind packet
            glColor4f(*p.color)
            glBegin(GL_LINES)
            glVertex3f(p.prev_x, p.prev_y, p.prev_z)
            glVertex3f(p.x, p.y, p.z)
            glEnd()
            
            payload_lbl = self.get_label(p.payload, COL_CYAN) # Set label color to cyan
            # Removed debug print: print(f"Packet Payload: '{p.payload}'")
            
            # Draw packet text
            glPushMatrix()
            glTranslatef(p.x, p.y, p.z)

            # Orient text to face along the orbital path (roughly outwards or tangent)
            # The +90 degrees is to make the text face outwards from the orbit center
            glRotatef(math.degrees(p.angle) + 90, 0, 1, 0)
            glRotatef(90, 1, 0, 0) # Orient text upright

            # Scale text
            scale_factor = 0.005 # Adjust as needed
            glScalef(scale_factor, scale_factor, scale_factor)

            # Center text texture
            glTranslatef(-payload_lbl.width / 2, -payload_lbl.height / 2, 0)

            payload_lbl.draw(0, 0, 0, scale=1.0) # TextTexture.draw handles its own scaling
            glPopMatrix()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glLineWidth(1.5) # Reset global line width after drawing packets



    def draw_satellites(self):
        for obj in self.active_procs_objs:
            angle = obj['angle'] + math.radians(self.rotation)
            x = math.cos(angle) * obj['radius']
            y = math.sin(angle) * obj['radius'] * 0.5
            z = -8.0 
            
            glPointSize(5) # Original size
            glBegin(GL_POINTS)
            glColor4f(*obj['color']) # Use original color (now 4 elements)
            glVertex3f(x, y, z)
            glEnd()
            
            glBegin(GL_LINES)
            glColor4f(0.0, 0.2, 0.2, 1.0) # Original tether color (now 4 elements)
            glVertex3f(x, y, z)
            glVertex3f(0, 0, -8.0) 
            glEnd()
            
            # Create a combined label for name and port
            process_text = obj['name']
            if obj['port']:
                process_text = f"{obj['name']}:{obj['port']}"
            
            lbl = self.get_label(process_text, (1.0, 1.0, 1.0, 1.0)) # White for names, now 4 elements
            lbl.draw(x + 0.2, y, z, scale=0.02) # Original scale


    def draw_chip(self):
        glPushMatrix()
        glTranslatef(0, 0, -8.0) 
        glRotatef(self.rotation * 2, 0, 1, 0)
        glRotatef(30, 1, 0, 0)
        s = 1.0 + (self.monitor.cpu / 200.0)
        glScalef(s, s, s)
        glColor3f(COL_RED[0], COL_RED[1], COL_RED[2]) # Original color
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
        glPopMatrix()

    def draw_stats_walls(self):
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

    def draw_hex_wall(self):
        start_z = 2.0
        for i in range(15):
            z = start_z + (i * 0.5)
            world_y = -3 + (i / 15.0) * 6
            hex_str = ' '.join(f'{random.randint(0,255):02X}' for _ in range(4))
            lbl = self.get_label(hex_str, COL_HEX) # Use COL_HEX (4 elements now)
            
            glPushMatrix()
            glTranslatef(-3.8, world_y, -10 + z) # Left wall deep
            glRotatef(90, 0, 1, 0)
            glTranslatef(0, 0, 0.05) # Push slightly out from the wall
            lbl.draw(0, 0, 0, 0.02)
            glPopMatrix()

    def draw_wifi_networks(self):
        networks = self.wifi_scanner.networks
        if not networks:
            return

        # Starting position for WiFi network labels
        start_x = 3.9  # Right wall
        start_z = WIFI_Z_OFFSET # Z depth for WiFi entities, defined in constants
        y_offset_step = 0.4 # Vertical spacing between networks
        signal_bar_offset_x = 0.05 # Offset for the signal bars from the text

        for i, (ssid, signal) in enumerate(networks[:10]): # Limit to first 10 for display
            # Determine color based on signal strength
            if signal >= 70:
                color = (0.0, 1.0, 0.0, 1.0) # Green for strong
            elif signal >= 40:
                color = (1.0, 1.0, 0.0, 1.0) # Yellow for medium
            else:
                color = (1.0, 0.0, 0.0, 1.0) # Red for weak

            # Draw SSID
            ssid_lbl = self.get_label(ssid, color)
            glPushMatrix()
            glTranslatef(start_x, 2.5 - (i * y_offset_step), start_z)
            glRotatef(-90, 0, 1, 0) # Rotate to face forward
            glTranslatef(0, 0, 0.05) # Push slightly out from the wall
            ssid_lbl.draw(0, 0, 0, 0.015)
            glPopMatrix()

            # Draw Signal Strength as bars
            glPushMatrix()
            glTranslatef(start_x, 2.5 - (i * y_offset_step) - 0.2, start_z) # Slightly below SSID
            glRotatef(-90, 0, 1, 0)
            glTranslatef(0, 0, 0.05) # Push slightly out from the wall

            num_bars = int(signal / 25) + 1 # 1 to 4 bars
            bar_scale_factor = 0.05 # Increased scale factor for visibility
            
            # Apply scaling for the bars
            glScalef(bar_scale_factor, bar_scale_factor, 1)

            bar_width = 8.0 # Larger absolute values, will be scaled down
            bar_height = 20.0
            bar_spacing = 4.0
            
            for bar_idx in range(num_bars):
                # Calculate position for each bar (these are now absolute, will be scaled)
                bar_x = signal_bar_offset_x + (bar_idx * (bar_width + bar_spacing))
                bar_y = 0.0 # Align with the bottom of the previous text
                
                glColor4f(*color)
                glBegin(GL_QUADS)
                glVertex3f(bar_x, bar_y, 0)
                glVertex3f(bar_x + bar_width, bar_y, 0)
                glVertex3f(bar_x + bar_width, bar_y + bar_height, 0)
                glVertex3f(bar_x, bar_y + bar_height, 0)
                glEnd()
            glPopMatrix()

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
        self.draw_hex_wall() 
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
                pygame.time.wait(10)
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