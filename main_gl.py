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

# --- Constants ---
WIN_WIDTH = 800
WIN_HEIGHT = 600
TUNNEL_DEPTH = 40.0 # Kept for camera clamping logic

# Colors (Normalized OpenGL RGBA)
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
    
    // Original Green/Cyan Tint
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
        glTexCoord2f(1, 1); glVertex2f(1, 1) 
        glTexCoord2f(0, 1); glVertex2f(-1, 1) 
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
        self._current_text = None 
        self._current_color = None 
        self.last_used = time.time() # For Garbage Collection

    def update(self, text, color=(1.0, 1.0, 1.0, 1.0)): 
        if text == self._current_text and color == self._current_color:
            return 

        self._current_text = text
        self._current_color = color

        processed_color = [0, 0, 0, 255] 
        
        if isinstance(color, tuple):
            if len(color) >= 3:
                for i in range(min(len(color), 3)): 
                    if isinstance(color[i], float):
                        processed_color[i] = int(color[i] * 255)
                    else: 
                        processed_color[i] = color[i]
                
                if len(color) == 4: 
                    if isinstance(color[3], float):
                        processed_color[3] = int(color[3] * 255)
                    else: 
                        processed_color[3] = color[3]
            elif len(color) == 1 and isinstance(color[0], (float, int)): 
                val = int(color[0] * 255) if isinstance(color[0], float) else color[0]
                processed_color[0:3] = [val, val, val]
        
        processed_color = tuple(max(0, min(255, c)) for c in processed_color)

        surf = self.font.render(text, True, processed_color)
        new_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        new_surf.fill((0, 0, 0, 0)) 
        new_surf.blit(surf, (0, 0)) 
        
        self.width = new_surf.get_width()
        self.height = new_surf.get_height()

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        data = pygame.image.tostring(new_surf, "RGBA", True) 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

    def draw(self, x, y, z, scale=0.02, yaw=0):
        self.last_used = time.time() # Update usage time
        if self.width == 0 or self.height == 0: return 
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(yaw, 0, 1, 0)
        glScalef(scale, scale, 1)
        glTranslatef(-self.width / 2, -self.height / 2, 0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # RESET COLOR STATE TO WHITE to prevent tinting
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
            if not conns: return COL_CYAN 
            for c in conns:
                if c.status == 'ESTABLISHED' and c.raddr:
                    port = c.raddr.port
                    if port == 80 or port == 8080: return (0.0, 0.0, 1.0, 1.0) 
                    if port == 443: return (0.0, 1.0, 0.0, 1.0) 
                    if port in [21, 22]: return (1.0, 1.0, 0.0, 1.0) 
            return COL_CYAN 
        except:
            return COL_CYAN 

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
                            port = None
                            if c.raddr and c.raddr.port:
                                port = c.raddr.port
                            unique[c.pid] = (p.name(), self.get_port_color(c.pid), port) 
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
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_BLEND) 
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.5) 
        glEnable(GL_COLOR_MATERIAL) 
        glShadeModel(GL_FLAT) 

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
        
        self.data_buffer = collections.deque(maxlen=2000) 
        for _ in range(200): self.data_buffer.append(random.randint(0, 255))

        self.wifi_scanner = wifi_scanner.WifiScanner() 
        self.wifi_scanner.start()

        # Resources
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.cam_tex = WebcamTexture()
        self.labels = {} 
        self.post_process = ShaderPostProcess(WIN_WIDTH, WIN_HEIGHT)
        
        # State
        self.cam_pos = [0.0, 0.0, 5.0] 
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.world_offset_z = 0.0 # Floating Origin Tracking
        self.rotation = 0.0
        self.last_texture_cleanup = time.time() # For Texture GC
        self.packets = [] 
        self.active_procs_objs = [] 
        self.start_time = time.time()
        self.glitch_level = 0.0 
        self.running = True
        self.clock = pygame.time.Clock()

        # --- Entity Initialization ---
        self.entities = []
        self.entities.append(entities.InfiniteTunnel(self))
        self.entities.append(entities.CyberArch(self))
        self.entities.append(entities.GhostWall(self))
        self.entities.append(entities.Hypercore(self))
        self.entities.append(entities.SatelliteSystem(self))
        self.entities.append(entities.PacketSystem(self))
        self.entities.append(entities.CyberCity(self))
        self.entities.append(entities.StatsWall(self))
        self.entities.append(entities.WifiVisualizer(self))
        
        # Matrix Rain (Left/Right)
        self.entities.append(entities.DigitalRain(self, side='left', color=entities.COL_HEX))
        self.entities.append(entities.DigitalRain(self, side='right', color=entities.COL_RED))
        
        # Intro Overlay (Last to be on top if not using post-process depth?)
        self.entities.append(entities.IntroOverlay(self))

        print("Pre-caching textures...")
        for i in range(256):
            h = f"{i:02X}"
            self.get_label(h, entities.COL_HEX)
            self.get_label(h, entities.COL_RED)
        print("Textures cached.")

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
        
        mdx, mdy = pygame.mouse.get_rel()
        self.cam_yaw += mdx * 0.1
        self.cam_pitch -= mdy * 0.1 
        self.cam_pitch = max(-89, min(89, self.cam_pitch))
        
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
        # Allow infinite forward movement (Negative Z), but clamp backward to 10.0
        self.cam_pos[2] = min(10.0, self.cam_pos[2])

    def update(self):
        ok, frame = self.cap.read()
        if ok:
            self.cam_tex.update(frame)

        for entity in self.entities:
            entity.update()

        self.rotation += 0.5
        
        # Floating Origin: Re-center if too far
        if self.cam_pos[2] < -100.0:
            shift = self.cam_pos[2]
            self.cam_pos[2] -= shift # Should become ~0
            self.world_offset_z += shift
        
        # Periodic Texture Cleanup (Garbage Collection)
        # Run every 10 seconds
        if time.time() - self.last_texture_cleanup > 10.0:
            self.last_texture_cleanup = time.time()
            # Delete textures unused for > 60 seconds
            threshold = time.time() - 60.0
            
            keys_to_delete = []
            for key, tex in self.labels.items():
                if tex.last_used < threshold:
                    keys_to_delete.append(key)
            
            if keys_to_delete:
                # print(f"GC: Cleaning {len(keys_to_delete)} old textures.")
                for key in keys_to_delete:
                    try:
                        glDeleteTextures([self.labels[key].texture_id])
                    except: pass
                    del self.labels[key]

    def draw(self):
        self.post_process.begin() 
        
        glClearColor(COL_DARK[0], COL_DARK[1], COL_DARK[2], COL_DARK[3]) 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_COLOR_MATERIAL) 
        glShadeModel(GL_FLAT) 
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        rad_yaw = math.radians(self.cam_yaw)
        rad_pitch = math.radians(self.cam_pitch)
        
        lx = math.sin(rad_yaw) * math.cos(rad_pitch)
        ly = math.sin(rad_pitch)
        lz = -math.cos(rad_yaw) * math.cos(rad_pitch)
        
        gluLookAt(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
                  self.cam_pos[0] + lx, self.cam_pos[1] + ly, self.cam_pos[2] + lz,
                  0, 1, 0)
        
        # Draw all entities
        for entity in self.entities:
            entity.draw()
        
        t = time.time() - self.start_time
        self.post_process.end(t, self.glitch_level) 
        
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
                self.clock.tick(60) 
        except Exception as e:
            print(f"Error in loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.listener.stop()
            self.monitor.running = False
            self.wifi_scanner.running = False 
            
            self.listener.join(timeout=5.0) 
            self.monitor.join(timeout=5.0)
            self.wifi_scanner.join(timeout=5.0)

            self.cap.release()
            pygame.quit()

if __name__ == "__main__":
    WiredEngine().loop()