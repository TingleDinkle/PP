import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import random
import time
import collections

# --- Constants ---
TUNNEL_DEPTH = 40.0
WIFI_Z_OFFSET = -35.0
MAX_PACKETS_DISPLAYED = 40

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

class GameObject:
    def __init__(self, engine):
        self.engine = engine

    def update(self):
        pass

    def draw(self):
        pass

class InfiniteTunnel(GameObject):
    def draw(self):
        cam_z = self.engine.cam_pos[2]
        
        # Determine visible range based on camera Z
        # We look down -Z. We want to see from slightly behind us (+Z relative) to far in front (-Z relative)
        draw_dist = 60.0 
        start_z = cam_z + 10.0
        end_z = cam_z - draw_dist
        
        # Snap to grid size (2.0)
        grid_step = 2.0
        
        # Calculate start/end indices for the loop
        # We iterate DOWNWARDS (from start_z to end_z) because z decreases as we go forward
        start_idx = int(math.ceil(start_z / grid_step))
        end_idx = int(math.floor(end_z / grid_step))

        if self.engine.monitor.disk_write:
            glColor3f(COL_WHITE[0], COL_WHITE[1], COL_WHITE[2])
        else:
            glColor3f(COL_GRID[0], COL_GRID[1], COL_GRID[2])
        
        # Draw "Ribs" (Square loops)
        for i in range(start_idx, end_idx - 1, -1):
            z = i * grid_step
            glBegin(GL_LINE_LOOP)
            glVertex3f(-4, -3, z)
            glVertex3f(4, -3, z)
            glVertex3f(4, 3, z)
            glVertex3f(-4, 3, z)
            glEnd()
            
        # Draw Longitudinal Lines (The long rails)
        # We just draw them long enough to cover the view
        glBegin(GL_LINES)
        for x, y in [(-4,-3), (4,-3), (4,3), (-4,3)]:
            glVertex3f(x, y, start_z)
            glVertex3f(x, y, end_z)
        glEnd()

class CyberArch(GameObject):
    def draw(self):
        cam_z = self.engine.cam_pos[2]
        spacing = 50.0 # Arch every 50 units
        
        # Find the closest "slot" in front of the camera
        # cam_z is e.g. -120. We want arches at -150, -200...
        # Also maybe one behind at -100.
        
        current_idx = int(cam_z / spacing)
        
        # Draw a few arches around the player
        draw_range = 3 # Draw 3 arches ahead/behind
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glLineWidth(2.0)
        
        for i in range(current_idx - draw_range, current_idx + 1):
            z_pos = i * spacing
            
            # Simple distance fade (optional, but looks nice)
            dist = abs(cam_z - z_pos)
            if dist > 100: continue # optimization
            
            alpha = max(0, min(1, 1.0 - (dist / 80.0)))
            if alpha <= 0: continue

            glPushMatrix()
            glTranslatef(0, 0, z_pos)
            
            # Arch Color - cycle through colors based on index
            colors = [COL_CYAN, COL_RED, COL_YELLOW, (0.5, 0.0, 1.0, 1.0)]
            col = colors[abs(i) % len(colors)]
            glColor4f(col[0], col[1], col[2], alpha)
            
            # Draw Arch Shape (Half Hexagon)
            glBegin(GL_LINE_STRIP)
            glVertex3f(-4, -3, 0)
            glVertex3f(-4, 2, 0) # Up left
            glVertex3f(-2, 3, 0) # Angle in
            glVertex3f(2, 3, 0)  # Top flat
            glVertex3f(4, 2, 0)  # Angle out
            glVertex3f(4, -3, 0) # Down right
            glEnd()
            
            # Glowing Core of the Arch
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glVertex3f(0, 3, 0)
            glEnd()
            
            glPopMatrix()
            
        glPopAttrib()

class GhostWall(GameObject):
    def draw(self):
        z = -TUNNEL_DEPTH
        glEnable(GL_TEXTURE_2D)
        self.engine.cam_tex.bind()
        glColor4f(*COL_GHOST)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-4, -3, z)
        glTexCoord2f(1, 0); glVertex3f(4, -3, z)
        glTexCoord2f(1, 1); glVertex3f(4, 3, z)
        glTexCoord2f(0, 1); glVertex3f(-4, 3, z)
        glEnd()
        glDisable(GL_TEXTURE_2D)

class Hypercore(GameObject):
    def draw(self):
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING) 
        glDisable(GL_DEPTH_TEST) # ALWAYS VISIBLE
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive glow
        
        glPushMatrix()
        glTranslatef(0, 0, -15.0)
        
        # 1. Nucleus
        pulse = 1.0 + math.sin(time.time() * 3.0) * 0.2
        glColor4f(1.0, 0.1, 0.1, 1.0)
        glPushMatrix()
        glScalef(pulse, pulse, pulse)
        sphere = gluNewQuadric()
        gluSphere(sphere, 0.8, 16, 16)
        glPopMatrix()
        
        # 2. Inner Cube
        glPushMatrix()
        glRotatef(self.engine.rotation * 3, 1, 1, 0)
        glColor4f(1.0, 0.6, 0.0, 0.8)
        glLineWidth(3.0)
        s = 2.0
        glScalef(s, s, s)
        self._draw_wire_cube()
        glPopMatrix()
        
        # 3. Octahedron
        glPushMatrix()
        glRotatef(-self.engine.rotation * 2, 0, 1, 1)
        glColor4f(0.0, 1.0, 1.0, 0.6)
        glLineWidth(2.0)
        s = 3.5
        glScalef(s, s, s)
        self._draw_wire_octahedron()
        glPopMatrix()
        
        # 4. Rings
        for i in range(4):
            glPushMatrix()
            glRotatef(self.engine.rotation * (1 + i*0.3), (i==0), (i==1), (i==2))
            glColor4f(0.2, 0.4, 1.0, 0.5)
            gluDisk(gluNewQuadric(), 4.0 + i*0.4, 4.1 + i*0.4, 64, 1)
            glPopMatrix()
            
        # 5. Data Rings
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        chars = "0123456789ABCDEF"
        num_chars = 16
        ring_radius = 5.5
        
        glPushMatrix()
        glRotatef(self.engine.rotation * 10, 0, 1, 0)
        
        for i in range(num_chars):
            angle = (i / num_chars) * 2 * math.pi
            x = math.cos(angle) * ring_radius
            z = math.sin(angle) * ring_radius
            
            char_idx = int(time.time() * 10 + i) % 16
            char = chars[char_idx]
            
            lbl = self.engine.get_label(char, COL_WHITE)
            
            glPushMatrix()
            glTranslatef(x, 0, z)
            glRotatef(-math.degrees(angle) - 90, 0, 1, 0) 
            
            lbl.draw(0, 0, 0, 0.08)
            glPopMatrix()
            
        glPopMatrix()
        glPopMatrix()
        glPopAttrib()

    def _draw_wire_cube(self):
        glBegin(GL_LINES)
        for x in [-1, 1]:
            for y in [-1, 1]:
                glVertex3f(x, y, -1); glVertex3f(x, y, 1)
                glVertex3f(x, -1, y); glVertex3f(x, 1, y)
                glVertex3f(-1, x, y); glVertex3f(1, x, y)
        glEnd()

    def _draw_wire_octahedron(self):
        glBegin(GL_LINES)
        # Top
        glVertex3f(0,1,0); glVertex3f(1,0,0)
        glVertex3f(0,1,0); glVertex3f(-1,0,0)
        glVertex3f(0,1,0); glVertex3f(0,0,1)
        glVertex3f(0,1,0); glVertex3f(0,0,-1)
        # Bottom
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

class SatelliteSystem(GameObject):
    def update(self):
        current_procs = self.engine.monitor.processes
        if len(self.engine.active_procs_objs) != len(current_procs):
            self.engine.active_procs_objs = []
            for i, (pid, name, color, port) in enumerate(current_procs):
                self.engine.active_procs_objs.append({
                    'name': name,
                    'pid': pid,
                    'angle': (i / len(current_procs)) * 2 * math.pi,
                    'radius': 3.0,
                    'color': color,
                    'port': port
                })

    def draw(self):
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING) 
        glEnable(GL_BLEND)
        glLineWidth(1.5) 
        
        center_z = -15.0
        radius = 8.0
        
        glPushMatrix()
        glTranslatef(0, 0, center_z)
        
        glRotatef(-self.engine.rotation * 0.5, 0, 0, 1) 
        
        glColor4f(0.0, 0.5, 0.5, 0.3) 
        glBegin(GL_LINE_LOOP)
        for i in range(80):
            theta = 2.0 * math.pi * i / 80.0
            glVertex3f(math.cos(theta) * radius, math.sin(theta) * radius, 0)
        glEnd()
        
        orbit_speed = 0.2
        t = time.time()
        
        for i, obj in enumerate(self.engine.active_procs_objs):
            base_angle = (i / max(1, len(self.engine.active_procs_objs))) * 2 * math.pi
            angle = base_angle + (t * orbit_speed)
            
            px = math.cos(angle) * radius
            py = math.sin(angle) * radius
            
            glPushMatrix()
            glTranslatef(px, py, 0)
            glRotatef(self.engine.rotation * 5, 0, 0, 1)
            glScalef(0.2, 0.2, 0.2)
            
            glColor4f(*obj['color'])
            glBegin(GL_QUADS)
            glVertex3f(0, 1, 0); glVertex3f(1, 0, 0)
            glVertex3f(0, -1, 0); glVertex3f(-1, 0, 0)
            glEnd()
            glPopMatrix()
            
            glBegin(GL_LINES)
            glColor4f(obj['color'][0], obj['color'][1], obj['color'][2], 0.1)
            glVertex3f(px, py, 0)
            glVertex3f(0, 0, 0)
            glEnd()
            
            glPushMatrix()
            glTranslatef(px, py + 0.5, 0)
            glRotatef(self.engine.rotation * 0.5, 0, 0, 1)
            glScalef(0.015, 0.015, 0.015)
            
            process_text = obj['name']
            lbl = self.engine.get_label(process_text, COL_WHITE)
            glTranslatef(-lbl.width/2, 0, 0)
            
            glEnable(GL_TEXTURE_2D)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            lbl.draw(0, 0, 0, 1.0)
            glDisable(GL_TEXTURE_2D)
            
            glPopMatrix()

        glPopMatrix()
        glPopAttrib()

class PacketSystem(GameObject):
    def update(self):
        # Consume queue
        try:
            while True:
                p = self.engine.packet_queue.get_nowait()
                self.engine.packets.append(p)
                if p.payload:
                    for char in p.payload:
                        self.engine.data_buffer.append(ord(char) % 256)
                
                if len(self.engine.packets) > MAX_PACKETS_DISPLAYED:
                    self.engine.packets.pop(0)
                self.engine.glitch_level = 0.3
        except Exception: pass # Queue empty or other error
        
        # Update packets
        updated_packets = []
        t = time.time()
        
        for i, p in enumerate(self.engine.packets):
            if not hasattr(p, 'history'):
                p.history = collections.deque(maxlen=20)
                p.base_z_offset = random.uniform(0, TUNNEL_DEPTH)
                if p.protocol == 'TCP':
                    p.lane_x = -2.5
                elif p.protocol == 'UDP':
                    p.lane_x = 2.5
                else:
                    p.lane_x = 0.0
                p.lane_y = random.uniform(-1.5, 1.5)

            p.prev_x = p.x 
            p.prev_y = p.y
            p.prev_z = p.z
            p.history.append((p.x, p.y, p.z))

            scroll_speed = 2.0
            range_len = TUNNEL_DEPTH + 10.0
            
            z_val = (p.base_z_offset + t * scroll_speed) % range_len
            p.z = -TUNNEL_DEPTH + z_val
            
            p.x = p.lane_x
            p.y = p.lane_y
            
            updated_packets.append(p)
        self.engine.packets = updated_packets
        self.engine.glitch_level *= 0.9

    def draw(self):
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glLineWidth(3.0) 
        
        for p in self.engine.packets:
            if not hasattr(p, 'history') or len(p.history) < 2: continue
            
            glBegin(GL_LINE_STRIP)
            for i, (hx, hy, hz) in enumerate(p.history):
                alpha = (i / len(p.history)) * 0.5
                glColor4f(p.color[0], p.color[1], p.color[2], alpha)
                glVertex3f(hx, hy, hz)
            glEnd()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        for p in self.engine.packets:
            label_col = p.color
            display_text = p.payload[:20] + "..." if len(p.payload) > 20 else p.payload
            lbl = self.engine.get_label(f"[{p.protocol}] {display_text}", label_col)
            
            glPushMatrix()
            glTranslatef(p.x, p.y + 0.3, p.z)
            glRotatef(-self.engine.cam_yaw, 0, 1, 0)
            glRotatef(-self.engine.cam_pitch, 1, 0, 0)
            
            scale = 0.008
            glScalef(scale, scale, scale)
            glTranslatef(-lbl.width/2, 0, 0)
            
            lbl.draw(0, 0, 0, 1.0)
            glPopMatrix()
            
        glPopAttrib()

class DigitalRain(GameObject):
    def __init__(self, engine, side='left', color=COL_HEX):
        super().__init__(engine)
        self.side = side
        self.color = color
        self.columns = 20
        self.drops = []
        for i in range(self.columns):
            self.drops.append({
                'col_idx': i,
                'y': random.uniform(-10, 10),
                'speed': random.uniform(0.1, 0.3),
                'chars': [self._get_char() for _ in range(random.randint(5, 15))]
            })

    def _get_char(self):
        if self.engine.data_buffer:
            val = self.engine.data_buffer[random.randint(0, len(self.engine.data_buffer)-1)]
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
            
            if random.random() < 0.1:
                drop['chars'][random.randint(0, len(drop['chars'])-1)] = self._get_char()

    def draw(self):
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_TEXTURE_BIT)
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        batches = {}
        
        for drop in self.drops:
            for i, char in enumerate(drop['chars']):
                z = -10 + (drop['col_idx'] * 0.8) 
                y = drop['y'] + (i * 0.4)
                
                if y > 5 or y < -5: continue
                
                if self.side == 'left':
                    x = -3.8
                else: 
                    x = 3.8
                
                alpha = 1.0 - (i / len(drop['chars']))
                
                if char not in batches:
                    batches[char] = []
                batches[char].append((x, y, z, alpha))

        glPushMatrix()
        if self.side == 'left':
            glTranslatef(-3.8, 0, 0)
            glRotatef(90, 0, 1, 0)
        else:
            glTranslatef(3.8, 0, 0)
            glRotatef(-90, 0, 1, 0)
        
        glTranslatef(0, 0, 0.1)
        
        scale = 0.02

        for char, instances in batches.items():
            tex = self.engine.get_label(char, self.color)
            if tex.width == 0: continue
            
            glBindTexture(GL_TEXTURE_2D, tex.texture_id)
            w = tex.width * scale
            h = tex.height * scale
            off_x = -w / 2
            off_y = -h / 2
            
            glBegin(GL_QUADS)
            for (gx, gy, gz, alpha) in instances:
                glColor4f(1, 1, 1, alpha)
                lx = -gz if self.side == 'left' else gz
                ly = gy
                
                glTexCoord2f(0, 0); glVertex3f(lx + off_x, ly + off_y, 0)
                glTexCoord2f(1, 0); glVertex3f(lx + off_x + w, ly + off_y, 0)
                glTexCoord2f(1, 1); glVertex3f(lx + off_x + w, ly + off_y + h, 0)
                glTexCoord2f(0, 1); glVertex3f(lx + off_x, ly + off_y + h, 0)
            glEnd()
            
        glPopMatrix()
        glPopAttrib()

class StatsWall(GameObject):
    def draw(self):
        glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        ram_lbl = self.engine.get_label(f"RAM: {self.engine.monitor.ram}%", (0.0, 1.0, 1.0, 1.0))
        glPushMatrix()
        glTranslatef(-3.9, 0, -5)
        glRotatef(90, 0, 1, 0) 
        glTranslatef(0, 0, 0.05)
        ram_lbl.draw(0, 0, 0, 0.03)
        glPopMatrix()
        
        cpu_lbl = self.engine.get_label(f"CPU: {self.engine.monitor.cpu}%", (1.0, 0.2, 0.2, 1.0))
        glPushMatrix()
        glTranslatef(3.9, 0, -5)
        glRotatef(-90, 0, 1, 0) 
        glTranslatef(0, 0, 0.05)
        cpu_lbl.draw(0, 0, 0, 0.03)
        glPopMatrix()
        
        glPopAttrib()

class WifiVisualizer(GameObject):
    def draw(self):
        networks = self.engine.wifi_scanner.networks
        if not networks:
            return
            
        glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        
        start_x = 3.9
        start_z = WIFI_Z_OFFSET
        y_offset_step = 0.4

        for i, (ssid, signal) in enumerate(networks[:10]):
            if signal >= 70:
                color = (0.0, 1.0, 0.0, 1.0)
            elif signal >= 40:
                color = (1.0, 1.0, 0.0, 1.0)
            else:
                color = (1.0, 0.0, 0.0, 1.0)

            glDisable(GL_TEXTURE_2D)
            glDisable(GL_LIGHTING)
            glPushMatrix()
            glTranslatef(start_x, 2.5 - (i * y_offset_step), start_z)
            glRotatef(-90, 0, 1, 0)
            glTranslatef(0, 0, 0.05)
            
            glPointSize(8.0)
            glBegin(GL_POINTS)
            glColor4f(*color)
            glVertex3f(0, 0, 0)
            glEnd()
            glPopMatrix()

            glEnable(GL_TEXTURE_2D)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            
            ssid_lbl = self.engine.get_label(ssid, color)
            glPushMatrix()
            glTranslatef(start_x, 2.5 - (i * y_offset_step), start_z)
            glRotatef(-90, 0, 1, 0)
            glTranslatef(0, 0, 0.05)
            glTranslatef(0.2, -0.05, 0)
            
            ssid_lbl.draw(0, 0, 0, 0.015)
            glPopMatrix()
        
        glPopAttrib()

class IntroOverlay(GameObject):
    def draw(self):
        t = time.time() - self.engine.start_time
        if t < 5.0:
            alpha = max(0, min(1, 1.0 - (t / 5.0)))
            
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            gluOrtho2D(0, self.engine.screen.get_width(), self.engine.screen.get_height(), 0)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND) 
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            lbl = self.engine.get_label("CLOSE THE WORLD", (255, 255, 255))
            glColor4f(1, 1, 1, alpha) 
            
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, lbl.texture_id)
            
            win_width = self.engine.screen.get_width()
            win_height = self.engine.screen.get_height()
            
            x = (win_width - lbl.width) / 2
            y = (win_height / 2) - 40
            w = lbl.width
            h = lbl.height
            
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(x, y)
            glTexCoord2f(1, 0); glVertex2f(x + w, y)
            glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
            glTexCoord2f(0, 1); glVertex2f(x, y + h)
            glEnd()
            
            lbl2 = self.engine.get_label("OPEN THE NEXT", (255, 255, 255))
            y += 50
            x = (win_width - lbl2.width) / 2
            
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
            
            glEnable(GL_DEPTH_TEST)
