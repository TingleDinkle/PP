import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
import numpy as np
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

class Mesh:
    """Helper to manage VBOs for static or semi-static geometry."""
    def __init__(self, vertices, mode=GL_LINES):
        self.vertex_count = len(vertices)
        self.mode = mode
        # Convert to float32 numpy array
        self.data = np.array(vertices, dtype=np.float32)
        self.vbo = vbo.VBO(self.data)

    def draw(self):
        if self.vertex_count == 0: return
        try:
            self.vbo.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.vbo)
            glDrawArrays(self.mode, 0, self.vertex_count)
            glDisableClientState(GL_VERTEX_ARRAY)
            self.vbo.unbind()
        except Exception as e:
            print(f"VBO Draw Error: {e}")

class DynamicMesh:
    """Helper to manage dynamic VBOs that change every frame."""
    def __init__(self, mode=GL_LINES):
        self.mode = mode
        self.vbo_pos = vbo.VBO(np.array([], dtype=np.float32))
        self.vbo_col = vbo.VBO(np.array([], dtype=np.float32))
        self.vertex_count = 0

    def update(self, vertices, colors):
        if not vertices:
            self.vertex_count = 0
            return
        
        pos_data = np.array(vertices, dtype=np.float32)
        col_data = np.array(colors, dtype=np.float32)
        
        self.vertex_count = len(pos_data) // 3 # 3 floats per vertex
        
        self.vbo_pos.set_array(pos_data)
        self.vbo_col.set_array(col_data)

    def draw(self):
        if self.vertex_count == 0: return
        try:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            self.vbo_pos.bind()
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            self.vbo_col.bind()
            glColorPointer(4, GL_FLOAT, 0, None)
            
            glDrawArrays(self.mode, 0, self.vertex_count)
            
            self.vbo_col.unbind()
            self.vbo_pos.unbind()
            
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        except Exception as e:
            print(f"DynamicVBO Draw Error: {e}")

class InfiniteTunnel(GameObject):
    def __init__(self, engine):
        super().__init__(engine)
        self.segment_length = 20.0
        self.grid_step = 2.0
        self.mesh = self._generate_segment_mesh()

    def _generate_segment_mesh(self):
        verts = []
        steps = int(self.segment_length / self.grid_step)
        for i in range(steps + 1):
            z = -i * self.grid_step
            verts.append((-4, 3, z)); verts.append((4, 3, z))
            verts.append((4, 3, z)); verts.append((4, -3, z))
            verts.append((4, -3, z)); verts.append((-4, -3, z))
            verts.append((-4, -3, z)); verts.append((-4, 3, z))

        corners = [(-4, 3), (4, 3), (4, -3), (-4, -3)]
        for x, y in corners:
            verts.append((x, y, 0))
            verts.append((x, y, -self.segment_length))
            
        return Mesh(verts, GL_LINES)

    def draw(self):
        global_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
        
        # Zone Logic
        zone_cycle = 1000.0
        zone_val = abs(global_z) % zone_cycle
        
        base_color = COL_GRID
        if zone_val < 300: # Standard Blue
            base_color = COL_GRID
        elif zone_val < 600: # Purple
            base_color = (0.6, 0.0, 0.8, 0.5)
        elif zone_val < 800: # Green
            base_color = (0.0, 0.8, 0.2, 0.5)
        else: # Red
            base_color = (0.8, 0.1, 0.1, 0.5)

        if self.engine.monitor.disk_write:
            glColor3f(COL_WHITE[0], COL_WHITE[1], COL_WHITE[2])
        else:
            glColor4f(base_color[0], base_color[1], base_color[2], base_color[3])
            
        cam_z = self.engine.cam_pos[2]
        current_seg_idx = math.floor(-cam_z / self.segment_length)
        
        for i in range(-1, 6):
            seg_idx = current_seg_idx + i
            z_pos = -(seg_idx * self.segment_length)
            glPushMatrix()
            glTranslatef(0, 0, z_pos)
            self.mesh.draw()
            glPopMatrix()

class CyberArch(GameObject):
    def __init__(self, engine):
        super().__init__(engine)
        self.mesh = self._generate_arch_mesh()
        self.spacing = 50.0

    def _generate_arch_mesh(self):
        verts = []
        verts.append((-4, -3, 0)); verts.append((-4, 2, 0))
        verts.append((-4, 2, 0)); verts.append((-2, 3, 0))
        verts.append((-2, 3, 0)); verts.append((2, 3, 0))
        verts.append((2, 3, 0)); verts.append((4, 2, 0))
        verts.append((4, 2, 0)); verts.append((4, -3, 0))
        return Mesh(verts, GL_LINES)

    def draw(self):
        global_cam_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
        current_idx = int(global_cam_z / self.spacing)
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glLineWidth(2.0)
        
        for i in range(current_idx - 2, current_idx + 4):
            arch_global_z = i * self.spacing
            arch_local_z = arch_global_z - getattr(self.engine, 'world_offset_z', 0.0)
            
            dist = abs(self.engine.cam_pos[2] - arch_local_z)
            if dist > 120: continue 
            
            alpha = max(0, min(1, 1.0 - (dist / 100.0)))
            if alpha <= 0: continue

            colors = [COL_CYAN, COL_RED, COL_YELLOW, (0.5, 0.0, 1.0, 1.0)]
            col = colors[abs(i) % len(colors)]
            glColor4f(col[0], col[1], col[2], alpha)

            glPushMatrix()
            glTranslatef(0, 0, arch_local_z)
            self.mesh.draw()
            
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glVertex3f(0, 3, 0)
            glEnd()
            glPopMatrix()
        glPopAttrib()

class GhostWall(GameObject):
    def draw(self):
        z = self.engine.cam_pos[2] + 10.0
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
        global_z = -15.0
        local_z = global_z - getattr(self.engine, 'world_offset_z', 0.0)
        
        if local_z > self.engine.cam_pos[2] + 20 or local_z < self.engine.cam_pos[2] - 100:
            return

        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING) 
        glDisable(GL_DEPTH_TEST) 
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) 
        
        glPushMatrix()
        glTranslatef(0, 0, local_z)
        
        pulse = 1.0 + math.sin(time.time() * 3.0) * 0.2
        glColor4f(1.0, 0.1, 0.1, 1.0)
        glPushMatrix()
        glScalef(pulse, pulse, pulse)
        sphere = gluNewQuadric()
        gluSphere(sphere, 0.8, 16, 16)
        glPopMatrix()
        
        glPushMatrix()
        glRotatef(self.engine.rotation * 3, 1, 1, 0)
        glColor4f(1.0, 0.6, 0.0, 0.8)
        glLineWidth(3.0)
        s = 2.0
        glScalef(s, s, s)
        self._draw_wire_cube()
        glPopMatrix()
        
        glPushMatrix()
        glRotatef(-self.engine.rotation * 2, 0, 1, 1)
        glColor4f(0.0, 1.0, 1.0, 0.6)
        glLineWidth(2.0)
        s = 3.5
        glScalef(s, s, s)
        self._draw_wire_octahedron()
        glPopMatrix()
        
        for i in range(4):
            glPushMatrix()
            glRotatef(self.engine.rotation * (1 + i*0.3), (i==0), (i==1), (i==2))
            glColor4f(0.2, 0.4, 1.0, 0.5)
            gluDisk(gluNewQuadric(), 4.0 + i*0.4, 4.1 + i*0.4, 64, 1)
            glPopMatrix()
            
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
        glVertex3f(0,1,0); glVertex3f(1,0,0)
        glVertex3f(0,1,0); glVertex3f(-1,0,0)
        glVertex3f(0,1,0); glVertex3f(0,0,1)
        glVertex3f(0,1,0); glVertex3f(0,0,-1)
        glVertex3f(0,-1,0); glVertex3f(1,0,0)
        glVertex3f(0,-1,0); glVertex3f(-1,0,0)
        glVertex3f(0,-1,0); glVertex3f(0,0,1)
        glVertex3f(0,-1,0); glVertex3f(0,0,-1)
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
        global_z = -15.0
        local_z = global_z - getattr(self.engine, 'world_offset_z', 0.0)
        
        if local_z > self.engine.cam_pos[2] + 20 or local_z < self.engine.cam_pos[2] - 100:
            return

        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING) 
        glEnable(GL_BLEND)
        glLineWidth(1.5) 
        
        radius = 8.0
        
        glPushMatrix()
        glTranslatef(0, 0, local_z)
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
            glVertex3f(px, py, 0); glVertex3f(0, 0, 0)
            glEnd()
            
            glPushMatrix()
            glTranslatef(px, py + 0.5, 0)
            glRotatef(self.engine.rotation * 0.5, 0, 0, 1)
            glScalef(0.015, 0.015, 0.015)
            lbl = self.engine.get_label(obj['name'], COL_WHITE)
            glTranslatef(-lbl.width/2, 0, 0)
            glEnable(GL_TEXTURE_2D)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            lbl.draw(0, 0, 0, 1.0)
            glDisable(GL_TEXTURE_2D)
            glPopMatrix()
        glPopMatrix()
        glPopAttrib()

class PacketSystem(GameObject):
    def __init__(self, engine):
        super().__init__(engine)
        self.trail_vbo = DynamicMesh(GL_LINES)

    def update(self):
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
        except Exception: pass
        
        updated_packets = []
        
        for i, p in enumerate(self.engine.packets):
            if not hasattr(p, 'history'):
                p.history = collections.deque(maxlen=20)
                current_global_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
                p.base_global_z = current_global_z + random.uniform(20, -20) 
                if p.protocol == 'TCP': p.lane_x = -2.5
                elif p.protocol == 'UDP': p.lane_x = 2.5
                else: p.lane_x = 0.0
                p.lane_y = random.uniform(-1.5, 1.5)

            if not hasattr(p, 'global_z'):
                p.global_z = p.base_global_z
            
            p.global_z -= 0.55 # Move packet forward (drifts slowly past player)
            
            p.x = p.lane_x
            p.y = p.lane_y
            p.history.append((p.x, p.y, p.global_z))
            updated_packets.append(p)
        self.engine.packets = updated_packets
        self.engine.glitch_level *= 0.9

    def draw(self):
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive Glow
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glLineWidth(3.0) # Original width 
        
        world_offset = getattr(self.engine, 'world_offset_z', 0.0)
        
        # Prepare VBO data for all packet trails
        pos_data = []
        col_data = []
        
        for p in self.engine.packets:
            if not hasattr(p, 'history') or len(p.history) < 2: continue
            
            # Convert deque to list for indexing
            pts = list(p.history)
            for i in range(len(pts) - 1):
                # Segment start
                hx1, hy1, hz1 = pts[i]
                lz1 = hz1 - world_offset
                # Segment end
                hx2, hy2, hz2 = pts[i+1]
                lz2 = hz2 - world_offset
                
                # Original Alpha fade
                alpha = (i / len(pts)) * 0.5
                
                # Append pos [x, y, z]
                pos_data.extend([hx1, hy1, lz1])
                pos_data.extend([hx2, hy2, lz2])
                
                # Append color [r, g, b, a]
                col_data.extend([p.color[0], p.color[1], p.color[2], alpha])
                col_data.extend([p.color[0], p.color[1], p.color[2], alpha])

        # Upload and Draw Trails
        if pos_data:
            self.trail_vbo.update(pos_data, col_data)
            self.trail_vbo.draw()

        # Draw Labels (Still Immediate Mode for unique text)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        for p in self.engine.packets:
            if not hasattr(p, 'global_z'): continue
            local_z = p.global_z - world_offset
            if local_z > self.engine.cam_pos[2] + 10 or local_z < self.engine.cam_pos[2] - 100: continue
            
            label_col = p.color
            display_text = p.payload[:20] + "..." if len(p.payload) > 20 else p.payload
            lbl = self.engine.get_label(f"[{p.protocol}] {display_text}", label_col)
            
            glPushMatrix()
            glTranslatef(p.x, p.y + 0.3, local_z)
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
            
            if random.random() < 0.02:
                drop['chars'][random.randint(0, len(drop['chars'])-1)] = self._get_char()

    def draw(self):
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_TEXTURE_BIT)
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        scale = 0.02
        global_cam_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
        segment_length = 20.0
        start_seg = int(global_cam_z / segment_length)
        
        # Batch by character to minimize texture binding
        # We collect vertices for ALL segments into these batches
        batches = collections.defaultdict(list)
        
        for s in range(start_seg - 4, start_seg + 2):
            seg_global_z = s * segment_length
            seg_local_z = seg_global_z - getattr(self.engine, 'world_offset_z', 0.0)
            
            # Pre-calculate base transform for this wall side
            # Left Wall: x = -3.8. Rot 90 (X+ -> Z-).
            # Right Wall: x = 3.8. Rot -90 (X+ -> Z+).
            
            # Since we are building a dynamic mesh, we must manually transform 
            # the quad vertices (0,0) -> (w,h) into World/Camera space.
            
            # Wall Local Basis Vectors (Camera Space)
            if self.side == 'left':
                # Pos: (-3.8, 0, 0.1) relative to segment origin
                # Rot 90 around Y: X -> Z, Z -> -X
                # Quad X (Width) aligns with -Z (into screen)
                # Quad Y (Height) aligns with Y (up)
                wall_origin_x = -3.8
                wall_origin_z = 0.1 + seg_local_z
                ux, uy, uz = (0, 0, -1) # Right vector of quad in cam space
                vx, vy, vz = (0, 1, 0)  # Up vector of quad in cam space
            else:
                # Pos: (3.8, 0, 0.1)
                # Rot -90 around Y: X -> -Z, Z -> X
                # Quad X (Width) aligns with Z (out of screen)
                wall_origin_x = 3.8
                wall_origin_z = 0.1 + seg_local_z
                ux, uy, uz = (0, 0, 1)
                vx, vy, vz = (0, 1, 0)

            for drop in self.drops:
                # Drop Base Position on Wall Surface
                # Drop Z (relative to segment) maps to Quad X axis
                drop_z_rel = (drop['col_idx'] * 0.8) - 10 
                # Drop Y maps to Quad Y axis
                
                # Position of drop origin in Camera Space:
                # Origin + (DropZ * U) + (DropY * V)
                
                # Wait, 'drop_z_rel' is distance along the wall.
                # In our immediate mode logic:
                # glTranslatef(wall_origin); glRotate(); translate(drop_z_rel, drop_y)
                # So yes, drop_z_rel moves along the Quad's X axis.
                
                base_x = wall_origin_x + (drop_z_rel * ux) 
                base_y = drop['y'] 
                base_z = wall_origin_z + (drop_z_rel * uz)
                
                for i, char in enumerate(drop['chars']):
                    # Per-char offset (Vertical/Y)
                    char_y_rel = i * 0.4
                    
                    # Final Char Position (Top-Left of quad approx)
                    cx = base_x + (char_y_rel * vx) # Actually Y aligns with V
                    cy = base_y + char_y_rel 
                    cz = base_z 
                    
                    # Check visibility (Vertical clip)
                    if cy > 5 or cy < -5: continue
                    
                    alpha = 1.0 - (i / len(drop['chars']))
                    
                    # Get Texture size to build quad
                    tex = self.engine.get_label(char, self.color)
                    if tex.width == 0: continue
                    
                    w = tex.width * scale
                    h = tex.height * scale
                    
                    # Quad centered? Previous logic:
                    # off_x = -w/2, off_y = -h/2
                    # Quad: (off_x, off_y) to (off_x+w, off_y+h)
                    
                    # Corner 1 (Bottom Left): Center + Offset
                    c1x = cx - (w/2)*ux - (h/2)*vx
                    c1y = cy - (w/2)*uy - (h/2)*vy
                    c1z = cz - (w/2)*uz - (h/2)*vz
                    
                    # Corner 2 (Bottom Right): C1 + W*U
                    c2x = c1x + w*ux; c2y = c1y + w*uy; c2z = c1z + w*uz
                    # Corner 3 (Top Right): C2 + H*V
                    c3x = c2x + h*vx; c3y = c2y + h*vy; c3z = c2z + h*vz
                    # Corner 4 (Top Left): C1 + H*V
                    c4x = c1x + h*vx; c4y = c1y + h*vy; c4z = c1z + h*vz
                    
                    # Add to batch
                    # x, y, z, u, v, r, g, b, a
                    # U/V are 0,0 1,0 1,1 0,1
                    r,g,b = 1.0, 1.0, 1.0 # White tint
                    
                    verts = [
                        c1x, c1y, c1z, 0.0, 0.0, r, g, b, alpha,
                        c2x, c2y, c2z, 1.0, 0.0, r, g, b, alpha,
                        c4x, c4y, c4z, 0.0, 1.0, r, g, b, alpha, # Tri 1
                        
                        c2x, c2y, c2z, 1.0, 0.0, r, g, b, alpha,
                        c3x, c3y, c3z, 1.0, 1.0, r, g, b, alpha,
                        c4x, c4y, c4z, 0.0, 1.0, r, g, b, alpha  # Tri 2
                    ]
                    batches[char].extend(verts)

        # Draw Batches
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        
        # Stride: 9 floats * 4 bytes
        stride = 9 * 4
        
        for char, verts in batches.items():
            tex = self.engine.get_label(char, self.color)
            glBindTexture(GL_TEXTURE_2D, tex.texture_id)
            
            data = np.array(verts, dtype=np.float32)
            
            # We can use a temporary VBO or just glVertexPointer directly with Client Memory (slower but works)
            # Or use one shared VBO and update it?
            # Since we have many chars, updating one VBO many times is okay.
            
            if not hasattr(self, 'shared_vbo'):
                 self.shared_vbo = vbo.VBO(np.array([], dtype=np.float32))

            self.shared_vbo.set_array(data)
            self.shared_vbo.bind()
            
            glVertexPointer(3, GL_FLOAT, stride, self.shared_vbo)
            glTexCoordPointer(2, GL_FLOAT, stride, self.shared_vbo + 12) # Offset 12 (3 floats)
            glColorPointer(4, GL_FLOAT, stride, self.shared_vbo + 20)    # Offset 20 (5 floats)
            
            glDrawArrays(GL_TRIANGLES, 0, len(data) // 9)
            self.shared_vbo.unbind()
            
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        
        glPopAttrib()

class StatsWall(GameObject):
    def draw(self):
        cam_z = self.engine.cam_pos[2]
        glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        ram_lbl = self.engine.get_label(f"RAM: {self.engine.monitor.ram}%", (0.0, 1.0, 1.0, 1.0))
        glPushMatrix()
        glTranslatef(-3.9, 0, cam_z - 8)
        glRotatef(90, 0, 1, 0)  
        glTranslatef(0, 0, 0.05)
        ram_lbl.draw(0, 0, 0, 0.03)
        glPopMatrix()
        
        cpu_lbl = self.engine.get_label(f"CPU: {self.engine.monitor.cpu}%", (1.0, 0.2, 0.2, 1.0))
        glPushMatrix()
        glTranslatef(3.9, 0, cam_z - 8)
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
        
        spacing = 5.0
        global_cam_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
        
        num_nets = len(networks)
        if num_nets == 0: return
        chunk_size = num_nets * spacing
        if chunk_size == 0: chunk_size = 100.0
        current_chunk = int(global_cam_z / chunk_size)
        
        for chunk_idx in range(current_chunk - 1, current_chunk + 2):
            chunk_start_z = chunk_idx * chunk_size
            for i, (ssid, signal) in enumerate(networks):
                net_global_z = chunk_start_z - (i * spacing)
                net_local_z = net_global_z - getattr(self.engine, 'world_offset_z', 0.0)
                
                if net_local_z > self.engine.cam_pos[2] + 20 or net_local_z < self.engine.cam_pos[2] - 100:
                    continue
                
                if signal >= 70: color = (0.0, 1.0, 0.0, 1.0)
                elif signal >= 40: color = (1.0, 1.0, 0.0, 1.0)
                else: color = (1.0, 0.0, 0.0, 1.0)
                
                start_x = 3.9
                
                glDisable(GL_TEXTURE_2D)
                glDisable(GL_LIGHTING)
                glPushMatrix()
                glTranslatef(start_x, 2.5, net_local_z)
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
                glTranslatef(start_x, 2.5, net_local_z)
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

class CyberCity(GameObject):
    def __init__(self, engine):
        super().__init__(engine)
        self.block_size = 20.0 # Size of a city block
        self.buildings = {} # Key: (block_z_index, side), Value: List of (x, height, width, depth)
        self.cube_mesh = self._create_cube_mesh()

    def _create_cube_mesh(self):
        # Unit cube centered at base (0,0,0) to (1,1,1) ? 
        # Let's do centered at 0,0,0 with size 1
        verts = [
            # Bottom
            (-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5),
            # Top
            (-0.5, 1, -0.5), (0.5, 1, -0.5), (0.5, 1, 0.5), (-0.5, 1, 0.5),
            # Verticals
            (-0.5, 0, -0.5), (-0.5, 1, -0.5),
            (0.5, 0, -0.5), (0.5, 1, -0.5),
            (0.5, 0, 0.5), (0.5, 1, 0.5),
            (-0.5, 0, 0.5), (-0.5, 1, 0.5)
        ]
        # Wireframe indices (lines)
        lines = []
        # Bottom loop
        lines.extend([0,1, 1,2, 2,3, 3,0])
        # Top loop
        lines.extend([4,5, 5,6, 6,7, 7,4])
        # Vertical connectors
        lines.extend([0,4, 1,5, 2,6, 3,7])
        
        # Flatten
        flat_verts = []
        for i in lines:
            flat_verts.append(verts[i])
            
        return Mesh(flat_verts, GL_LINES)

    def _generate_block(self, index):
        random.seed(index)
        buildings = []
        
        # Left side (-X)
        for i in range(random.randint(2, 5)):
            x = -random.uniform(6.0, 15.0)
            z_offset = random.uniform(-self.block_size/2, self.block_size/2)
            w = random.uniform(2.0, 5.0)
            d = random.uniform(2.0, 5.0)
            h = random.uniform(2.0, 15.0) # Height
            buildings.append((x, z_offset, w, d, h, COL_CYAN))

        # Right side (+X)
        for i in range(random.randint(2, 5)):
            x = random.uniform(6.0, 15.0)
            z_offset = random.uniform(-self.block_size/2, self.block_size/2)
            w = random.uniform(2.0, 5.0)
            d = random.uniform(2.0, 5.0)
            h = random.uniform(2.0, 15.0)
            buildings.append((x, z_offset, w, d, h, (1.0, 0.0, 1.0, 1.0))) # Purple/Magenta
            
        return buildings

    def draw(self):
        global_cam_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
        current_block = int(global_cam_z / self.block_size)
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glLineWidth(1.0)
        
        for i in range(current_block - 2, current_block + 5):
            if i not in self.buildings:
                self.buildings[i] = self._generate_block(i)
                # Cleanup old
                if i - 10 in self.buildings:
                    del self.buildings[i-10]
            
            block_base_z = i * self.block_size
            local_base_z = block_base_z - getattr(self.engine, 'world_offset_z', 0.0)
            
            for (x, z_off, w, d, h, col) in self.buildings[i]:
                # Visibility check
                b_local_z = local_base_z + z_off
                if b_local_z > self.engine.cam_pos[2] + 10 or b_local_z < self.engine.cam_pos[2] - 120:
                    continue
                    
                glPushMatrix()
                glTranslatef(x, -5.0, b_local_z) # Ground level approx -5
                glScalef(w, h, d)
                
                # Distance fade
                dist = abs(self.engine.cam_pos[2] - b_local_z)
                alpha = max(0, min(0.6, 1.0 - (dist / 100.0)))
                
                glColor4f(col[0], col[1], col[2], alpha)
                self.cube_mesh.draw()
                
                # Fill bottom slightly to cover grid lines below? No, wireframe is cool.
                glPopMatrix()
                
        glPopAttrib()
