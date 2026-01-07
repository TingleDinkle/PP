import collections
import math
import random
import time
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional

import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

import config

if TYPE_CHECKING:
    from main_gl import WiredEngine

class GameObject:
    """Base class for all 3D entities in the world."""
    def __init__(self, engine: 'WiredEngine'):
        self.engine = engine

    def update(self):
        """Update logic per frame."""
        pass

    def draw(self):
        """Render logic per frame."""
        pass

class Mesh:
    """Helper to manage VBOs for static or semi-static geometry."""
    def __init__(self, vertices: List[Tuple[float, float, float]], mode=GL_LINES):
        self.vertex_count = len(vertices)
        self.mode = mode
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

    def update(self, vertices: List[float], colors: List[float]):
        if not vertices:
            self.vertex_count = 0
            return
        
        pos_data = np.array(vertices, dtype=np.float32)
        col_data = np.array(colors, dtype=np.float32)
        
        self.vertex_count = len(pos_data) // 3
        
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

class ParticleSystem(GameObject):
    """Manages simple one-shot particle explosions."""
    def __init__(self, engine: 'WiredEngine'):
        super().__init__(engine)
        # list of [x, y, z, vx, vy, vz, life, max_life, color_tuple]
        self.particles = [] 
        self.vbo = DynamicMesh(GL_POINTS)

    def explode(self, x: float, y: float, z: float, color: config.Color = (1, 0.5, 0, 1), count: int = 50):
        for _ in range(count):
            vx = random.uniform(-5, 5)
            vy = random.uniform(-5, 5)
            vz = random.uniform(-5, 5)
            life = random.uniform(0.5, 1.5)
            self.particles.append([x, y, z, vx, vy, vz, life, life, color])

    def update(self):
        dt = 0.016
        alive = []
        for p in self.particles:
            p[0] += p[3] * dt # x
            p[1] += p[4] * dt # y
            p[2] += p[5] * dt # z
            p[6] -= dt # life
            if p[6] > 0:
                alive.append(p)
        self.particles = alive

    def draw(self):
        if not self.particles: return
        
        verts = []
        cols = []
        for p in self.particles:
            verts.extend([p[0], p[1], p[2]])
            # Fade out
            alpha = p[6] / p[7]
            # p[8] is color tuple
            cols.extend([p[8][0], p[8][1], p[8][2], alpha])
            
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glPointSize(3.0)
        self.vbo.update(verts, cols)
        self.vbo.draw()
        glPopAttrib()

class Hunter(GameObject):
    """Aggressive entity that spawns during Breach Mode."""
    def __init__(self, engine: 'WiredEngine', start_pos: Tuple[float, float, float]):
        super().__init__(engine)
        self.pos = list(start_pos)
        self.speed = 8.0 
        self.size = 1.0
        
    def update(self):
        # Chase Camera
        target = self.engine.cam_pos
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dz = target[2] - (self.pos[2] - self.engine.world_offset_z)
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > 0.5:
            self.pos[0] += (dx/dist) * self.speed * 0.016
            self.pos[1] += (dy/dist) * self.speed * 0.016
            self.pos[2] += (dz/dist) * self.speed * 0.016 
            
    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        
        glRotatef(time.time() * 200, 1, 1, 0)
        
        glColor4f(1.0, 0.0, 0.0, 0.8)
        scale = 1.0 + math.sin(time.time() * 20) * 0.2
        glScalef(scale, scale, scale)
        
        glBegin(GL_TRIANGLES)
        # Face 1
        glVertex3f(0, 1, 0); glVertex3f(-1, -1, 1); glVertex3f(1, -1, 1)
        # Face 2
        glVertex3f(0, 1, 0); glVertex3f(1, -1, 1); glVertex3f(0, -1, -1)
        # Face 3
        glVertex3f(0, 1, 0); glVertex3f(0, -1, -1); glVertex3f(-1, -1, 1)
        # Bottom
        glVertex3f(-1, -1, 1); glVertex3f(0, -1, -1); glVertex3f(1, -1, 1)
        glEnd()
        
        glPopMatrix()

class InfiniteTunnel(GameObject):
    """Renders the scrolling grid environment."""
    def __init__(self, engine: 'WiredEngine'):
        super().__init__(engine)
        self.segment_length = 20.0
        self.grid_step = 2.0
        self.mesh = self._generate_segment_mesh()

    def _generate_segment_mesh(self) -> Mesh:
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
        grid_col = self.engine.zone_state.get('grid_color', config.COL_GRID)
        zone_name = self.engine.zone_state.get('name', 'SURFACE')
        
        if self.engine.monitor.disk_write:
            # "Pulse"
            r = min(1.0, grid_col[0] + 0.4)
            g = min(1.0, grid_col[1] + 0.4)
            b = min(1.0, grid_col[2] + 0.4)
            a = min(1.0, grid_col[3] + 0.3)
            glColor4f(r, g, b, a)
        else:
            glColor4f(grid_col[0], grid_col[1], grid_col[2], grid_col[3])
            
        cam_z = self.engine.cam_pos[2]
        current_seg_idx = math.floor(-cam_z / self.segment_length)
        
        if zone_name == 'OLD_NET':
            glPointSize(2.0)
        
        # Increased draw distance to match new FOV/Render Distance
        for i in range(-1, 16):
            seg_idx = current_seg_idx + i
            z_pos = -(seg_idx * self.segment_length)
            glPushMatrix()
            glTranslatef(0, 0, z_pos)
            
            # Zone-specific deformations
            if zone_name == 'DEEP_WEB':
                 glRotatef(math.sin(z_pos * 0.1 + time.time()) * 5.0, 0, 0, 1)
            elif zone_name == 'BLACKWALL':
                 glRotatef(math.sin(z_pos * 0.5 + time.time() * 5.0) * 2.0, 1, 0, 0)
            elif zone_name == 'OLD_NET':
                 glRotatef(time.time() * 10.0 + z_pos, 0, 0, 1)
                 glScalef(1.0 + math.sin(time.time()*5)*0.1, 1.0, 1.0)
                 
            if zone_name == 'OLD_NET':
                self.mesh.vbo.bind()
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, self.mesh.vbo)
                glDrawArrays(GL_POINTS, 0, self.mesh.vertex_count)
                glDisableClientState(GL_VERTEX_ARRAY)
                self.mesh.vbo.unbind()
            else:
                self.mesh.draw()
                
            glPopMatrix()

class CyberArch(GameObject):
    def __init__(self, engine: 'WiredEngine'):
        super().__init__(engine)
        self.mesh = self._generate_arch_mesh()
        self.spacing = 50.0

    def _generate_arch_mesh(self) -> Mesh:
        verts = []
        verts.append((-4, -3, 0)); verts.append((-4, 2, 0))
        verts.append((-4, 2, 0)); verts.append((-2, 3, 0))
        verts.append((-2, 3, 0)); verts.append((2, 3, 0))
        verts.append((2, 3, 0)); verts.append((4, 2, 0))
        verts.append((4, 2, 0)); verts.append((4, -3, 0))
        return Mesh(verts, GL_LINES)

    def draw(self):
        global_cam_z = self.engine.cam_pos[2] + self.engine.world_offset_z
        current_idx = int(global_cam_z / self.spacing)
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glLineWidth(2.0)
        
        # Draw more arches to fill the distance
        for i in range(current_idx - 2, current_idx + 8):
            arch_global_z = i * self.spacing
            arch_local_z = arch_global_z - self.engine.world_offset_z
            
            dist = abs(self.engine.cam_pos[2] - arch_local_z)
            if dist > config.CULL_DISTANCE: continue 
            
            alpha = max(0, min(1, 1.0 - (dist / config.CULL_DISTANCE)))
            if alpha <= 0: continue

            colors = [config.COL_CYAN, config.COL_RED, config.COL_YELLOW, (0.5, 0.0, 1.0, 1.0)]
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
        if self.engine.cam_tex:
            self.engine.cam_tex.bind()
        glColor4f(*config.COL_GHOST)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-4, -3, z)
        glTexCoord2f(1, 0); glVertex3f(4, -3, z)
        glTexCoord2f(1, 1); glVertex3f(4, 3, z)
        glTexCoord2f(0, 1); glVertex3f(-4, 3, z)
        glEnd()
        glDisable(GL_TEXTURE_2D)

class Hypercore(GameObject):
    def __init__(self, engine: 'WiredEngine'):
        super().__init__(engine)
        self.quadric = gluNewQuadric()

    def draw(self):
        global_z = -15.0
        local_z = global_z - self.engine.world_offset_z
        
        if local_z > self.engine.cam_pos[2] + 20 or local_z < self.engine.cam_pos[2] - config.CULL_DISTANCE:
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
        gluSphere(self.quadric, 0.8, 16, 16)
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
            gluDisk(self.quadric, 4.0 + i*0.4, 4.1 + i*0.4, 64, 1)
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
            lbl = self.engine.get_label(char, config.COL_WHITE)
            
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
        verts = [
            (0,1,0), (1,0,0), (0,1,0), (-1,0,0), (0,1,0), (0,0,1), (0,1,0), (0,0,-1),
            (0,-1,0), (1,0,0), (0,-1,0), (-1,0,0), (0,-1,0), (0,0,1), (0,-1,0), (0,0,-1),
            (1,0,0), (0,0,1), (0,0,1), (-1,0,0), (-1,0,0), (0,0,-1), (0,0,-1), (1,0,0)
        ]
        for v in verts: glVertex3f(*v)
        glEnd()

class SatelliteSystem(GameObject):
    def update(self):
        current_procs = self.engine.monitor.processes
        self.engine.active_procs_objs = []
        for i, (pid, name, color, port) in enumerate(current_procs):
            # Stable angle via PID
            stable_angle = (pid % 12) * (2 * math.pi / 12)
            
            self.engine.active_procs_objs.append({
                'name': name,
                'pid': pid,
                'angle': stable_angle,
                'radius': 3.0,
                'color': color,
                'port': port,
                'pos': (0,0,0) 
            })

    def draw(self):
        global_z = -15.0
        local_z = global_z - self.engine.world_offset_z
        
        if local_z > self.engine.cam_pos[2] + 20 or local_z < self.engine.cam_pos[2] - config.CULL_DISTANCE:
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
            
            # Since we need exact world coords for raycasting, calculating it here:
            # Ring Rotation Angle
            ring_rot = -self.engine.rotation * 0.5 # Degrees
            ring_rad = math.radians(ring_rot)
            
            # Point in ring space
            rx = px
            ry = py
            
            # Rotated to World Space
            wx = rx * math.cos(ring_rad) - ry * math.sin(ring_rad)
            wy = rx * math.sin(ring_rad) + ry * math.cos(ring_rad)
            wz = local_z
            
            obj['pos'] = (wx, wy, wz)
            
            glPushMatrix()
            glTranslatef(px, py, 0)
            
            glRotatef(self.engine.rotation * 5, 0, 0, 1)
            glScalef(0.2, 0.2, 0.2)
            glColor4f(*obj['color'])
            
            glBegin(GL_QUADS)
            glVertex3f(-1,-1,0); glVertex3f(1,-1,0); glVertex3f(1,1,0); glVertex3f(-1,1,0)
            glEnd()
            glPopMatrix()
            
            glBegin(GL_LINES)
            glColor4f(obj['color'][0], obj['color'][1], obj['color'][2], 0.1)
            glVertex3f(px, py, 0); glVertex3f(0, 0, 0)
            glEnd()
            
            # Label
            glPushMatrix()
            glTranslatef(px, py + 0.5, 0)
            glRotatef(self.engine.rotation * 0.5, 0, 0, 1) # Counter-rotate
            glScalef(0.015, 0.015, 0.015)
            lbl = self.engine.get_label(obj['name'], config.COL_WHITE)
            glTranslatef(-lbl.width/2, 0, 0)
            glEnable(GL_TEXTURE_2D)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            lbl.draw(0, 0, 0, 1.0)
            glDisable(GL_TEXTURE_2D)
            glPopMatrix()
            
        glPopMatrix()
        glPopAttrib()

class DummyPacket:
    """Mock packet for fallback/demo mode."""
    def __init__(self, p_type):
        self.protocol = p_type
        self.size = random.randint(64, 1500)
        self.payload = f"SIMULATED_{random.randint(1000,9999)}"
        self.src = "192.168.1.X"
        self.dst = "10.0.0.X"
        if p_type == 'TCP': self.color = config.COL_CYAN
        elif p_type == 'UDP': self.color = config.COL_ORANGE
        else: self.color = config.COL_GREY

class PacketSystem(GameObject):
    def __init__(self, engine: 'WiredEngine'):
        super().__init__(engine)
        self.trail_vbo = DynamicMesh(GL_LINES)

    def update(self):
        # 1. Process Queue
        try:
            while True:
                p = self.engine.packet_queue.get_nowait()
                self.engine.packets.append(p)
                if p.payload:
                    for char in p.payload:
                        self.engine.data_buffer.append(ord(char) % 256)
                if len(self.engine.packets) > config.MAX_PACKETS_DISPLAYED:
                    self.engine.packets.pop(0)
                self.engine.glitch_level = 0.3
        except Exception: pass
        
        # 2. Simulation Fallback
        if len(self.engine.packets) < 5 and random.random() < 0.1:
            proto = random.choice(['TCP', 'UDP', 'HTTP'])
            p = DummyPacket(proto)
            self.engine.packets.append(p)
            self.engine.glitch_level = 0.1 
        
        # 3. Physics Update
        updated_packets = []
        for p in self.engine.packets:
            if not hasattr(p, 'history'):
                p.history = collections.deque(maxlen=20)
                current_global_z = self.engine.cam_pos[2] + self.engine.world_offset_z
                p.base_global_z = current_global_z + random.uniform(20, -20)
                p.global_z = p.base_global_z # Initialize position relative to camera
                
                # Lane Assignment with Jitter
                jitter = random.uniform(-0.8, 0.8)
                if p.protocol == 'TCP': p.lane_x = -3.0 + jitter
                elif p.protocol == 'UDP': p.lane_x = 3.0 + jitter
                else: p.lane_x = 0.0 + jitter
                
                p.lane_y = random.uniform(-2.0, 2.0)

            p.global_z -= 0.55 
            
            p.x = p.lane_x
            p.y = p.lane_y
            p.history.append((p.x, p.y, p.global_z))
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
        
        world_offset = self.engine.world_offset_z
        pos_data = []
        col_data = []
        
        for p in self.engine.packets:
            if not hasattr(p, 'history') or len(p.history) < 2: continue
            pts = list(p.history)
            for i in range(len(pts) - 1):
                hx1, hy1, hz1 = pts[i]
                lz1 = hz1 - world_offset
                hx2, hy2, hz2 = pts[i+1]
                lz2 = hz2 - world_offset
                alpha = (i / len(pts)) * 0.5
                pos_data.extend([hx1, hy1, lz1, hx2, hy2, lz2])
                c = p.color
                col_data.extend([c[0], c[1], c[2], alpha, c[0], c[1], c[2], alpha])

        if pos_data:
            self.trail_vbo.update(pos_data, col_data)
            self.trail_vbo.draw()

        # Draw Packet Heads (Projectiles) & Labels
        glPointSize(10.0)
        
        glBegin(GL_POINTS)
        for p in self.engine.packets:
            if hasattr(p, 'history') and len(p.history) > 0:
                hx, hy = p.x, p.y
                hz = p.global_z - world_offset
                glColor4f(*p.color)
                glVertex3f(hx, hy, hz)
        glEnd()

        # Labels
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        cam_pos = self.engine.cam_pos
        
        for p in self.engine.packets:
            if hasattr(p, 'history') and len(p.history) > 0:
                hx, hy = p.x, p.y
                hz = p.global_z - world_offset
                
                dx = hx - cam_pos[0]
                dy = hy - cam_pos[1]
                dz = hz - cam_pos[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                if dist < 60.0 and dist > 1.5:
                    glPushMatrix()
                    glTranslatef(hx + 0.8, hy + 0.8, hz)
                    
                    glRotatef(-self.engine.cam_yaw, 0, 1, 0)
                    glRotatef(-self.engine.cam_pitch, 1, 0, 0)
                    
                    scale = 0.015 * (dist / 12.0) 
                    scale = max(0.01, min(0.03, scale))
                    glScalef(scale, scale, scale)
                    
                    label_text = f"{p.protocol} > {p.dst}"
                    if len(label_text) > 24: label_text = label_text[:24] + ".."
                    
                    lbl = self.engine.get_label(label_text, p.color)
                    lbl.draw(0, 0, 0, 1.0)
                    glPopMatrix()
                    
        glDisable(GL_TEXTURE_2D)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPopAttrib()

class DigitalRain(GameObject):
    """
    Optimized Digital Rain using a Texture Atlas and Batched VBO.
    """
    def __init__(self, engine: 'WiredEngine', side='left', color=config.COL_HEX):
        super().__init__(engine)
        self.side = side
        self.color = color
        self.atlas_texture = None
        
        # Drops: list of [x_local, y_global, speed, char_indices_list]
        self.drops = []
        self.num_drops = 150
        
        self.vbo = vbo.VBO(np.array([], dtype=np.float32))

        # Generate Drops
        for i in range(self.num_drops):
             self.drops.append(self._create_drop())

        # Char Map
        self.chars = "0123456789ABCDEF"

    def _create_drop(self):
        x_local = random.uniform(-20, 20) 
        y_global = random.uniform(-10, 10)
        speed = random.uniform(0.1, 0.4)
        length = random.randint(5, 20)
        
        # Use real data from buffer if available
        if len(self.engine.data_buffer) > 50:
            chars = []
            for _ in range(length):
                byte_val = random.choice(self.engine.data_buffer)
                chars.append(byte_val & 0x0F) 
        else:
            chars = [random.randint(0, 15) for _ in range(length)]
            
        return {'x': x_local, 'y': y_global, 's': speed, 'c': chars}

    def _ensure_atlas(self):
        if self.atlas_texture: return
        
        font = self.engine.font
        surf = pygame.Surface((16 * 20, 32), pygame.SRCALPHA)
        
        for i, char in enumerate(self.chars):
            s = font.render(char, True, (255, 255, 255))
            x = i * 20 + (20 - s.get_width())//2
            y = (32 - s.get_height())//2
            surf.blit(s, (x, y))
            
        data = pygame.image.tostring(surf, "RGBA", True)
        self.atlas_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.atlas_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surf.get_width(), surf.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        self.atlas_w = surf.get_width()
        self.atlas_h = surf.get_height()
        self.char_w = 20
        self.char_uv_w = 20 / self.atlas_w

    def update(self):
        for drop in self.drops:
            drop['y'] -= drop['s']
            if drop['y'] < -15:
                drop['y'] = random.uniform(10, 20)
                # Randomly change some chars
            if random.random() < 0.05:
                 idx = random.randint(0, len(drop['c'])-1)
                 drop['c'][idx] = random.randint(0, 15)

    def draw(self):
        self._ensure_atlas()
        
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) 
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.atlas_texture)
        
        cam_z = self.engine.cam_pos[2] + self.engine.world_offset_z
        visible_data = []
        
        # Basis Vectors
        if self.side == 'left':
            wall_x = -3.8
            ux, uy, uz = (0, 0, -1)
            vx, vy, vz = (0, 1, 0)
        else:
            wall_x = 3.8
            ux, uy, uz = (0, 0, 1)
            vx, vy, vz = (0, 1, 0)
            
        r, g, b = self.color[:3]
        
        # Pre-calc UVs
        uvs = []
        for i in range(16):
            u1 = i * self.char_uv_w
            u2 = u1 + self.char_uv_w
            uvs.append((u1, u2))
            
        for drop in self.drops:
            dz = drop['x'] 
            
            cx = wall_x
            cy = drop['y']
            cz = dz 
            
            wx = cx
            wy = cy
            wz = cam_z + cz - self.engine.world_offset_z 
            
            for i, char_idx in enumerate(drop['c']):
                char_y = wy + (i * 0.4)
                if char_y > 5 or char_y < -5: continue
                
                alpha = 1.0 - (i / len(drop['c']))
                u1, u2 = uvs[char_idx]
                
                # Quad Size
                w = 0.4
                h = 0.6
                
                # Manual expansion for performance
                if self.side == 'left':
                     z_off = -w/2
                else:
                     z_off = w/2
                     
                x, y, z = wx, char_y, wz
                
                visible_data.extend([
                    x, y - h/2, z - z_off, u1, 1.0, r, g, b, alpha,
                    x, y - h/2, z + z_off, u2, 1.0, r, g, b, alpha,
                    x, y + h/2, z + z_off, u2, 0.0, r, g, b, alpha,
                    x, y + h/2, z - z_off, u1, 0.0, r, g, b, alpha
                ])

        if visible_data:
            data = np.array(visible_data, dtype=np.float32)
            self.vbo.set_array(data)
            self.vbo.bind()
            
            stride = 9 * 4
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            glVertexPointer(3, GL_FLOAT, stride, self.vbo)
            glTexCoordPointer(2, GL_FLOAT, stride, self.vbo + 12)
            glColorPointer(4, GL_FLOAT, stride, self.vbo + 20)
            
            glDrawArrays(GL_QUADS, 0, len(visible_data) // 9)
            
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            self.vbo.unbind()
            
        glPopAttrib()

class CyberCity(GameObject):
    def __init__(self, engine: 'WiredEngine'):
        super().__init__(engine)
        self.block_size = 20.0
        self.buildings = {} 
        self.cube_mesh = self._create_cube_mesh()

    def _create_cube_mesh(self) -> Mesh:
        verts = [
            (-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5),
            (-0.5, 1, -0.5), (0.5, 1, -0.5), (0.5, 1, 0.5), (-0.5, 1, 0.5),
            (-0.5, 0, -0.5), (-0.5, 1, -0.5),
            (0.5, 0, -0.5), (0.5, 1, -0.5),
            (0.5, 0, 0.5), (0.5, 1, 0.5),
            (-0.5, 0, 0.5), (-0.5, 1, 0.5)
        ]
        lines = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
        flat_verts = [verts[i] for i in lines]
        return Mesh(flat_verts, GL_LINES)

    def _generate_block(self, index: int):
        random.seed(index)
        buildings = []
        for i in range(random.randint(2, 5)):
            x = -random.uniform(6.0, 15.0)
            z_offset = random.uniform(-self.block_size/2, self.block_size/2)
            w = random.uniform(2.0, 5.0); d = random.uniform(2.0, 5.0); h = random.uniform(2.0, 15.0)
            buildings.append((x, z_offset, w, d, h, config.COL_CYAN))
        for i in range(random.randint(2, 5)):
            x = random.uniform(6.0, 15.0)
            z_offset = random.uniform(-self.block_size/2, self.block_size/2)
            w = random.uniform(2.0, 5.0); d = random.uniform(2.0, 5.0); h = random.uniform(2.0, 15.0)
            buildings.append((x, z_offset, w, d, h, (1.0, 0.0, 1.0, 1.0)))
        return buildings

    def draw(self):
        if self.engine.zone_state.get('name') != 'SPRAWL': return
        global_cam_z = self.engine.cam_pos[2] + self.engine.world_offset_z
        current_block = int(global_cam_z / self.block_size)
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glLineWidth(1.0)
        
        # Draw more blocks
        for i in range(current_block - 2, current_block + 15):
            if i not in self.buildings:
                self.buildings[i] = self._generate_block(i)
                if i - 10 in self.buildings: del self.buildings[i-10]
            
            block_base_z = i * self.block_size
            local_base_z = block_base_z - self.engine.world_offset_z
            
            for (x, z_off, w, d, h, col) in self.buildings[i]:
                b_local_z = local_base_z + z_off
                if b_local_z > self.engine.cam_pos[2] + 10 or b_local_z < self.engine.cam_pos[2] - config.CULL_DISTANCE: continue
                glPushMatrix()
                glTranslatef(x, -5.0, b_local_z)
                glScalef(w, h, d)
                dist = abs(self.engine.cam_pos[2] - b_local_z)
                alpha = max(0, min(0.6, 1.0 - (dist / config.CULL_DISTANCE)))
                glColor4f(col[0], col[1], col[2], alpha)
                self.cube_mesh.draw()
                glPopMatrix()
        glPopAttrib()

class Blackwall(GameObject):
    """
    The Firewall Boundary. A complex, morphing hyper-structure.
    """
    def __init__(self, engine: 'WiredEngine'):
        super().__init__(engine)
        self.primary_mesh = self._generate_hyper_lattice(radius=50.0, density=40)
        self.core_mesh = self._generate_polyhedron(size=20.0)
        self.glitch_timer = 0.0
        self.glitch_state = 0.0

    def _generate_hyper_lattice(self, radius: float, density: int) -> Mesh:
        """Generates a dense, circular grid lattice."""
        verts = []
        for i in range(density):
            angle = (i / density) * 2 * math.pi
            # Outer Ring
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            verts.extend([(x, y, 0), (x*0.9, y*0.9, 5)])
            
            # Cross lattice
            if i % 2 == 0:
                verts.extend([(x, y, 0), (-x, -y, 0)])
                
        # Concentric Rings
        for r in [radius * 0.5, radius * 0.75, radius]:
            for i in range(density):
                a1 = (i / density) * 2 * math.pi
                a2 = ((i+1) / density) * 2 * math.pi
                verts.extend([
                    (math.cos(a1)*r, math.sin(a1)*r, 0),
                    (math.cos(a2)*r, math.sin(a2)*r, 0)
                ])
        return Mesh(verts, GL_LINES)

    def _generate_polyhedron(self, size: float) -> Mesh:
        """Generates a jagged core structure."""
        verts = []
        # Octahedron-ish
        axis = [
            (size, 0, 0), (-size, 0, 0),
            (0, size, 0), (0, -size, 0),
            (0, 0, size), (0, 0, -size)
        ]
        # Connect everything to everything for a chaotic core
        for i in range(len(axis)):
            for j in range(i+1, len(axis)):
                verts.extend([axis[i], axis[j]])
        return Mesh(verts, GL_LINES)

    def draw(self):
        zone_name = self.engine.zone_state.get('name')
        if zone_name not in ['DEEP_WEB', 'BLACKWALL', 'OLD_NET']: return
        
        # Position logic
        wall_z = config.ZONE_THRESHOLDS['DEEP_WEB'] 
        local_wall_z = wall_z - self.engine.world_offset_z
        
        # Culling (Keep visible for longer)
        if local_wall_z > self.engine.cam_pos[2] + 200 or local_wall_z < self.engine.cam_pos[2] - config.CULL_DISTANCE * 3: 
             return

        breached = self.engine.blackwall_state.get('breached', False)
        t = time.time()
        
        # Calculate Morphing parameters
        if breached:
            # Chaos Mode
            pulse = 1.0 + math.sin(t * 20.0) * 0.5
            rotate_speed = t * 100.0
            scale_factor = 4.0 + math.sin(t * 5.0) # Massive expansion
            color = (1.0, 0.0, 0.0, 0.8) if int(t*10)%2==0 else (0.0, 0.0, 0.0, 1.0)
        else:
            # Warning Mode
            pulse = 1.0 + math.sin(t * 2.0) * 0.05
            rotate_speed = t * 10.0
            scale_factor = 1.0
            color = (1.0, 0.0, 0.0, 0.6 + math.sin(t)*0.2)

        # Draw
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glLineWidth(2.0 if breached else 1.5)
        
        glPushMatrix()
        glTranslatef(0, 0, local_wall_z)
        
        # Main Scale
        glScalef(scale_factor, scale_factor, scale_factor)
        
        # --- Layer 1: Hyper Lattice (Rotating) ---
        glPushMatrix()
        glRotatef(rotate_speed, 0, 0, 1)
        glScalef(pulse, pulse, 1.0)
        glColor4f(*color)
        self.primary_mesh.draw()
        glPopMatrix()
        
        # --- Layer 2: Counter-Rotating Lattice ---
        glPushMatrix()
        glRotatef(-rotate_speed * 0.5, 0, 0, 1)
        glScalef(pulse * 0.8, pulse * 0.8, 1.0)
        glColor4f(color[0], 0.2, 0.2, 0.4)
        self.primary_mesh.draw()
        glPopMatrix()

        # --- Layer 3: The Core (Chaos Geometry) ---
        glPushMatrix()
        # Random axis rotation
        glRotatef(t * 50, math.sin(t), math.cos(t), math.sin(t*0.5))
        glScalef(0.5, 0.5, 0.5)
        
        if breached:
             # Core explodes outward
             glScalef(t % 5, t % 5, t % 5) 
             glColor4f(1.0, 1.0, 1.0, 0.8)
        else:
             glColor4f(1.0, 0.0, 0.0, 0.9)
             
        self.core_mesh.draw()
        glPopMatrix()

        glPopMatrix()
        glPopAttrib()

class AlienSwarm(GameObject):
    def __init__(self, engine: 'WiredEngine'):
        super().__init__(engine)
        self.entities = []
        for i in range(100):
             self.entities.append({
                 'offset': (random.uniform(-40, 40), random.uniform(-20, 20), random.uniform(-80, 50)),
                 'speed': random.uniform(0.5, 4.0),
                 'scale': random.uniform(0.5, 3.0),
                 'axis': (random.random(), random.random(), random.random())
             })

    def draw(self):
        if self.engine.zone_state.get('name') != 'OLD_NET': return
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glLineWidth(1.0)
        cam_z = self.engine.cam_pos[2]
        global_z = cam_z + self.engine.world_offset_z
        base_z = global_z
        local_base_z = base_z - self.engine.world_offset_z

        for e in self.entities:
            ox, oy, oz = e['offset']
            t = time.time()
            dx = math.sin(t * e['speed'] + ox) * 10.0
            dy = math.cos(t * e['speed'] * 0.5 + oy) * 10.0
            px = ox + dx; py = oy + dy; pz = local_base_z + oz 
            if pz > cam_z + 50: continue 
            glPushMatrix()
            glTranslatef(px, py, pz)
            glRotatef(t * 100 * e['speed'], e['axis'][0], e['axis'][1], e['axis'][2])
            glScalef(e['scale'], e['scale'], e['scale'])
            glColor4f(1.0, 0.8, 0.2, 0.9) 
            glBegin(GL_LINES)
            verts = [(0,1,0), (-1,-1,1), (1,-1,1), (0,-1,-1)]
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    glVertex3f(*verts[i]); glVertex3f(*verts[j])
            glEnd()
            glPopMatrix()
        glPopAttrib()

class StatsWall(GameObject):
    def draw(self):
        cam_z = self.engine.cam_pos[2]
        glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        ram_lbl = self.engine.get_label(f"RAM: {self.engine.monitor.ram}%", config.COL_CYAN)
        glPushMatrix()
        glTranslatef(-3.9, 0, cam_z - 8)
        glRotatef(90, 0, 1, 0)  
        glTranslatef(0, 0, 0.05)
        ram_lbl.draw(0, 0, 0, 0.03)
        glPopMatrix()
        cpu_lbl = self.engine.get_label(f"CPU: {self.engine.monitor.cpu}%", config.COL_RED)
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
        if not networks: return
        glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        spacing = 5.0
        global_cam_z = self.engine.cam_pos[2] + self.engine.world_offset_z
        num_nets = len(networks)
        if num_nets == 0: return
        chunk_size = num_nets * spacing
        if chunk_size == 0: chunk_size = 100.0
        current_chunk = int(global_cam_z / chunk_size)
        
        for chunk_idx in range(current_chunk - 1, current_chunk + 4):
            chunk_start_z = chunk_idx * chunk_size
            for i, (ssid, signal) in enumerate(networks):
                net_global_z = chunk_start_z - (i * spacing)
                net_local_z = net_global_z - self.engine.world_offset_z
                if net_local_z > self.engine.cam_pos[2] + 20 or net_local_z < self.engine.cam_pos[2] - config.CULL_DISTANCE: continue
                
                if signal >= 70: color = config.COL_HEX
                elif signal >= 40: color = config.COL_YELLOW
                else: color = config.COL_RED
                
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
            lbl = self.engine.get_label("CLOSE THE WORLD", config.COL_WHITE)
            glColor4f(1, 1, 1, alpha) 
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, lbl.texture_id)
            win_width = self.engine.screen.get_width()
            win_height = self.engine.screen.get_height()
            x = (win_width - lbl.width) / 2
            y = (win_height / 2) - 40
            w = lbl.width; h = lbl.height
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(x, y); glTexCoord2f(1, 0); glVertex2f(x + w, y)
            glTexCoord2f(1, 1); glVertex2f(x + w, y + h); glTexCoord2f(0, 1); glVertex2f(x, y + h)
            glEnd()
            lbl2 = self.engine.get_label("OPEN THE NEXT", config.COL_WHITE)
            y += 50
            x = (win_width - lbl2.width) / 2
            glBindTexture(GL_TEXTURE_2D, lbl2.texture_id)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(x, y); glTexCoord2f(1, 0); glVertex2f(x + lbl2.width, y)
            glTexCoord2f(1, 1); glVertex2f(x + lbl2.width, y + lbl2.height); glTexCoord2f(0, 1); glVertex2f(x, y + lbl2.height)
            glEnd()
            glDisable(GL_TEXTURE_2D)
            glMatrixMode(GL_PROJECTION); glPopMatrix()
            glMatrixMode(GL_MODELVIEW); glPopMatrix()
            glEnable(GL_DEPTH_TEST)