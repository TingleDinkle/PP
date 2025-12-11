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

class InfiniteTunnel(GameObject):
    def __init__(self, engine):
        super().__init__(engine)
        self.segment_length = 20.0
        self.grid_step = 2.0
        self.mesh = self._generate_segment_mesh()

    def _generate_segment_mesh(self):
        # Generate lines for one 20-unit segment of the tunnel
        verts = []
        
        # Ribs (Cross-sections)
        # We start at z=0 and go to z=-20
        steps = int(self.segment_length / self.grid_step)
        for i in range(steps + 1):
            z = -i * self.grid_step
            # Top, Right, Bottom, Left lines
            # (-4,3) -> (4,3)
            verts.append((-4, 3, z)); verts.append((4, 3, z))
            # (4,3) -> (4,-3)
            verts.append((4, 3, z)); verts.append((4, -3, z))
            # (4,-3) -> (-4,-3)
            verts.append((4, -3, z)); verts.append((-4, -3, z))
            # (-4,-3) -> (-4,3)
            verts.append((-4, -3, z)); verts.append((-4, 3, z))

        # Longitudinal Lines (The rails)
        # 4 rails running from 0 to -segment_length
        corners = [(-4, 3), (4, 3), (4, -3), (-4, -3)]
        for x, y in corners:
            verts.append((x, y, 0))
            verts.append((x, y, -self.segment_length))
            
        return Mesh(verts, GL_LINES)

    def draw(self):
        if self.engine.monitor.disk_write:
            glColor3f(COL_WHITE[0], COL_WHITE[1], COL_WHITE[2])
        else:
            glColor3f(COL_GRID[0], COL_GRID[1], COL_GRID[2])
            
        cam_z = self.engine.cam_pos[2]
        
        # We need to draw segments to cover the view.
        # View range: from cam_z + 10 to cam_z - 80
        # Determine the "index" of the segment the camera is in.
        
        # Because segments are length 20 and go negative:
        # Segment 0: 0 to -20
        # Segment 1: -20 to -40
        # Segment -1: 20 to 0
        
        # If cam_z is -15, we are in segment 0.
        # We want to draw segment -1, 0, 1, 2, 3, 4 (covering -80 to +20)
        
        current_seg_idx = math.floor(-cam_z / self.segment_length)
        
        # Draw 5 segments ahead and 1 behind
        for i in range(-1, 6):
            seg_idx = current_seg_idx + i
            # Position of this segment
            # Segment 0 starts at 0. Segment 1 starts at -20.
            # z_pos = -seg_idx * 20
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
        # Hexagon half-shape
        # (-4, -3) -> (-4, 2)
        verts.append((-4, -3, 0)); verts.append((-4, 2, 0))
        # (-4, 2) -> (-2, 3)
        verts.append((-4, 2, 0)); verts.append((-2, 3, 0))
        # (-2, 3) -> (2, 3)
        verts.append((-2, 3, 0)); verts.append((2, 3, 0))
        # (2, 3) -> (4, 2)
        verts.append((2, 3, 0)); verts.append((4, 2, 0))
        # (4, 2) -> (4, -3)
        verts.append((4, 2, 0)); verts.append((4, -3, 0))
        
        return Mesh(verts, GL_LINES)

    def draw(self):
        # We use Global Z for placement logic (so they stay put in the world)
        # But render relative to Camera Z (local)
        
        # Current Global Z of camera
        global_cam_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
        
        current_idx = int(global_cam_z / self.spacing)
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glLineWidth(2.0)
        
        # Draw 3 arches ahead and 2 behind
        for i in range(current_idx - 2, current_idx + 4):
            # Logical Z position of this arch
            arch_global_z = i * self.spacing
            
            # Calculate Local Z (relative to current camera coordinate system)
            # local_pos = global_pos - world_offset
            arch_local_z = arch_global_z - getattr(self.engine, 'world_offset_z', 0.0)
            
            # Distance fade using local coordinates (cam_pos is local)
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
            
            # Glowing Core (Immediate mode is fine for single points)
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glVertex3f(0, 3, 0)
            glEnd()
            
            glPopMatrix()
            
        glPopAttrib()

class GhostWall(GameObject):
    def draw(self):
        # The Ghost Wall follows the camera at a fixed distance or stays at 0?
        # Original: z = -TUNNEL_DEPTH (fixed).
        # In infinite world, maybe it's a "rearview mirror" or just behind?
        # Let's put it fixed relative to camera for now so it doesn't disappear.
        z = self.engine.cam_pos[2] - 30.0 # Always 30 units in front? No, original was back.
        # Actually, let's make it the "Event Horizon" behind the player.
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
        # Hypercore stays at Local Z = -15 relative to camera?
        # Or does it disappear?
        # Original: glTranslatef(0, 0, -15.0). This was absolute 0,0,-15.
        # If we want it to be a "Start" object that you leave behind:
        # global_z = -15.
        
        global_z = -15.0
        local_z = global_z - getattr(self.engine, 'world_offset_z', 0.0)
        
        # Don't draw if too far away
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
        # Satellites orbit the Hypercore. They should also follow Global Z logic.
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
        t = time.time()
        
        for i, p in enumerate(self.engine.packets):
            if not hasattr(p, 'history'):
                p.history = collections.deque(maxlen=20)
                # Packets should appear near the camera? Or randomly in the "Active" tunnel zone?
                # Let's start them at logical Z relative to camera.
                current_global_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
                # Spawn ahead (more negative) and move towards us (positive)
                p.base_global_z = current_global_z + random.uniform(-50, -150) 
                
                if p.protocol == 'TCP':
                    p.lane_x = -2.5
                elif p.protocol == 'UDP':
                    p.lane_x = 2.5
                else:
                    p.lane_x = 0.0
                p.lane_y = random.uniform(-1.5, 1.5)

            # Store GLOBAL positions in history to be safe against shifts
            # But simpler: Store LOCAL positions and manually shift them when the world shifts?
            # Or store GLOBAL positions and convert to local during draw?
            # Storing Global is cleaner.
            
            # Update position (Global)
            # scroll_speed = 2.0 # Unused now
            # Packet moves forward (negative Z)
            # p.current_global_z = p.base_global_z - (t - self.engine.start_time) * scroll_speed
            # Wait, this logic is tricky if time is continuous.
            # Simpler: just move p.z
            
            if not hasattr(p, 'global_z'):
                p.global_z = p.base_global_z
            
            p.global_z += 0.4 # Move packet BACKWARDS (Positive Z) -> Incoming Traffic
            
            # Keep packet alive if within reasonable distance
            # If it falls way behind camera, maybe respawn or kill?
            # For now, just let them exist.
            
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
        
        world_offset = getattr(self.engine, 'world_offset_z', 0.0)
        
        for p in self.engine.packets:
            if not hasattr(p, 'history') or len(p.history) < 2: continue
            
            glBegin(GL_LINE_STRIP)
            for i, (hx, hy, hz) in enumerate(p.history):
                # Convert hz (global) to local
                local_z = hz - world_offset
                
                alpha = (i / len(p.history)) * 0.5
                glColor4f(p.color[0], p.color[1], p.color[2], alpha)
                glVertex3f(hx, hy, local_z)
            glEnd()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        for p in self.engine.packets:
            if not hasattr(p, 'global_z'): continue
            local_z = p.global_z - world_offset
            
            # Cull
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
            
            if random.random() < 0.1:
                drop['chars'][random.randint(0, len(drop['chars'])-1)] = self._get_char()

    def draw(self):
        # We want the rain to be fixed in the world (Global Z), not attached to camera.
        
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_TEXTURE_BIT)
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        scale = 0.02
        
        # Determine visible range in Global Z
        # Global Z decreases as we go forward.
        global_cam_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
        
        # We tile the rain pattern every 20 units (tunnel segment length)
        # So we don't need infinite unique drops, just repeating blocks of rain logic?
        # Or do we want continuous unique rain?
        # The current implementation generates a fixed list of drops in __init__.
        # If we just draw that list at fixed Z, it will be one small patch.
        
        # SOLUTION: Treat the 'drops' list as a "pattern" that repeats every 20 units.
        segment_length = 20.0
        start_seg = int(global_cam_z / segment_length)
        
        # Draw 4 segments ahead, 1 behind
        for s in range(start_seg - 4, start_seg + 2):
            # Base Global Z for this segment
            seg_global_z = s * segment_length
            
            # Calculate Local Z for this segment
            seg_local_z = seg_global_z - getattr(self.engine, 'world_offset_z', 0.0)
            
            # Draw the batch of drops offset by this segment Z
            # But wait, drops have their own z_offset (0..16).
            # We need to render them relative to the segment start.
            
            batches = {}
            for drop in self.drops:
                for i, char in enumerate(drop['chars']):
                    # Drop Z relative to the pattern origin
                    drop_z_rel = (drop['col_idx'] * 0.8) - 10 
                    
                    # Final Local Z = Segment Local Z + Drop Relative Z
                    z = seg_local_z + drop_z_rel
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
            # Rotation logic remains
            if self.side == 'left':
                glTranslatef(-3.8, 0, 0)
                glRotatef(90, 0, 1, 0)
            else:
                glTranslatef(3.8, 0, 0)
                glRotatef(-90, 0, 1, 0)
            
            glTranslatef(0, 0, 0.1)
            
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
                    # gx, gy are wall-local X/Y. gz is Camera-Local Z.
                    
                    # We need to map Camera-Local Z to Wall-Local X.
                    # Because we translated to (WallX, 0, 0) and rotated:
                    # The "Wall Space" origin is at (WallX, 0, 0).
                    # We want to draw at 'gz'.
                    # Distance from Wall Origin Z (0) to Target Z (gz) is just 'gz'.
                    
                    # Left Wall (Rot 90): Wall X+ = Global Z-.
                    # Right Wall (Rot -90): Wall X+ = Global Z+.
                    
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
        # Attach to camera
        cam_z = self.engine.cam_pos[2]
        
        glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT)
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        ram_lbl = self.engine.get_label(f"RAM: {self.engine.monitor.ram}%", (0.0, 1.0, 1.0, 1.0))
        glPushMatrix()
        glTranslatef(-3.9, 0, cam_z - 5)
        glRotatef(90, 0, 1, 0) 
        glTranslatef(0, 0, 0.05)
        ram_lbl.draw(0, 0, 0, 0.03)
        glPopMatrix()
        
        cpu_lbl = self.engine.get_label(f"CPU: {self.engine.monitor.cpu}%", (1.0, 0.2, 0.2, 1.0))
        glPushMatrix()
        glTranslatef(3.9, 0, cam_z - 5)
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
        
        # Anchor to World Space
        # We want them to be "signs" along the road.
        # Let's space them out every 20 units.
        spacing = 20.0
        
        # Current Global Camera Z
        global_cam_z = self.engine.cam_pos[2] + getattr(self.engine, 'world_offset_z', 0.0)
        
        # We need to map the list of networks to specific world locations.
        # Let's say network[0] is at Global Z = -50. Network[1] at -70...
        # But since the list changes, this might jitter.
        # Ideally, we hash the SSID to a position? 
        # For now, let's just place the current list relative to the "Current Segment" 
        # so they stay roughly in place as long as the list order doesn't change wildly.
        # Better yet: Just place them starting from the "Beginning of Time" (Z=0) 
        # and repeat them if we run out?
        
        # Let's place them starting at a fixed offset ahead of the *current* view 
        # effectively making them "keep appearing" as we travel.
        # But the user wants to "reach" them.
        # So they must be fixed.
        
        # Fix: Place network[i] at Global Z = -i * spacing - 50.
        # This means network 0 is at -50. Network 1 at -70.
        # If I am at -1000, I passed them long ago.
        # This is bad if I want to see them *now*.
        
        # Compromise: Map the networks to the "Current World Chunk".
        # We repeat the list every N units.
        
        num_nets = len(networks)
        if num_nets == 0: return
        
        # Segment the world into chunks of (num_nets * spacing)
        chunk_size = num_nets * spacing
        if chunk_size == 0: chunk_size = 100.0
        
        # Which chunk are we in?
        current_chunk = int(global_cam_z / chunk_size)
        
        # Draw current chunk and next chunk (so we see them coming)
        for chunk_idx in range(current_chunk - 1, current_chunk + 2):
            chunk_start_z = chunk_idx * chunk_size
            
            for i, (ssid, signal) in enumerate(networks):
                # Fixed Global Z for this instance
                # We place them moving Negative Z (forward)
                net_global_z = chunk_start_z - (i * spacing)
                
                # Convert to Local Z
                net_local_z = net_global_z - getattr(self.engine, 'world_offset_z', 0.0)
                
                # Cull
                if net_local_z > self.engine.cam_pos[2] + 20 or net_local_z < self.engine.cam_pos[2] - 100:
                    continue
                
                if signal >= 70:
                    color = (0.0, 1.0, 0.0, 1.0)
                elif signal >= 40:
                    color = (1.0, 1.0, 0.0, 1.0)
                else:
                    color = (1.0, 0.0, 0.0, 1.0)
                
                start_x = 3.9
                
                # Dot
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

                # Text
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