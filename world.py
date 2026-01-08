import math
import random
import time
import collections
import pygame
from pygame.locals import *
import config
import entities
import psutil

class World:
    """
    Encapsulates the Game Logic, State, and Scene Management.
    Separates 'What is happening' from 'How it is rendered'.
    """
    def __init__(self, engine):
        self.engine = engine
        
        # Camera / Player State
        self.cam_pos = [0.0, 0.0, 5.0] 
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.world_offset_z = 0.0 
        self.rotation = 0.0
        
        # Game Data State
        self.packets = []
        self.active_procs_objs = [] 
        self.data_buffer = collections.deque(maxlen=2000) 
        for _ in range(200): self.data_buffer.append(random.randint(0, 255))
        
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
        
        self.glitch_level = 0.0 

        self.entities = []
        self.particle_system = None
        
    def init_entities(self):
        """Initialize all game entities."""
        self.entities = [
            entities.InfiniteTunnel(self.engine),
            entities.CyberArch(self.engine),
            entities.GhostWall(self.engine),
            entities.Hypercore(self.engine),
            entities.SatelliteSystem(self.engine),
            entities.PacketSystem(self.engine),
            entities.CyberCity(self.engine),
            entities.Blackwall(self.engine),
            entities.AlienSwarm(self.engine),
            entities.StatsWall(self.engine),
            entities.WifiVisualizer(self.engine),
            entities.DigitalRain(self.engine, side='left', color=config.COL_HEX),
            entities.DigitalRain(self.engine, side='right', color=config.COL_RED),
            entities.CustomModel(self.engine),
            entities.GhostRoom(self.engine),
            entities.IntroOverlay(self.engine)
        ]
        self.particle_system = entities.ParticleSystem(self.engine)
        self.entities.append(self.particle_system)

    def handle_input(self, dt: float):
        keys = pygame.key.get_pressed()
        speed = 2.0 
        
        # Ghost Room Ending Logic
        ghost_room_z = config.ZONE_THRESHOLDS['GHOST_ROOM'] - 100.0
        current_global_z = self.cam_pos[2] + self.world_offset_z
        
        if self.ghost_room_reached:
            # Lock controls completely
            dx = 0; dz = 0
            # Force position to "sweet spot"
            target_local_z = ghost_room_z - self.world_offset_z + 40.0
            
            # Smoothly interpolate
            self.cam_pos[0] = self.cam_pos[0] * 0.9 + 0.0 * 0.1
            self.cam_pos[2] = self.cam_pos[2] * 0.9 + target_local_z * 0.1
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

        # Wall Collision Logic
        next_z = self.cam_pos[2] + dz
        global_z = next_z + self.world_offset_z
        wall_z = config.ZONE_THRESHOLDS['DEEP_WEB'] 
        
        # Soft Wall / Resistance Logic
        if not self.blackwall_state['breached'] and global_z < wall_z + 100:
             dist = global_z - wall_z
             if dist < 50:
                 resistance = (50 - dist) * 0.05
                 dz += resistance
                 # Screen Shake
                 self.cam_pos[0] += random.uniform(-0.05, 0.05)
                 self.cam_pos[1] += random.uniform(-0.05, 0.05)
        
        self.cam_pos[0] += dx
        self.cam_pos[2] += dz
        self.cam_pos[0] = max(-3.5, min(3.5, self.cam_pos[0]))
        self.cam_pos[1] = max(-2.5, min(2.5, self.cam_pos[1]))
        self.cam_pos[2] = min(10.0, self.cam_pos[2])

    def update(self):
        # Update all entities
        for entity in self.entities: 
            entity.update()
            
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
                 if self.engine.explode_sound: self.engine.explode_sound.play()
                 self.cam_pos[2] -= 20.0 

             # Warning logic
             if dist < 300 and self.blackwall_state['warnings'] == 0:
                 self.blackwall_state['warnings'] = 1
                 self.blackwall_state['message'] = "WARNING: CLASSIFIED DATA"
                 self.blackwall_state['last_warning_time'] = time.time()
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
                if self.engine.screech_sound: self.engine.screech_sound.play()
                for _ in range(5):
                    spawn_pos = (
                        self.cam_pos[0] + random.uniform(-10, 10), 
                        self.cam_pos[1] + random.uniform(-5, 5), 
                        self.cam_pos[2] - 50
                    )
                    self.entities.append(entities.Hunter(self.engine, spawn_pos))
        
        # Floating Origin
        if self.cam_pos[2] < -100.0:
            shift = self.cam_pos[2]
            self.cam_pos[2] -= shift
            self.world_offset_z += shift

    def ray_sphere_intersect(self, r_origin, r_dir, s_center, s_radius):
        lx = s_center[0] - r_origin[0]
        ly = s_center[1] - r_origin[1]
        lz = s_center[2] - r_origin[2]
        tca = lx*r_dir[0] + ly*r_dir[1] + lz*r_dir[2]
        if tca < 0: return False 
        d2 = (lx*lx + ly*ly + lz*lz) - tca*tca
        if d2 > s_radius * s_radius: return False
        return True

    def handle_mouse_click(self, start, direction):
        if not start: return
        for obj in self.active_procs_objs:
            if 'pos' not in obj: continue
            center = obj['pos']
            if self.ray_sphere_intersect(start, direction, center, 2.0):
                print(f"TERMINATING PROCESS: {obj['name']} ({obj['pid']})")
                try:
                    p = psutil.Process(obj['pid'])
                    p.terminate()
                    if self.particle_system:
                        self.particle_system.explode(center[0], center[1], center[2], color=(1,0,0,1))
                    if self.engine.explode_sound: self.engine.explode_sound.play()
                except: pass
                break 
