# config.py
"""
Central configuration and constants for the Navi application.
"""
from typing import Tuple

# --- Screen & Engine Settings ---
WIN_WIDTH: int = 1280
WIN_HEIGHT: int = 720
TUNNEL_DEPTH: float = 40.0
FPS_TARGET: int = 60
FOV: int = 100
RENDER_DISTANCE: float = 300.0
CULL_DISTANCE: float = 300.0

# --- Colors (Normalized RGBA) ---
Color = Tuple[float, float, float, float]

COL_CYAN: Color = (0.0, 1.0, 1.0, 1.0)
COL_RED: Color = (1.0, 0.2, 0.2, 1.0)
COL_GRID: Color = (0.0, 0.15, 0.3, 0.4)
COL_DARK: Color = (0.02, 0.02, 0.04, 1.0)
COL_TEXT: Color = (0.8, 0.8, 0.8, 1.0)
COL_GHOST: Color = (0.2, 1.0, 0.2, 0.8)
COL_WHITE: Color = (1.0, 1.0, 1.0, 1.0)
COL_YELLOW: Color = (1.0, 1.0, 0.0, 1.0)
COL_HEX: Color = (0.0, 0.6, 0.0, 1.0)
COL_MAGENTA: Color = (1.0, 0.2, 0.8, 1.0)
COL_ORANGE: Color = (1.0, 153/255.0, 0.0, 1.0)
COL_GREY: Color = (100/255.0, 100/255.0, 100/255.0, 1.0)

# --- Packet Configuration ---
PACKET_ORBITAL_RADIUS_MIN: float = 5.0
PACKET_ORBITAL_RADIUS_MAX: float = 7.0
PACKET_ORBITAL_SPEED_MIN: float = 0.02
PACKET_ORBITAL_SPEED_MAX: float = 0.05
MAX_PACKETS_DISPLAYED: int = 40

# --- World Settings ---
ZONE_THRESHOLDS = {
    'SURFACE': -1000,
    'SPRAWL': -3000,
    'DEEP_WEB': -4500,
    'BLACKWALL': -5500,
    'GHOST_ROOM': -7000,
}
