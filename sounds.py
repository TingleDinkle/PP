import pygame
import numpy as np
import math

# Initialize Pygame Mixer
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    AUDIO_ENABLED = True
except Exception as e:
    print(f"Audio Init Failed: {e}")
    AUDIO_ENABLED = False

def generate_drone(freq=55, duration=1.0, sample_rate=44100):
    """Generates a low sci-fi drone sound."""
    if not AUDIO_ENABLED: return None
    
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    
    # Base Wave (Sawtooth-ish)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    wave += 0.25 * np.sin(2 * np.pi * (freq * 2.02) * t) # Detuned octave
    wave += 0.125 * np.sin(2 * np.pi * (freq * 0.5) * t) # Sub octave
    
    # Modulation
    mod = 1.0 + 0.1 * np.sin(2 * np.pi * 0.2 * t)
    wave *= mod
    
    # Stereo Field (Left/Right phase shift)
    audio = np.zeros((n_samples, 2), dtype=np.int16)
    max_val = 32767 * 0.5 # 50% Volume
    
    audio[:, 0] = (wave * max_val).astype(np.int16)
    audio[:, 1] = (np.roll(wave, 100) * max_val).astype(np.int16) # Slight delay for stereo
    
    return pygame.sndarray.make_sound(audio)

def generate_screech(duration=0.5, sample_rate=44100):
    """Generates a high-pitch modem/glitch noise."""
    if not AUDIO_ENABLED: return None
    
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    
    # Carrier
    freq = 3000
    carrier = np.sin(2 * np.pi * freq * t)
    
    # FM Modulation
    mod_freq = 150
    modulator = np.sin(2 * np.pi * mod_freq * t) * 500
    
    wave = np.sin(2 * np.pi * (freq + modulator) * t)
    
    # Add White Noise
    noise = np.random.uniform(-0.5, 0.5, n_samples)
    wave = wave * 0.5 + noise * 0.5
    
    # Envelope (Decay)
    envelope = np.exp(-3 * t)
    wave *= envelope
    
    audio = np.zeros((n_samples, 2), dtype=np.int16)
    max_val = 32767 * 0.3
    
    audio[:, 0] = (wave * max_val).astype(np.int16)
    audio[:, 1] = (wave * max_val).astype(np.int16)
    
    return pygame.sndarray.make_sound(audio)

def generate_explosion(duration=0.8, sample_rate=44100):
    if not AUDIO_ENABLED: return None
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    
    noise = np.random.uniform(-1, 1, n_samples)
    envelope = np.exp(-5 * t)
    wave = noise * envelope
    
    audio = np.zeros((n_samples, 2), dtype=np.int16)
    max_val = 32767 * 0.6
    audio[:,0] = (wave * max_val).astype(np.int16)
    audio[:,1] = (wave * max_val).astype(np.int16)
    return pygame.sndarray.make_sound(audio)
