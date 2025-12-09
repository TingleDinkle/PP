import numpy as np
import pygame

def generate_drone(duration=1.0, sample_rate=44100):
    """
    Generates a 'Dark Ambient' server room texture.
    Low frequencies, very subtle, no harsh static.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Deep sub-bass (40Hz) + Low mid (100Hz) for warmth
    # The slight offset (40 vs 41) creates a slow, relaxing binaural beat
    wave1 = 0.6 * np.sin(2 * np.pi * 40 * t)
    wave2 = 0.4 * np.sin(2 * np.pi * 41 * t)
    wave3 = 0.2 * np.sin(2 * np.pi * 100 * t)
    
    # Combine
    wave = wave1 + wave2 + wave3
    
    # Normalize and keep volume LOW (10% of max)
    max_val = np.max(np.abs(wave))
    if max_val > 0:
        wave = wave / max_val
    
    audio = (wave * 32767 * 0.1).astype(np.int16)
    
    return np.column_stack((audio, audio))

def generate_screech(duration=0.1, sample_rate=44100):
    """
    Replaces the 'screech' with a soft, watery data 'blip'.
    Think: A droplet falling in a cavern, or a soft UI click.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Pure sine tone, mid-high frequency but very short
    freq = 880 # A5
    wave = np.sin(2 * np.pi * freq * t)
    
    # Exponential decay to make it percussive (pluck sound)
    envelope = np.exp(-30 * t) 
    
    audio = wave * envelope
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        
    # Keep volume subtle
    audio = (audio * 32767 * 0.15).astype(np.int16)
    
    return np.column_stack((audio, audio))
