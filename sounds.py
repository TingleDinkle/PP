import numpy as np
import pygame

def generate_drone(freq=60, duration=1.0, sample_rate=44100):
    """Generates a 60Hz hum with slight static."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Sine wave
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    # Add some noise/harmonics
    noise = 0.1 * np.random.normal(0, 1, wave.shape)
    audio = wave + noise
    
    # Normalize to 16-bit range
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    # Stereo
    return np.column_stack((audio, audio))

def generate_screech(duration=0.5, sample_rate=44100):
    """Generates a digital screech/modem sound."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Carrier frequency that modulates
    carrier = np.sin(2 * np.pi * 1200 * t) 
    
    # Modulator (change pitch rapidly)
    modulator = np.sin(2 * np.pi * 50 * t)
    
    # Combined FM synthesis-ish
    wave = 0.5 * np.sin(2 * np.pi * (1500 + 500 * modulator) * t)
    
    # Add high pitched noise
    white_noise = 0.3 * np.random.normal(0, 1, wave.shape)
    
    audio = wave + white_noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    return np.column_stack((audio, audio))
