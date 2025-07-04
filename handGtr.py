# This is a Python module for handling guitar chords and notes.
import re
import numpy as np
# Define the mapping of notes to their corresponding frequencies
note_frequencies = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
    'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
}
# Define the mapping of notes to their corresponding MIDI numbers
note_midi = {
    'C': 60, 'C#': 61, 'D': 62, 'D#': 63,
    'E': 64, 'F': 65, 'F#': 66, 'G': 67,
    'G#': 68, 'A': 69, 'A#': 70, 'B': 71
}
# Define the mapping of notes to their corresponding guitar fret numbers
note_frets = {
    'C': 3, 'C#': 4, 'D': 5, 'D#': 6,
    'E': 7, 'F': 8, 'F#': 9, 'G': 10,
    'G#': 11, 'A': 12, 'A#': 13, 'B': 14
}
def note_to_frequency(note):
    """Convert a note to its corresponding frequency."""
    return note_frequencies.get(note, None)
def note_to_midi(note):
    """Convert a note to its corresponding MIDI number."""
    return note_midi.get(note, None)
def note_to_fret(note):
    """Convert a note to its corresponding guitar fret number."""
    return note_frets.get(note, None)
def parse_chord(chord):
    """Parse a chord string into its constituent notes."""
    # Remove any whitespace and split the chord by commas
    chord = chord.replace(" ", "")
    notes = re.split(r'[,+]', chord)
    return [note for note in notes if note in note_frequencies]
def chord_to_frequencies(chord):
    """Convert a chord string into a list of frequencies."""
    notes = parse_chord(chord)
    return [note_to_frequency(note) for note in notes if note_to_frequency(note) is not None]
def chord_to_midi(chord):
    """Convert a chord string into a list of MIDI numbers."""
    notes = parse_chord(chord)
    return [note_to_midi(note) for note in notes if note_to_midi(note) is not None]
def chord_to_frets(chord):
    """Convert a chord string into a list of guitar fret numbers."""
    notes = parse_chord(chord)
    return [note_to_fret(note) for note in notes if note_to_fret(note) is not None]
def chord_to_numpy(chord):
    """Convert a chord string into a NumPy array of frequencies."""
    freqs = chord_to_frequencies(chord)
    return np.array(freqs, dtype=np.float32) if freqs else np.array([], dtype=np.float32)