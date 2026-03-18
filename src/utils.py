# utils.py
import re

def extract_timeline(text):
    """
    Simple timeline extraction from text based on timestamps or patterns.
    Example: looks for HH:MM or words like 'first', 'then', 'after'.
    """
    lines = text.split("\n")
    timeline = []

    # match HH:MM format
    time_pattern = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
    for line in lines:
        if time_pattern.search(line) or any(w in line.lower() for w in ["first", "then", "after", "next"]):
            timeline.append(line.strip())

    return timeline