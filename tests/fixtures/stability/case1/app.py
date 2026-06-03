from pathlib import Path
def has_data():
    return Path('data/config.txt').exists()
