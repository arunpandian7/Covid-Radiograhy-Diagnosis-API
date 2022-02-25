from pathlib import Path


def clean_dir(path: Path):
    for item in path.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            clean_dir(item)
    return