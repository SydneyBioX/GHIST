import tifffile
import imageio
from pathlib import Path


def load_image(fp):
    if Path(fp).suffix.lower() == '.tif':
        image = tifffile.imread(fp)
    else:
        image = imageio.imread(fp)

    return image