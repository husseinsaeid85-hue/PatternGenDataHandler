"""Demo script: renders each pattern, then one augmented batch from the loader.

Run with ``python main.py``. The generator section needs the sample dataset,
so unpack ``data.zip`` next to this file first -- it provides the
``exercise_data/`` directory and ``Labels.json``.
"""

from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator

if __name__ == '__main__':
    # Checker pattern
    checker = Checker(100, 10)  # resolution, tile size
    checker.draw()
    checker.show()

    # Circle pattern
    circle = Circle(1000, 200, (400, 600))  # resolution, radius, position
    circle.draw()
    circle.show()

    # Spectrum pattern
    spectrum = Spectrum(2500)  # resolution
    spectrum.draw()
    spectrum.show()

    # Generator
    file_path = 'exercise_data'
    label_path = 'Labels.json'
    batch_size = 10
    image_size = [32, 32, 3]  # height, width, channels

    gen = ImageGenerator(file_path, label_path, batch_size, image_size,
                         rotation=False, mirroring=False, shuffle=True)
    gen.next()   # pull one batch programmatically
    gen.show()   # pull the following batch and plot it with its class names
