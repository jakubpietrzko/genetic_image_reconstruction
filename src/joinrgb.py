import numpy as np
from PIL import Image

def merge_rgb_images(red_image_path, green_image_path, blue_image_path, output_path):
    """
    Łączy trzy obrazy RGB, z których każdy ma tylko jedną składową (R, G lub B) i zapisuje wynikowy obraz RGB.

    :param red_image_path: Ścieżka do obrazu z kanałem czerwonym (R).
    :param green_image_path: Ścieżka do obrazu z kanałem zielonym (G).
    :param blue_image_path: Ścieżka do obrazu z kanałem niebieskim (B).
    :param output_path: Ścieżka do zapisu obrazu wynikowego.
    """
    # Wczytaj obrazy RGB (tylko jedna składowa w każdym)
    red_image = np.array(Image.open(red_image_path))[:, :, 0]  # Zbieramy tylko pierwszy kanał (czerwony)
    green_image = np.array(Image.open(green_image_path))[:, :, 1]  # Zbieramy tylko drugi kanał (zielony)
    blue_image = np.array(Image.open(blue_image_path))[:, :, 2]  # Zbieramy tylko trzeci kanał (niebieski)

    # Sprawdzenie, czy wszystkie obrazy mają ten sam rozmiar
    if red_image.shape != green_image.shape or red_image.shape != blue_image.shape:
        raise ValueError("Wszystkie obrazy muszą mieć ten sam rozmiar!")

    # Łączenie kanałów w jeden obraz RGB
    rgb_image = np.stack((red_image, green_image, blue_image), axis=-1).astype(np.uint8)

    # Zapis obrazu wynikowego
    output_image = Image.fromarray(rgb_image, mode="RGB")
    output_image.save(output_path)

    print(f"Obraz RGB zapisany do {output_path}")

merge_rgb_images("fragment_1\\best_image_gen_390R.png", "fragment_1\\best_image_gen_390G.png", "fragment_1\\best_image_gen_390B.png", "rgb_image.png")