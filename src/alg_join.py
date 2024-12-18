import torch
import numpy as np
from PIL import Image
import os

def load_target_image(image_path, target_size, mode='L', channel=None):
    """Wczytuje obraz docelowy i skaluje go do odpowiedniego rozmiaru."""
    target_image = Image.open(image_path).convert(mode)  # Czarno-biały lub RGB obraz
    target_image = target_image.resize(target_size)
    image_tensor = torch.tensor(np.array(target_image), dtype=torch.uint8).cuda()
    
    if channel is not None and mode == 'RGB':
        image_tensor = image_tensor[:, :, channel]
    
    return image_tensor

def divide_image(image_tensor, grid_size, output_folder):
    """Dzieli obraz na siatkę fragmentów i zapisuje każdy fragment jako osobny plik obrazu."""
    if image_tensor.ndim == 3:
        h, w, c = image_tensor.shape
    else:
        h, w = image_tensor.shape
        c = 1
    fragment_height, fragment_width = h // grid_size[0], w // grid_size[1]
    fragments = []
    for i in range(grid_size[0] * grid_size[1]):
        os.makedirs(f"{output_folder}{i+1}", exist_ok=True)
    cnt = 1
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            fragment = image_tensor[
                i * fragment_height : (i + 1) * fragment_height,
                j * fragment_width : (j + 1) * fragment_width
            ]
            fragments.append(fragment)
            fragment_image = Image.fromarray(fragment.cpu().numpy(), mode='L' if c == 1 else 'RGB')
            fragment_image.save(os.path.join(f"{output_folder}{cnt}", "fragment_target.png"))
            cnt += 1
    return fragments

def load_image_from_folder(folder_path, image_name="final_image.png"):
    """Wczytuje obraz z danego folderu."""
    image_path = os.path.join(folder_path, image_name)
    return Image.open(image_path).convert('L')

def combine_images(grid_size, image_size, output_path):
    """Łączy obrazy w skali szarości z folderów w jeden duży obraz."""
    combined_image = Image.new('L', (grid_size[1] * image_size[0], grid_size[0] * image_size[1]))

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            folder_index = i * grid_size[1] + j + 1
            folder_path = f"fragment_{folder_index}"
            image = load_image_from_folder(folder_path)
            combined_image.paste(image, (j * image_size[0], i * image_size[1]))

    combined_image.save(output_path)
    print(f"Połączony obraz zapisany jako {output_path}")

if __name__ == "__main__":
    # Ścieżka do obrazu i konfiguracja
    """image_path = "mona_lisa.jpg"
    grid_size = (5, 5)  # Podział na 25 fragmentów
    image_size = (100, 150)
    mode = 'RGB'  # Ustawienie trybu na RGB
    channel = None  # None dla wszystkich kanałów

    target_image = load_target_image(image_path, image_size, mode, channel)
    print("Obraz wczytany.")
    fragments = divide_image(target_image, grid_size, "fragment_")
    print("Obraz podzielony i zapisany.")"""
    grid_size = (2, 2)  # Siatka 5x5
    image_size = (35, 50)  # Rozmiar pojedynczego obrazu
    output_path = "combined_image.png"  # Ścieżka do zapisu połączonego obrazu

    combine_images(grid_size, image_size, output_path)