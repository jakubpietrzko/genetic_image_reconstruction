import os
import cv2
from PIL import Image

def create_animation_from_images(image_folder, output_file, start_fps=10, end_fps=120, total_frames=1000):
    # Pobierz listę plików w folderze
    image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sprawdź, czy lista plików nie jest pusta
    if not image_files:
        raise ValueError("Brak obrazów w folderze lub nieprawidłowe rozszerzenia plików.")
    
    # Sprawdź, czy wszystkie pliki są prawidłowymi obrazami
    valid_image_files = []
    for img in image_files:
        try:
            with Image.open(img) as im:
                valid_image_files.append(img)
        except Exception as e:
            print(f"Nieprawidłowy plik obrazu: {img}, błąd: {e}")

    if not valid_image_files:
        raise ValueError("Brak prawidłowych obrazów w folderze.")
    
    # Sortowanie plików zgodnie z numeracją w nazwach plików
    valid_image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x).split('_')[-1]))))
    
    # Pobierz rozmiar pierwszego obrazu
    frame = cv2.imread(valid_image_files[0])
    height, width, layers = frame.shape

    # Utwórz obiekt VideoWriter z maksymalnym FPS
    max_fps = end_fps
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_file, fourcc, max_fps, (width, height))

    # Obliczanie liczby powtórzeń dla każdej klatki
    num_images = len(valid_image_files)
    fps_increment = (end_fps - start_fps) / num_images
    current_fps = start_fps

    # Dodaj obrazy do wideo z dynamicznym FPS
    for i, image_file in enumerate(valid_image_files):
        frame = cv2.imread(image_file)
        repeat_count = int(max_fps / current_fps)
        for _ in range(repeat_count):
            video.write(frame)
        current_fps = min(current_fps + fps_increment, end_fps)

    # Zakończ i zapisz wideo
    video.release()

if __name__ == "__main__":
    # Ścieżka do folderu, w którym znajduje się skrypt
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(current_directory, 'black_and_white')  # Folder z obrazami wewnątrz bieżącego folderu
    output_file = 'gr2.avi'  # Nazwa pliku wyjściowego
    create_animation_from_images(image_folder, output_file, start_fps=20, end_fps=20)