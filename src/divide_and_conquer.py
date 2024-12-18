import torch
import numpy as np
from PIL import Image
import os
import time
import torch.nn.functional as F
from multiprocessing import Process

def load_target_image(image_path, target_size, mode='L', channel=None):
    """Wczytuje obraz docelowy i skaluje go do odpowiedniego rozmiaru."""
    target_image = Image.open(image_path).convert(mode)  # Czarno-biały lub RGB obraz
    target_image = target_image.resize(target_size)
    image_tensor = torch.tensor(np.array(target_image), dtype=torch.uint8).cuda()
    
    if channel is not None and mode == 'RGB':
        image_tensor = image_tensor[:, :, channel]
    
    return image_tensor

def divide_image(image_tensor, grid_size):
    """Dzieli obraz na siatkę fragmentów."""
    if image_tensor.ndim == 3:
        h, w, c = image_tensor.shape
    else:
        h, w = image_tensor.shape
        c = 1
    fragment_height, fragment_width = h // grid_size[0], w // grid_size[1]
    fragments = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            fragment = image_tensor[
                i * fragment_height : (i + 1) * fragment_height,
                j * fragment_width : (j + 1) * fragment_width
            ]
            fragments.append(fragment)
    return fragments

def create_individual(image_size):
    """Tworzy losowego osobnika."""
    return torch.randint(0, 256, size=image_size, dtype=torch.uint8).cuda()
def create_individual_v1(image_size, unique_colors):
    """Tworzy losowego osobnika z unikalnych kolorów występujących w obrazie docelowym."""
    indices = torch.randint(0, len(unique_colors), size=image_size, dtype=torch.long).cuda()
    return unique_colors[indices]

def fitness(population, target_image):
    """Oblicza fitness dla populacji."""
    population_tensor = torch.stack(population)
    fitness_scores = torch.sum(torch.abs(population_tensor - target_image), dim=(1, 2))

    kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    population_tensor_float = population_tensor.float().unsqueeze(1)  # Przekształcenie do formatu NCHW
    noise_penalty = F.conv2d(population_tensor_float, kernel, padding=1)
    noise_penalty = torch.sum(torch.abs(noise_penalty), dim=(1, 2, 3))

    return fitness_scores + noise_penalty

def crossover(parents):
    """Przeprowadza krzyżowanie między rodzicami."""
    parent1, parent2 = parents
    mask = torch.rand_like(parent1, dtype=torch.float32) < 0.5
    return torch.where(mask, parent1, parent2)

def mutate(population, mutation_rate=0.01):
    """Mutuje populację."""
    mutation_mask = torch.rand_like(population, dtype=torch.float32) < mutation_rate
    mutation_values = torch.randint(0, 256, size=population.shape, dtype=torch.uint8).cuda()
    return torch.where(mutation_mask, mutation_values, population)
def get_unique_colors(image_tensor):
    """Zwraca unikalne kolory występujące w obrazie."""

    if image_tensor.ndim == 2:  # Skala szarości
        unique_colors = torch.unique(image_tensor.reshape(-1))
    
    elif image_tensor.ndim == 3:  # RGB
        unique_colors = torch.unique(image_tensor.view(-1, image_tensor.size(-1)), dim=0)
    else:
        raise ValueError("Nieobsługiwany format obrazu")
    return unique_colors
def genetic_algorithm(target_image, output_folder, generations=100, population_size=2000, mutation_rate=0.01):
    """Uruchamia algorytm genetyczny dla jednego fragmentu."""
    os.makedirs(output_folder, exist_ok=True)
    image_size = target_image.shape
    #unique_colors = get_unique_colors(target_image)
    population = [create_individual(image_size) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = fitness(population, target_image)
        best_idx = torch.argmin(fitness_scores).item()
        best_individual = population[best_idx]
        print(f"{output_folder}: Generacja {generation}, Najlepszy wynik: {fitness_scores[best_idx].item()}")

        if generation % 10 == 0:
            # Zakładam, że best_individual to tensor 2D dla kanału czerwonego
            channel = best_individual.cpu().numpy()
            
            # Tworzenie pełnego obrazu RGB 
            rgb_image = np.zeros((channel.shape[0], channel.shape[1], 3), dtype=np.uint8)
            rgb_image[..., 1] = channel  # Wypełniamy kanał 0-red,1green,2 blue
            
            # Zapis obrazu RGB
            best_image = Image.fromarray(rgb_image, mode='RGB')
            best_image.save(os.path.join(output_folder, f"best_image_gen_{generation}.png"))


        top_n = int(population_size * 0.4)
        sorted_indices = torch.argsort(fitness_scores)
        top_indices = sorted_indices[:top_n]

        top_individuals = [population[idx] for idx in top_indices]
        new_population = top_individuals[:]

        parents_indices = torch.randint(0, top_n, (population_size - top_n, 2), dtype=torch.long).cuda()
        parents = [(top_individuals[i], top_individuals[j]) for i, j in parents_indices]
        children = [crossover(parent_pair) for parent_pair in parents]
        children = mutate(torch.stack(children), mutation_rate)
        new_population.extend(children)

        population = new_population

    best_image.save(os.path.join(output_folder, "final_image.png"))

def process_fragment(index, fragment, generations, population_size, mutation_rate):
    """Proces obsługujący jeden fragment obrazu."""
    output_folder = f"fragment_{index + 1}"
    genetic_algorithm(fragment, output_folder, generations, population_size, mutation_rate)

if __name__ == "__main__":
    # Ścieżka do obrazu i konfiguracja
    image_path = "mona_lisa.jpg"
    grid_size = (1, 1)  # Podział na 9 fragmentów
    image_size = (20, 30)
    mode = 'RGB'  # Ustawienie trybu na RGB
    channel =1 # 0 dla czerwonego, 1 dla zielonego, 2 dla niebieskiego, None dla wszystkich kanałów

    target_image = load_target_image(image_path, image_size, mode, channel)
    print("Obraz wczytany.")
    fragments = divide_image(target_image, grid_size)
    print("Obraz podzielony.")
    # Uruchamianie procesów dla każdego fragmentu
    processes = []
    for idx, fragment in enumerate(fragments):
        p = Process(target=process_fragment, args=(idx, fragment, 400, 2000, 0.01))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()