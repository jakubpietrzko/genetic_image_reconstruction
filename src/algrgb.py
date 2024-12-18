import torch
import numpy as np
from PIL import Image
import os
import time
import torch.nn.functional as F
from multiprocessing import Process

def load_target_image(image_path, target_size):
    """Wczytuje obraz docelowy w trybie RGB i skaluje go do odpowiedniego rozmiaru."""
    target_image = Image.open(image_path).convert('RGB')  # Wczytanie jako RGB
    target_image = target_image.resize(target_size)
    image_tensor = torch.tensor(np.array(target_image), dtype=torch.uint8).cuda()
    return image_tensor

def create_individual(image_size):
    """Tworzy losowego osobnika."""
    return torch.randint(0, 256, size=image_size, dtype=torch.uint8).cuda()

def fitness(population, target_channel):
    """Oblicza fitness dla populacji w odniesieniu do kanału docelowego."""
    population_tensor = torch.stack(population)
    fitness_scores = torch.sum(torch.abs(population_tensor - target_channel), dim=(1, 2))

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

def genetic_algorithm(target_channel, output_folder, generations=100, population_size=500, mutation_rate=0.01):
    """Uruchamia algorytm genetyczny dla pojedynczego kanału."""
    os.makedirs(output_folder, exist_ok=True)
    image_size = target_channel.shape
    population = [create_individual(image_size) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = fitness(population, target_channel)
        best_idx = torch.argmin(fitness_scores).item()
        best_individual = population[best_idx]
        print(f"{output_folder}: Generacja {generation}, Najlepszy wynik: {fitness_scores[best_idx].item()}")

        if generation % 10 == 0:
            best_image = Image.fromarray(best_individual.cpu().numpy(), mode='L')
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
    return best_individual

def process_channel(channel_name, target_channel, generations, population_size, mutation_rate):
    """Proces obsługujący jeden kanał obrazu."""
    output_folder = f"channel_{channel_name}"
    return genetic_algorithm(target_channel, output_folder, generations, population_size, mutation_rate)

def process_image(target_image, generations, population_size, mutation_rate, divide=False):
    """Przeprowadza proces dla wszystkich kanałów RGB."""
    channels = ['R', 'G', 'B']
    target_channels = [target_image[:, :, i] for i in range(3)]

    # Jeśli obraz jest podzielony na dwie części
    if divide:
        mid = target_image.shape[1] // 2
        left_half = target_image[:, :mid, :]
        right_half = target_image[:, mid:, :]
        halves = [("left", left_half), ("right", right_half)]
        
        for name, half in halves:
            processes = []
            for idx, channel in enumerate(target_channels):
                p = Process(target=process_channel, args=(f"{name}_{channels[idx]}", channel, generations, population_size, mutation_rate))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
    else:
        # Pojedynczy proces dla każdego kanału
        results = []
        for idx, channel in enumerate(target_channels):
            result = process_channel(channels[idx], channel, generations, population_size, mutation_rate)
            results.append(result)

        # Łączenie kanałów w pełny obraz RGB
        final_image = torch.stack(results, dim=-1).cpu().numpy().astype(np.uint8)
        final_image = Image.fromarray(final_image, mode='RGB')
        final_image.save("final_image_rgb.png")

if __name__ == "__main__":
    # Ścieżka do obrazu i konfiguracja
    image_path = "mona_lisa.jpg"
    image_size = (20, 30)  # Rozmiar obrazu docelowego
    generations = 100
    population_size = 3000
    mutation_rate = 0.01
    divide = False  # Ustaw na True, aby podzielić obraz na dwie części

    target_image = load_target_image(image_path, image_size)
    process_image(target_image, generations, population_size, mutation_rate, divide)
