import torch
import numpy as np
from PIL import Image
import random
import time
import os
import subprocess
import torch.nn.functional as F

def load_target_image(image_path, target_size):
    target_image = Image.open(image_path).convert('RGB')
    target_image = target_image.resize(target_size)  # Zmiana rozmiaru obrazu docelowego
    return torch.tensor(np.array(target_image), dtype=torch.uint8).cuda()  # Przenieś na GPU

def create_individual(image_size):
    # Tworzenie losowego koloru dla każdego piksela
    individual_array = torch.randint(0, 256, size=(image_size[0], image_size[1], 3), dtype=torch.uint8).cuda()
    return individual_array

def fitness(population, target_image):
    # Przetwarzanie całej populacji jednocześnie
    population_tensor = torch.stack(population)
    # Obliczanie podstawowego fitness
    fitness_scores = torch.sum(torch.abs(population_tensor - target_image), dim=(1, 2, 3))
    # Dodanie penalizacji za szum
    # Tworzenie filtru do obliczania różnic między pikselem a jego sąsiadami
    kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    population_tensor_float = population_tensor.float().permute(0, 3, 1, 2)  # Przekształcenie do formatu NCHW
    # Zastosowanie filtru osobno dla każdego kanału
    noise_penalty = torch.zeros_like(population_tensor_float)
    for i in range(3):  # Dla każdego kanału (R, G, B)
        noise_penalty[:, i:i+1, :, :] = F.conv2d(population_tensor_float[:, i:i+1, :, :], kernel, padding=1)
    noise_penalty = torch.sum(torch.abs(noise_penalty), dim=(1, 2, 3))
    
    # Łączny fitness
    total_fitness = fitness_scores + noise_penalty
    
    return total_fitness
def crossover(parents):
    parent1, parent2 = parents
    mask = torch.rand_like(parent1[:, :, 0], dtype=torch.float32) < 0.5
    child = torch.where(mask.unsqueeze(-1), parent1, parent2)
    return child

def select_parents(population, fitness_scores, top_indices):
    parent1 = population[random.choice(top_indices)]
    parent2 = population[random.choice(top_indices)]
    return parent1, parent2

def mutate(population, mutation_rate=0.01):
    mutation_mask = torch.rand_like(population[:, :, :, 0], dtype=torch.float32) < mutation_rate
    mutation_values = torch.randint(0, 256, size=population.shape, dtype=torch.uint8).cuda()
    mutated_population = torch.where(mutation_mask.unsqueeze(-1), mutation_values, population)
    return mutated_population

def save_population(population, filename):
    np.save(filename, [ind.cpu().numpy() for ind in population])

def load_population(filename, image_size, population_size):
    if os.path.exists(filename):
        population = np.load(filename, allow_pickle=True)
        population = [torch.tensor(ind, dtype=torch.uint8).cuda() for ind in population]
        print("uzyto populacji z pliku")
        # Uzupełnianie populacji losowymi osobnikami, jeśli jest za mała
        while len(population) < population_size:
            population.append(create_individual(image_size))
        return population
    else:
        return [create_individual(image_size) for _ in range(population_size)]
def fitness_v0(population, target_image): 
    population_tensor = torch.stack(population)
    fitness_scores = torch.sum(torch.abs(population_tensor - target_image), dim=(1, 2, 3))
    return fitness_scores 
def genetic_algorithm(target_image, generations=100, population_size=500, mutation_rate=0.01, resume=False):
    image_size = target_image.shape[:2]
    population_file = "population2.npy"

    # Ładowanie populacji z pliku lub tworzenie nowej
    if resume:
        population = load_population(population_file, image_size, population_size)
    else:
        population = [create_individual(image_size) for _ in range(population_size)]

    no_improvement_generations = 0
    best_average_fitness = float('inf')

    for generation in range(generations):
        start_time = time.time()

        # Ocena fitnessu dla całej populacji jednocześnie
        fitness_start = time.time()
        fitness_scores = fitness(population, target_image)
        fitness_end = time.time()

        average_fitness = torch.mean(fitness_scores.float()).item()  # Konwersja na float
        best_idx = torch.argmin(fitness_scores).item()
        best_individual = population[best_idx]

        print(f"Generacja {generation}, Najlepszy wynik: {fitness_scores[best_idx].item()}, Średni wynik: {average_fitness}")

        if generation % 10 == 0:
            best_image = Image.fromarray(best_individual.cpu().numpy())  # Przenieś na CPU przed zapisaniem
            best_image.save(f"100150best_image_generation_{generation+11000}.png")

        crossover_mutate_start = time.time()
        new_population = []  # Zainicjowanie nowej populacji

        # Wybór najlepszych 20% osobników
        top_n = int(population_size * 0.4)
        sorted_indices = torch.argsort(fitness_scores)  # Sortowanie na GPU
        top_indices = sorted_indices[:top_n]

        # Dodanie najlepszych 20% osobników do nowej populacji
        top_individuals = [population[idx] for idx in top_indices]
        new_population.extend(top_individuals)

        # Generowanie nowych osobników
        parents_indices = torch.randint(0, top_n, (population_size - top_n, 2), dtype=torch.long).cuda()
        parents = [(top_individuals[i], top_individuals[j]) for i, j in parents_indices]
        children = [crossover(parent_pair) for parent_pair in parents]
        children = mutate(torch.stack(children), mutation_rate)
        new_population.extend(children)
        crossover_mutate_end = time.time()

        population = new_population  # Uaktualnienie populacji

        end_time = time.time()
        print(f"Czas oceny fitness: {fitness_end - fitness_start:.4f} s")
        print(f"Czas krzyżowania i mutacji: {crossover_mutate_end - crossover_mutate_start:.4f} s")
        print(f"Czas całkowity generacji: {end_time - start_time:.4f} s")

        # Sprawdzenie, czy średnia fitness się poprawiła
        if average_fitness < best_average_fitness:
            best_average_fitness = average_fitness
            no_improvement_generations = 0
        else:
            no_improvement_generations += 1

        # Zatrzymanie, jeśli nie było poprawy przez 30 generacji
        if no_improvement_generations >= 100:
            print("Zatrzymanie algorytmu z powodu braku poprawy przez 30 generacji.")
            break

    # Zapisanie populacji do pliku
    save_population(population, population_file)

    return Image.fromarray(best_individual.cpu().numpy())  # Przenieś na CPU przed zwróceniem

if __name__ == "__main__":
    target_image_path = "mona_lisa.jpg"  # Ścieżka do obrazu docelowego
    image_size = (100, 150)  # Przykładowy rozmiar docelowy
    target_image = load_target_image(target_image_path, image_size)
    best_image = genetic_algorithm(target_image, resume=True)
    # Rejestracja przeglądarki Edge
   # Otwieranie linku w przeglądarce Edge za pomocą subprocess
    edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    url = "https://www.youtube.com/watch?v=A30Fx3wnfwE"
    print(f"Otwieranie linku w przeglądarce Edge za pomocą subprocess: {edge_path} {url}")
    subprocess.run([edge_path, url])
    
    
 # Wyświetlenie najlepszego obrazu
    best_image.save("best_generated_image.png")  # Zapisanie najlepszego obrazu
 