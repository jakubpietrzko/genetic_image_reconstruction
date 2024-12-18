import numpy as np
import matplotlib.pyplot as plt

def plot_fitness(generation_data_file):
    # Załaduj dane z pliku
    generation_data = np.load(generation_data_file, allow_pickle=True)

    # Wyodrębnij dane do wykresów
    generations = [data['generation'] for data in generation_data]
    best_fitness = [data['best_fitness'] for data in generation_data]
    average_fitness = [data['average_fitness'] for data in generation_data]

    # Utwórz wykres najlepszej wartości fitness
    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_fitness, label='Najlepsza wartość fitness')
    plt.plot(generations, average_fitness, label='Średnia wartość fitness')
    plt.xlabel('Generacja')
    plt.ylabel('Fitness')
    plt.title('Zmiany wartości fitness w zależności od generacji')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_plot.png')
    plt.show()

if __name__ == "__main__":
    generation_data_file = 'generation_data.npy'  # Plik z danymi generacji
    plot_fitness(generation_data_file)