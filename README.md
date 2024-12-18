# Dokumentacja projektu: Rekonstrukcja obrazów za pomocą programowania genetycznego

## Opis projektu

Projekt realizuje rekonstrukcję obrazów przy użyciu algorytmów genetycznych, które symulują proces ewolucji. Celem algorytmu jest wygenerowanie obrazu, który jak najdokładniej odwzorowuje obraz docelowy, wykorzystując operacje takie jak selekcja, krzyżowanie i mutacja. Proces obliczeń został zoptymalizowany pod kątem użycia GPU z wykorzystaniem biblioteki PyTorch i technologii CUDA.

## Wykorzystane technologie

### **Python**
Podstawowy język programowania wykorzystany do implementacji algorytmu genetycznego oraz operacji na obrazach.

### **NumPy**
- Umożliwia szybkie operacje na tablicach numerycznych.
- Służy do wczytywania obrazów i konwersji ich do formatu tensorów kompatybilnych z PyTorch.

### **Pillow (PIL)**
- Biblioteka do manipulacji obrazami.
- Używana do:
  - Wczytywania obrazów w różnych formatach.
  - Konwersji obrazów do skali szarości (tryb „L”).
  - Zmiany rozmiaru obrazów w celu dostosowania ich do wymagań algorytmu.
  - Zapisu wygenerowanych obrazów w trakcie działania algorytmu.

### **PyTorch**
- Obsługuje operacje tensorowe niezbędne do implementacji funkcji fitness, krzyżowania i mutacji.
- Dostarcza narzędzia do obliczeń na GPU, co znacząco przyspiesza proces ewolucji dla dużych populacji obrazów.
- Funkcje takie jak konwolucje (np. `torch.nn.functional.conv2d`) zostały wykorzystane do obliczania dodatkowych penalizacji związanych z "szumem" obrazu.

### **CUDA**
- Technologia przetwarzania równoległego opracowana przez NVIDIA.
- Używana do:
  - Przyspieszenia obliczeń związanych z oceną fitness i manipulacją populacją.
  - Automatycznego przesyłania danych do GPU za pomocą PyTorch (`.cuda()`).

## Proces działania algorytmu

1. **Inicjalizacja populacji:**
   - Populacja składa się z losowo wygenerowanych obrazów o zadanych wymiarach.
   - Każdy obraz reprezentowany jest jako tensor przechowujący wartości pikseli w skali 0–255.

2. **Ocena fitness:**
   - Każdy osobnik w populacji jest oceniany pod względem podobieństwa do obrazu docelowego.
   - Ocena fitness opiera się na różnicy pikselowej między obrazem osobnika a obrazem docelowym (suma różnic absolutnych).
   - Dodano dodatkową penalizację za obecność "szumu" (nagłe zmiany wartości pikseli w sąsiadujących punktach).

3. **Selekcja:**
   - Wybranych zostaje 20–40% najlepszych osobników na podstawie wartości fitness.

4. **Krzyżowanie:**
   - Z par rodziców (losowo wybranych spośród najlepszych osobników) generowane są nowe obrazy.
   - Każdy piksel nowego obrazu pochodzi z jednego z rodziców w zależności od losowo wygenerowanej maski.

5. **Mutacja:**
   - Każdy piksel obrazu ma niewielką szansę na zmianę swojej wartości na losową w zakresie 0–255.
   - Częstotliwość mutacji jest regulowana parametrem `mutation_rate`.

6. **Iteracje:**
   - Proces selekcji, krzyżowania i mutacji powtarza się przez określoną liczbę generacji lub do momentu osiągnięcia zadowalającego poziomu podobieństwa.

7. **Zapisywanie wyników:**
   - Najlepsze obrazy są zapisywane w trakcie działania algorytmu, umożliwiając śledzenie postępu ewolucji.
   - Cała populacja może być zapisana i załadowana w kolejnych uruchomieniach algorytmu (funkcja wznowienia).

## Parametry techniczne

- **Rozmiar obrazu docelowego:** Obraz docelowy jest skalowany do zadanych wymiarów, co wpływa na wydajność obliczeń i jakość rekonstrukcji.
- **Rozmiar populacji:** Liczba osobników w populacji. Duże populacje zapewniają lepsze pokrycie przestrzeni poszukiwań, ale zwiększają wymagania obliczeniowe.
- **Liczba generacji:** Maksymalna liczba iteracji algorytmu.
- **Częstotliwość mutacji:** Procent pikseli podlegających mutacji w każdej generacji.
- **Top-n selekcja:** Procent najlepszych osobników przenoszonych bezpośrednio do kolejnej generacji.
- **CUDA:** Akceleracja GPU jest używana w każdej fazie obliczeń, znacznie skracając czas przetwarzania.


## Wymagania systemowe

- **Procesor:** Dowolny nowoczesny procesor wielordzeniowy.
- **GPU:** Karta graficzna z obsługą CUDA (zalecane modele NVIDIA).
- **RAM:** Co najmniej 8 GB (więcej dla większych populacji).
- **Biblioteki Python:** NumPy, Pillow, PyTorch.

## Jak uruchomić
 
1) Zainstaluj wymagane biblioteki Python:
```sh
pip install torch
pip install numpy
pip install pillow
pip install opencv-python
```
torcha nalezy zainstlaowac w wersji zgodnej z wersją zainstalowanej CUDY

2) Różne wersje progamu:
 - Aby urchomić odtworzenie w barwach czarnej i białej należy uruchomić skrypt black_and_white.py
 - obraz RGB rgb.py
 - Metoda dziel i rządź dzieląca obraz na mniejsze pod obrazy oraz na trzy przestrzenie barw czerwoną, zieloną i niebieską
```sh
grid_size = (1, 1) #na ile obraz zostanie podzielony 
channel = 0 # 0 red, 1 green, 2 blue, nalezy zmienić w miejscu zapisu pliku i w miejscu zapisu zdjeć docelowych
```
join_rgb.py - łączy trzy obrazy które mają tylko jeden kanał barw w jeden obraz posiadający 3 kanały
alg_join.py - łączy małe fragmenty w jeden duży obraz, należy odpowiednio dobrac rozmiar pojedynczego zdjęcia zdjecia i liczbe grid_size 
make_animation3.py - tworzy film z zdjęć, należy umeiscić skrypt w folderze z zdjęciami
plot.py - rysuje krzywą uczenia dla po generacji skryptem black_and_white.py

