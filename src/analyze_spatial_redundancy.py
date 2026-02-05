import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

def analyze_spatial_redundancy(image_path):
    """
    Analyse de la redondance spatiale d'une image pour décrire ses caractéristiques principales.
    Visualisation de la distribution des pixels pour évaluer les zones de haute ou basse complexité.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    pixels = np.array(img)

    print("Image size")
    print(" - Height:", len(pixels), "pixels")
    print(" - Width:", len(pixels[0]), "pixels")
    
    # On applati l'array en 1D
    flat_pixels = pixels.flatten()
    
    # Calcule de l'entropie
    hist_counts, _ = np.histogram(flat_pixels, bins=256, range=(0, 256))
    probabilities = hist_counts / np.sum(hist_counts)
    probabilities = probabilities[probabilities > 0] # Eviter log(0)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # print("hist_counts:", hist_counts)
    print("hist_counts lenght:", len(hist_counts))
    print("prob total:", np.sum(hist_counts))
    # print("probabilities:", probabilities)
    print("entropy:", entropy)

    # On creer 2 array décalés de 1 pour les comparer Pixel[i] with Pixel[i+1]
    # ça nous permet d'analyser la correlation entre les pixels adjescents
    x = flat_pixels[:-1]
    y = flat_pixels[1:]
    
    # Calcule le coéfficient de corrélation
    # On prends l'élément de la rangé 0, colomne 1 (ou )
    correlation = np.corrcoef(x, y)[0, 1] if len(x) > 0 else 0
    print("correlation matrix:", np.corrcoef(x, y))

    print(f"Analysis for image: {os.path.basename(image_path)}")
    print(f"  - Entropy: {entropy:.4f} bits/pixel")
    print(f"  - Spatial Correlation: {correlation:.4f}")

    # On génère une 'Complexity Map' à partir des gradients
    # On calcule la difference absolue entre les pixels adjescents pour trouver les bords des transitions
    pixels_float = pixels.astype(float)
    grad_y = np.abs(pixels_float[1:, :] - pixels_float[:-1, :])
    grad_x = np.abs(pixels_float[:, 1:] - pixels_float[:, :-1])
    
    complexity_map = np.zeros_like(pixels_float)
    complexity_map[:-1, :] += grad_y
    complexity_map[:, :-1] += grad_x

    # Visualisation
    plt.figure(figsize=(12, 10))
    
    # Graph 1: Histogramme des valeurs des pixels
    plt.subplot(2, 2, 1)
    plt.hist(flat_pixels, bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(f"Histogram (Entropy: {entropy:.2f} bits)")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence")


    # Graph 2: 'Scatter plot' des pixels adjescents (Visualisation de la redondance spatiale)
    plt.subplot(2, 2, 2)
    plt.scatter(x, y, s=0.5, alpha=0.5)
    plt.title(f"Corrélation Spatiale(r={correlation:.2f})")
    plt.xlabel("Pixel[i]")
    plt.ylabel("Pixel[i+1]")

    # Graph 3: Image Originale
    plt.subplot(2, 2, 3)
    plt.imshow(pixels, cmap='gray')
    plt.title("Image Originale")
    plt.axis('off')

    # Graph 4: Complexity Map
    plt.subplot(2, 2, 4)
    plt.imshow(complexity_map, cmap='hot')
    plt.title("Zones de Complexité (Gradients)")
    plt.axis('off')

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"analyse_{os.path.basename(image_path)}"))
    
    # plt.show()

def analyze_spatial_redundancy_rgb(image_path):
    """
    Analyse la redondance spatiale d'une image pour chacun des cannaux RGB.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        img = Image.open(image_path)
        img_rgb = img.convert('RGB')
        img_gray = img.convert('L')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    pixels_rgb = np.array(img_rgb)
    pixels_gray = np.array(img_gray)

    print("pixels_rgb dimensions: ", pixels_rgb.ndim)
    print("dim 1:", len(pixels_rgb))
    print("dim 2:", len(pixels_rgb[0]))
    print("dim 3:", len(pixels_rgb[0][0]))

    print(f"Analyse RGB pour l'image: {os.path.basename(image_path)}")
    
    colors = ['Red', 'Green', 'Blue']
    plot_colors = ['red', 'green', 'blue']
    
    plt.figure(figsize=(12, 10))

    # Graph 1: RGB Histogrammes
    plt.subplot(2, 2, 1)
    for i in range(3):
        chan_pixels = pixels_rgb[:, :, i].flatten()
        
        # Entropie
        counts, _ = np.histogram(chan_pixels, bins=256, range=(0, 256))
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        
        # Correlation
        x = chan_pixels[:-1]
        y = chan_pixels[1:]
        corr = np.corrcoef(x, y)[0, 1] if len(x) > 0 else 0
        
        print(f"  - {colors[i]} Channel: Entropy={entropy:.4f} bits/pixel, Correlation={corr:.4f}")
        
        plt.hist(chan_pixels, bins=256, range=(0, 256), color=plot_colors[i], alpha=0.3, label=colors[i])

    plt.title("Histogrammes RGB")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence")
    plt.legend()

    # Graph 2: Corrélation Spatiale (Luminance)
    flat_gray = pixels_gray.flatten()
    x = flat_gray[:-1]
    y = flat_gray[1:]
    corr_gray = np.corrcoef(x, y)[0, 1] if len(x) > 0 else 0
    
    plt.subplot(2, 2, 2)
    plt.scatter(x, y, s=0.5, alpha=0.5, c='black')
    plt.title(f"Corrélation Spatiale (Luminance)r={corr_gray:.2f}")
    plt.xlabel("Pixel[i]")
    plt.ylabel("Pixel[i+1]")

    # Graph 3: Image Originale (RGB)
    plt.subplot(2, 2, 3)
    plt.imshow(pixels_rgb)
    plt.title("Image Originale (RGB)")
    plt.axis('off')

    # Graph 4: Complexity Map (Gradients sur Grayscale)
    pixels_float = pixels_gray.astype(float)
    grad_y = np.abs(pixels_float[1:, :] - pixels_float[:-1, :])
    grad_x = np.abs(pixels_float[:, 1:] - pixels_float[:, :-1])
    
    complexity_map = np.zeros_like(pixels_float)
    complexity_map[:-1, :] += grad_y
    complexity_map[:, :-1] += grad_x
    
    plt.subplot(2, 2, 4)
    plt.imshow(complexity_map, cmap='hot')
    plt.title("Zones de Complexité (Gradients sur Grayscale)")
    plt.axis('off')

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"analyse_rgb_{os.path.basename(image_path)}"))
    
    # plt.show()