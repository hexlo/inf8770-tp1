import huffman_coding
from huffman_coding import huffman
import generate_histograms

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

import analyze_spatial_redundancy
from analyze_spatial_redundancy import analyze_spatial_redundancy, analyze_spatial_redundancy_rgb


def main():

    IMAGE_1 = 'images/image1_natural.png'
    IMAGE_2 = 'images/image2_synthetic.png'
    IMAGE_3 = 'images/image3_binary.png'

    current_image = IMAGE_1

    # analyze_spatial_redundancy(IMAGE_1)
    # analyze_spatial_redundancy(IMAGE_2)
    # analyze_spatial_redundancy(IMAGE_3)

    # analyze_spatial_redundancy_rgb(IMAGE_1)
    # analyze_spatial_redundancy_rgb(IMAGE_2)
    # analyze_spatial_redundancy_rgb(IMAGE_3)

    huffman(IMAGE_1)
    huffman(IMAGE_2)
    huffman(IMAGE_3)
    
    # Générer les histogrammes de compression
    print("\n" + "="*60)
    print("Génération des histogrammes...")
    print("="*60)
    generate_histograms.generate_compression_histograms([IMAGE_1, IMAGE_2, IMAGE_3])

if __name__ == "__main__":
    main()