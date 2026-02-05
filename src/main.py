import huffman_coding
from huffman_coding import huffman

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

import analyze_spatial_redundancy
from analyze_spatial_redundancy import analyze_spatial_redundancy, analyze_spatial_redundancy_rgb

def get_individual_color_channel(image_path, channel_name):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        img = Image.open(image_path)
        img_rgb = img.convert('RGB')

    except Exception as e:
        print(f"Error loading image: {e}")
        return
    

    # print("img_rgb:", img_rgb)
    pixels_rgb = np.array(img_rgb)
    # print("pixels_rgb as np.array:", pixels_rgb)

    match channel_name.lower():
        case 'r' | 'red':
            return pixels_rgb[:, :, 0]
        
        case 'g' | 'green':
            return pixels_rgb[:, :, 1]
        
        case 'b' | 'blue':
            return pixels_rgb[:, :, 2]
        
        case _:
            print('''Possible names for color channels are:''R'', ''G'', ''B'', ''Red'', ''Green'', ''Blue''. Case insensitive.''')
            raise ValueError('Invalid channel name')


def main():

    IMAGE_1 = 'images/image1_natural.png'
    IMAGE_2 = 'images/image2_synthetic.png'
    IMAGE_3 = 'images/image3_binary.png'

    current_image = IMAGE_1
    

    analyze_spatial_redundancy(IMAGE_1)
    analyze_spatial_redundancy(IMAGE_2)
    analyze_spatial_redundancy(IMAGE_3)

    analyze_spatial_redundancy_rgb(IMAGE_1)
    analyze_spatial_redundancy_rgb(IMAGE_2)
    analyze_spatial_redundancy_rgb(IMAGE_3)

    red_channel = get_individual_color_channel(current_image, 'red')
    flattened_red_channel = red_channel.flatten()

    green_channel = get_individual_color_channel(current_image, 'green')
    flattened_green_channel = green_channel.flatten()

    blue_channel = get_individual_color_channel(current_image, 'blue')
    flattened_blue_channel = blue_channel.flatten()

    # Message = "EBFFBEABEFCFDEBBFFFEFEFCCBFACBBABCDACDCABABBBCDDCDCAAABBBABABAABAAB"
    # message_array = np.array(list(Message))

    huffman(flattened_green_channel)

if __name__ == "__main__":
    main()