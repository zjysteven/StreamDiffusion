import os
import sys
import argparse

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn.functional as F
import requests

from diffusers import (
    AutoencoderTiny, LCMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
)

from diffusers.utils import load_image, make_image_grid
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from img_utils import downsample, tensor2img
from PIL import Image

from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['cars', 'other'], default='cars')
    parser.add_argument('--save_dir', type=str, default='clip_exp_images_cars')

    args = parser.parse_args()

    return args

def gen_heatmap(data, row_labels, col_labels, title, ax, cbar_kw=None, cbarlabel="", **kwargs):
    if cbar_kw is None:
        cbar_kw = {}

    # Plot heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Title
    ax.title.set_text(title)

    return im, cbar

def annotate_heatmap(im, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    data = im.get_array()

    # Normalize the threshold to the images color range
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be overwritten by textkw
    kw = dict(horizontalalignment = "center", verticalalignment = "center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a 'Text' for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color = textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def car_experiments(save_dir):
    print('Loading CLIP from clip-vit-large-patch14')
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    #processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    print('Loading some test images...')
    images_blue = []
    images_red = []
    images_blue.append((load_image("blue_car256.png"), "Blue Car"))
    images_red.append((load_image("red_car256.png"), "Red Car"))

    red_titles = ['Red 0']
    blue_titles = ['Blue 0']

    rotations = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360]
    #rotations = [45, 90, 135, 180, 225, 270, 315, 360]

    for i in rotations:
        img_blue = load_image("blue_car256.png")
        img_red = load_image("red_car256.png")
        images_blue.append((img_blue.rotate(i), f'Blue Car - {i} Degree Rotation'))
        images_red.append((img_red.rotate(i), f'Red Car - {i} Degree Rotation'))
        blue_titles.append(f'Blue_{i}')
        red_titles.append(f'Red_{i}')

    # Save images
    tmp_img_blue = []
    tmp_img_red = []
    for img, title in images_blue:
        tmp_img_blue.append(img)
    for img, title in images_red:
        tmp_img_red.append(img)

    grid_output_blue = make_image_grid(
        tmp_img_blue,
        rows=5,
        cols=5
    )
    grid_output_blue.save(f'{save_dir}/blue_cars.png')

    grid_output_red = make_image_grid(
        tmp_img_red,
        rows=5,
        cols=5
    )
    grid_output_red.save(f'{save_dir}/red_cars.png')

    del tmp_img_blue
    del tmp_img_red

    blue_car_embeddings = []
    # Run CLIP for each to generate some embeddings
    for img, title in images_blue:
        inputs = processor(images=img, return_tensors="pt").to("cuda")
        image_features = model.get_image_features(**inputs)
        blue_car_embeddings.append(image_features)
        #print(f'Image {title} --> Embeddings (dim {image_features.shape}): {image_features}\n')

        sim = F.cosine_similarity(image_features, blue_car_embeddings[0])
        minval, minidx = torch.min(image_features, dim=1, keepdim=False)
        maxval, maxidx = torch.max(image_features, dim=1, keepdim=False)
        print(f'Image {title} --> Similarity to original: {sim}\n')
        print(f'\tMax value {maxval} at index {maxidx} -- Min value {minval} at index {minidx}\n\n')

    red_car_embeddings = []
    for img, title in images_red:
        inputs = processor(images=img, return_tensors="pt").to("cuda")
        image_features = model.get_image_features(**inputs)
        red_car_embeddings.append(image_features)
        #print(f'Image {title} --> Embeddings (dim {image_features.shape}): {image_features}\n')

        sim = F.cosine_similarity(image_features, red_car_embeddings[0])
        minval, minidx = torch.min(image_features, dim=1, keepdim=False)
        maxval, maxidx = torch.max(image_features, dim=1, keepdim=False)
        print(f'Image {title} --> Similarity to original: {sim}\n')
        print(f'\tMax value {maxval} at index {maxidx} -- Min value {minval} at index {minidx}\n\n')

    # A whole bunch of plotting....
    plt.ioff()

    for i in range(len(red_car_embeddings)):
        print('Generating plots for red embedding matrices...')
        plt_name = f'{red_titles[i]}_embedding.png'
        plt_title = f'{red_titles[i]} Embedding Matrix'
        plt_dat = np.array(red_car_embeddings[i][0].detach().cpu())

        fig = plt.figure(figsize=(12,10), facecolor='white')
        ax = fig.add_subplot(1, 1, 1, frameon = False)

        ax.set_title(plt_title)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.plot(plt_dat)

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, plt_name))
        plt.close(fig)

    for i in range(len(blue_car_embeddings)):
        print('Generating plots for blue embedding matrices...')
        plt_name = f'{blue_titles[i]}_embedding.png'
        plt_title = f'{blue_titles[i]} Embedding Matrix'
        plt_dat = np.array(blue_car_embeddings[i][0].detach().cpu())

        fig = plt.figure(figsize=(12,10), facecolor='white')
        ax = fig.add_subplot(1, 1, 1, frameon=False)

        ax.set_title(plt_title)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.plot(plt_dat)

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, plt_name))
        plt.close(fig)

    # For comparison - load + get embedding for a white camry
    img_camry = load_image("white_camry256.png")
    camry_inp = processor(images=img_camry, return_tensors="pt").to("cuda")
    camry_features = model.get_image_features(**camry_inp)
    
    fig = plt.figure(figsize=(12,10), facecolor='white')
    ax = fig.add_subplot(1, 1, 1, frameon=False)
    ax.set_title('White Camry Embedding Matrix')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.plot(np.array(camry_features[0].detach().cpu()))

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'white_camry_embedding.png'))
    plt.close(fig)

    # -------- Similarity heatmap between blue + red cars
    sim_matrix = []
    for i in range(len(red_car_embeddings)):
        sim_blue2red = []
        for j in range(len(blue_car_embeddings)):
            print(f'Generating similarity score for image {red_titles[i]} and image {blue_titles[j]}')
            sim = F.cosine_similarity(red_car_embeddings[i], blue_car_embeddings[j])
            sim_blue2red.append(sim.detach().cpu())

            # SANITY CHECK for plotting
            if i == j:
                print(f'\tSANITY CHECK -- Similarity between {red_titles[i]} and {blue_titles[j]}: {sim.detach().cpu()}')

        sim_matrix.append(sim_blue2red)

    sim_matrix = np.array(sim_matrix)

    # Generate heatmap
    plt.ioff()
    figSim = plt.figure(figsize=(18, 20), facecolor='white')
    axSim = figSim.add_subplot(1, 1, 1, frameon=False)

    im, _ = gen_heatmap(sim_matrix, red_titles, blue_titles,
                        "Similarity Matrix for Red and Blue Cars",
                        ax=axSim, cmap="Blues", cbarlabel="Similarity")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")

    figSim.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarities_between_colors.png'))
    plt.close(figSim)

    """
    ## =============================================================================
    # Generate some simple text embeddings, e.g. "red car" "blue car" for comparison
    captions_red = [
        'red car',
        'blue car',
        'red car, facing left',
        'red car, facing down',
        'red car, facing right',
        'red car, facing up',
        'red car, facing up and right',
        'red car, facing up and left',
        'red car, facing down and right',
        'red car, facing down and left'
    ]

    captions_blue = [
        'blue car',
        'red car',
        'blue car, facing left',
        'blue car, facing down',
        'blue car, facing right',
        'blue car, facing up',
        'blue car, facing up and right',
        'blue car, facing up and left',
        'blue car, facing down and right',
        'blue car, facing down and left'
    ]

    red_text_embeddings = []
    blue_text_embeddings = []
    # Run CLIP for each caption to generate some embeddings
    for cap in captions_red:
        inputs = tokenizer(cap, padding=True, return_tensors="pt").to("cuda")
        text_features = model.get_text_features(**inputs)
        red_text_embeddings.append(text_features)
        print(f'Caption: {cap} --> Embeddings: {text_features}\n')

    for cap in captions_blue:
        inputs = tokenizer(cap, padding=True, return_tensors="pt").to("cuda")
        text_features = model.get_text_features(**inputs)
        blue_text_embeddings.append(text_features)
        print(f'Caption: {cap} --> Embeddings: {text_features}\n')


    # Create similarity matrices between captions and images
    red_sim_matrix = []
    blue_sim_matrix = []
    for i in range(len(red_text_embeddings)):
        sim_img2cap = []
        for j in range(len(red_car_embeddings)):
            print(f'Generating similarity score for image {red_titles[j]} and caption {captions_red[i]}')
            sim = F.cosine_similarity(red_text_embeddings[i], red_car_embeddings[j])
            sim_img2cap.append(sim.detach().cpu())

        red_sim_matrix.append(sim_img2cap)

    for i in range(len(blue_text_embeddings)):
        sim_img2cap = []
        for j in range(len(blue_car_embeddings)):
            print(f'Generating similarity score for image {blue_titles[j]} and caption {captions_blue[i]}')
            sim = F.cosine_similarity(blue_text_embeddings[i], blue_car_embeddings[j])
            sim_img2cap.append(sim.detach().cpu())

        blue_sim_matrix.append(sim_img2cap)

    red_sim_matrix = np.array(red_sim_matrix)
    blue_sim_matrix = np.array(blue_sim_matrix)


    # Generate some heatmaps of similarities for image <--> text embeddings
    plt.ioff()
    figRed = plt.figure(figsize=(18, 20), facecolor = 'white')
    axRed = figRed.add_subplot(1, 1, 1, frameon = False)

    im, _ = gen_heatmap(red_sim_matrix, captions_red, red_titles,
                        "Red Car Image-Text Similarity Matrix",
                        ax = axRed, cmap = "Oranges", cbarlabel = "Similarity")
    texts = annotate_heatmap(im, valfmt = "{x:.3f}")

    figRed.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarities_red.png'))
    plt.close(figRed)

    figBlue = plt.figure(figsize=(18, 20), facecolor='white')
    axBlue = figBlue.add_subplot(1, 1, 1, frameon = False)

    im, _ = gen_heatmap(blue_sim_matrix, captions_blue, blue_titles,
                        "Blue Car Image-Text Similarity Matrix",
                        ax = axBlue, cmap = "Blues", cbarlabel = "Similarity")
    texts = annotate_heatmap(im, valfmt = "{x:.3f}")

    figBlue.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarities_blue.png'))
    plt.close(figBlue)


    ## =============================================================================
    # Comparison of some prompts for the cars to see how well CLIP performs
    # at detecting characteristics of images
    test_prompts = [
        'car',
        'blue car',
        'red car',
        'facing left',
        'facing right',
        'truck',
        'cat',
        'dog',
        'human',
        'car on road',
        'car, no background',
        'an image of a car moving left'
    ]

    test_text_embeddings = []

    # Run CLIP for each caption to generate embeddings
    for prompt in test_prompts:
        inputs = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
        text_features = model.get_text_features(**inputs)
        test_text_embeddings.append(text_features)
        print(f'Prompt: {prompt} --> Embeddings: {text_features}\n')

    # Create similarity matrices between prompts and embedding of original car image (red + blue)
    test_sim_matrix = []
    test_car_embeddings = [red_car_embeddings[0], blue_car_embeddings[0]]
    test_titles = [red_titles[0], blue_titles[0]]
    for i in range(len(test_text_embeddings)):
        test_img2cap = []
        for j in range(len(test_car_embeddings)):
            print(f'Generating similarity score for image {test_titles[j]} and prompt {test_prompts[i]}')
            sim = F.cosine_similarity(test_text_embeddings[i], test_car_embeddings[j])
            test_img2cap.append(sim.detach().cpu())

        test_sim_matrix.append(test_img2cap)

    test_sim_matrix = np.array(test_sim_matrix)

    # Generate some heatmaps of similarities for image <--> text embeddings
    plt.ioff()
    figTest = plt.figure(figsize=(8, 20), facecolor = 'white')
    axTest = figTest.add_subplot(1, 1, 1, frameon = False)

    im, _ = gen_heatmap(test_sim_matrix, test_prompts, test_titles,
                        "Car Image-Text Similarity Matrix for Various Prompts",
                        ax = axTest, cmap = "Blues", cbarlabel = "Similarity")
    texts = annotate_heatmap(im, valfmt = "{x:.3f}")

    figTest.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarities_test.png'))
    plt.close(figTest)
    """


def run_tests(save_dir):
    print('Loading CLIP from clip-vit-large-patch14')
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    #processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    print('Loading some test images...')
    img_names = ["bird256.png", "ocean256.png", "person256.png", "pikachu256.png", "simpsons256.png"]
    titles = ["Blue Jay", "Ocean", "Person", "Pikachu", "Homer Simpson"]

    print('Generating embeddings...')
    for i, fname in enumerate(img_names):
        img = load_image(fname)
        inputs = processor(images=img, return_tensors="pt").to("cuda")
        features = model.get_image_features(**inputs)
        minval, minidx = torch.min(features, dim=1, keepdim=False)
        maxval, maxidx = torch.max(features, dim=1, keepdim=False)
        print(f'Image {titles[i]} --> Max value {maxval} at index {maxidx} -- Min value {minval} at index {minidx}\n\n')

        # Plotting
        fig = plt.figure(figsize=(12,10), facecolor='white')
        ax = fig.add_subplot(1, 1, 1, frameon=False)
        ax.set_title(f'{titles[i]} Embedding Vector')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.plot(np.array(features[0].detach().cpu()))

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, fname.split('.')[0] + '_embedding.png'))
        plt.close(fig)


if __name__ == '__main__':
    # Run argument parser
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Dataset
    if args.data == 'cars':
        car_experiments(args.save_dir)
    else:
        run_tests(args.save_dir)