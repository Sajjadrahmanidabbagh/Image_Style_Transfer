"""
Created on Dec 11 2021
author: Sajjad RD
"""

import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess images
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')

    # Resize image
    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        image = image.resize((int(image.width * scale), int(image.height * scale)))

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Display image
def show_image(tensor, title="Image"):
    image = tensor.cpu().clone().detach().squeeze(0)  # Remove batch dimension
    image = image.numpy().transpose(1, 2, 0)  # Convert to HWC
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
    image = np.clip(image, 0, 1)  # Clip values to valid range
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Feature extraction
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  #Content representation
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix for style representation
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Main style transfer function
def style_transfer(content_image, style_image, model, content_weight=1e4, style_weight=1e-2, num_steps=500):
    # Clone content image as input
    target = content_image.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.003)

    # Extract features
    content_features = get_features(content_image, model)
    style_features = get_features(style_image, model)

    # Compute style Gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for step in range(1, num_steps + 1):
        target_features = get_features(target, model)

        # Compute content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # Compute style loss
        style_loss = 0
        for layer in style_features:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            _, d, h, w = target_features[layer].shape
            style_loss += torch.mean((target_gram - style_gram) ** 2) / (d * h * w)

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Total Loss: {total_loss.item()}")

    return target

# Load pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad = False
vgg = vgg.to("cuda" if torch.cuda.is_available() else "cpu")

# File paths
content_path = "content.jpg"
style_path = "style.jpg"

# Load images
content = load_image(content_path).to("cuda" if torch.cuda.is_available() else "cpu")
style = load_image(style_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Perform style transfer
stylized_image = style_transfer(content, style, vgg)

# Show final image
show_image(stylized_image, title="Stylized Image")

# Save stylized image
output_path = "stylized_image.jpg"
stylized_image = stylized_image.cpu().clone().detach().squeeze(0)
stylized_image = stylized_image.numpy().transpose(1, 2, 0)
stylized_image = stylized_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
stylized_image = (np.clip(stylized_image, 0, 1) * 255).astype(np.uint8)
cv2.imwrite(output_path, cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR))
