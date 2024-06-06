import os
import random
import openai
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Set OpenAI API key
openai.api_key = 'API Key'

# Load your model
model = tf.keras.models.load_model('model/flower_classification.h5')

# Flower categories
categories = ["Lilly", "Lotus", "Orchid", "Sunflower", "Tulip"]

# Function to classify images
def classify_images(images, model):
    predictions = []
    for img in images:
        img_array = img_to_array(img.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        predictions.append(categories[class_idx])
    return predictions

# Function to generate images using OpenAI
def generate_images(prompt):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url = response.data[0].url
    return image_url

# Function to download an image from a URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

# Function to get the next version number for the PDF
def get_next_version(subfolder, filename_prefix):
    existing_files = [f for f in os.listdir(subfolder) if f.startswith(filename_prefix) and f.endswith('.pdf')]
    if not existing_files:
        return 1
    version_numbers = [int(f.split('_v')[-1].split('.pdf')[0]) for f in existing_files]
    return max(version_numbers) + 1

# Main script
def main():
    selected_category = random.choice(categories)
    print(f"Selected category for testing: {selected_category}")

    # Prepare to save results
    fig, axs = plt.subplots(5, 2, figsize=(15, 25))
    fig.suptitle(f'Generated Images and Predictions for Category: {selected_category}', fontsize=20)

    images = []
    predictions = []

    # Generate, download, classify, and display 10 images
    for i in range(10):
        prompt = f"A beautiful photo of a {selected_category}"
        image_url = generate_images(prompt)
        downloaded_image = download_image(image_url)
        prediction = classify_images([downloaded_image], model)[0]

        images.append(downloaded_image)
        predictions.append(prediction)

        # Save image and prediction in the document
        ax = axs[i // 2, i % 2]
        ax.imshow(downloaded_image)
        ax.axis('off')
        ax.set_title(f'Prediction: {prediction}', fontsize=15)

        print(f"Image {i+1} URL: {image_url}, Predicted Class: {prediction}")

    # Ensure the subfolder exists
    subfolder = 'Validation/test_outputs'
    os.makedirs(subfolder, exist_ok=True)

    # Save the document with all images and predictions
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    filename_prefix = 'validation_results'
    version = get_next_version(subfolder, filename_prefix)
    filename = os.path.join(subfolder, f'{filename_prefix}_v{version}.pdf')
    
    plt.savefig(filename)
    plt.show()

    print(f'Results saved to {filename}')

if __name__ == "__main__":
    main()
