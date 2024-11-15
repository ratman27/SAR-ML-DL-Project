import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# Define constants for image size
IMG_HEIGHT = 256
IMG_WIDTH = 256

def load_image(image_path):
    """Load and preprocess an image, converting grayscale to RGB."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

def load_and_preprocess_image(image_path):
    """Load and preprocess an image, converting grayscale to RGB."""
    img = load_image(image_path)
    img = tf.expand_dims(img, axis=0)  # Create a batch axis
    return img

def load_models(generator_path='generator.h5', discriminator_path='discriminator.h5'):
    """Load the generator and discriminator models."""
    generator = tf.keras.models.load_model(generator_path)
    discriminator = tf.keras.models.load_model(discriminator_path)
    return generator, discriminator

def sharpen_image(image_array, num_iterations=3, blur_radius=1):
    """Apply basic sharpening filter to the image iteratively, then reduce noise."""
    # Rescale the image_array from [-1, 1] to [0, 1]
    image_array = (image_array + 1) / 2
    image_array = np.clip(image_array, 0, 1)  # Clip values to [0, 1]
    img = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert to PIL Image
    
    for _ in range(num_iterations):
        # Apply the basic sharpening filter
        img = img.filter(ImageFilter.SHARPEN)
    
    # Apply Gaussian Blur to reduce noise
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return np.array(img) / 255.0  # Normalize back to [0, 1]

def generate_images(model, test_input):
    """Generate and display images."""
    # Generate prediction
    prediction = model(test_input, training=False)
    prediction = np.squeeze(prediction)  # Remove batch dimension
    normalised_prediction = (prediction + 1) / 2  # Rescale to [0, 1]
    
    # Sharpen the predicted image
    sharpened_prediction = sharpen_image(prediction)
    
    plt.figure(figsize=(12, 6))

    display_list = [test_input[0], normalised_prediction, sharpened_prediction]
    title = ['Input Image', 'Predicted Image', 'Sharpened Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        if i == 0:
            # No clipping for the input image
            plt.imshow((display_list[i] + 1) / 2)  # Rescale back to [0, 1]
        else:
            # Clip values for predicted and sharpened images
            plt.imshow(np.clip(display_list[i], 0, 1))
        plt.axis('off')
    plt.show()

def process_and_display_image(image_path):
    """Load models, preprocess image, generate and display images."""
    # Load models
    generator, _ = load_models()
    
    # Preprocess image
    img_array = load_and_preprocess_image(image_path)
    
    # Generate and display images
    generate_images(generator, img_array)

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    process_and_display_image(image_path)