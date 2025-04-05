import os
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

# Disease-to-medicine mapping with new disease added
DISEASE_MEDICINE_MAP = {
    "healthy": "No treatment needed. Your plant is healthy!",
    "late_blight": "Use Mancozeb or Chlorothalonil-based fungicides.",
    "powdery_mildew": "Apply sulfur or potassium bicarbonate.",
    "leaf_spot": "Use copper-based fungicides.",
    "bacterial_blight": "Spray with Streptomycin or Copper oxychloride.",
    "blast": "Use Saaf 2% fungicide and improve field drainage.",
    "sooty_mould": "Apply starch solution spray or use a horticultural oil.",
    "alternaria_blight_bottle_gourd": "Use fungicides like Chlorothalonil or Copper oxychloride.",
}

# Updated color thresholds with typical colors for various diseases including Alternaria Blight of Bottle Gourd
COLOR_THRESHOLDS = {
    "healthy": (0, 128, 0),  # Green
    "late_blight": (128, 128, 128),  # Grayish
    "powdery_mildew": (200, 200, 200),  # Light gray/white
    "leaf_spot": (150, 75, 0),  # Brownish
    "bacterial_blight": (220, 220, 80), # Yellowish
    "blast": (160, 100, 60),  # Yellow-brownish (typical of blast disease)
    "sooty_mould": (0, 0, 0),  # Black (typical of sooty mould)
    "alternaria_blight_bottle_gourd": (139, 69, 19),  # Dark brown with yellow edges
}

def preprocess_image(image_path):
    """Load the image and resize it for analysis."""
    try:
        if not os.access(image_path, os.R_OK):
            print(f"Permission denied for the file: {image_path}")
            return None

        print(f"Loading image from: {image_path}")
        img = Image.open(image_path).resize((224, 224))
        
        # Convert the image to a TensorFlow tensor
        img_tensor = tf.keras.utils.img_to_array(img)
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
        
        # Display the image using matplotlib
        print("Displaying the input image...")
        plt.imshow(img)
        plt.axis('off')  # Turn off the axis
        plt.show()
        
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def calculate_accuracy(distance, max_distance):
    """Calculate accuracy based on distance."""
    accuracy = max(0, 100 - (distance / max_distance) * 100)
    return round(accuracy, 2)

def analyze_image(image):
    """Analyze the image and predict the disease based on color."""
    try:
        # Convert to RGB
        image = image.convert("RGB")
        pixels = image.getdata()
        avg_color = tuple(sum(x) // len(x) for x in zip(*pixels))  # Average color

        print(f"Average color detected: {avg_color}")

        # Compare average color with thresholds
        closest_disease = "unknown"
        min_distance = float("inf")
        max_distance = 441.67  # Max distance in RGB space: sqrt(255^2 + 255^2 + 255^2)
        for disease, color in COLOR_THRESHOLDS.items():
            distance = sum((a - b) ** 2 for a, b in zip(avg_color, color)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_disease = disease

        # Calculate accuracy
        accuracy = calculate_accuracy(min_distance, max_distance)
        return closest_disease, accuracy
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return "unknown", 0.0

def predict_and_recommend(image_path):
    """Process the image, predict disease, and recommend a solution."""
    image = preprocess_image(image_path)
    if image is None:
        return None

    disease, accuracy = analyze_image(image)
    medicine = DISEASE_MEDICINE_MAP.get(disease, "No treatment found.")

    return {
        "disease": disease,
        "accuracy": accuracy,
        "medicine": medicine,
    }

def main():
    print("Plant Disease Detection System")
    image_path = input("Enter the path to the plant image: ")

    if not os.path.exists(image_path):
        print("Error: The specified file does not exist.")
        return

    print("\nProcessing...")
    result = predict_and_recommend(image_path)

    if result:
        print("\n=== Prediction Result ===")
        print(f"Disease Detected: {result['disease']}")
        print(f"Accuracy: {result['accuracy']}%")
        print(f"Recommended Treatment: {result['medicine']}")
    else:
        print("Prediction failed. Please check the input image or setup.")

if __name__ == "__main__":
    main()
