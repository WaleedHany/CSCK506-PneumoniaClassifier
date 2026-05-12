import os
import cv2    # Open-cv for image processing
import shutil # For file and folder operations
from sklearn.model_selection import train_test_split


# Unified image size (Can be changed later)
TARGET_SIZE = 224


def create_directory(path):
    """
    Create a directory if it does not already exist.
    Parameters:
        path (str):
            Directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def preprocess_image(image_path):
    """
    Load and preprocess a single image.
    Preprocessing steps:
    1. Load image
    2. Perform center crop to obtain square image
    3. Resize image to fixed dimensions
    4. Normalize pixel values to range [0, 1]

    Parameters:
        image_path (str):
            Path of the image file.

    Returns:
        numpy.ndarray:
            Preprocessed image array.
        None:
            Returned if image loading fails.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # return None if image is None
    if image is None:
        return None

    height, width = image.shape

    # Determine smallest dimension in the image
    min_dimension = min(height, width)

    # Compute center crop coordinates
    start_x = (width - min_dimension) // 2   # devide and floor the value
    start_y = (height - min_dimension) // 2  # devide and floor the value

    # Crop square region from center
    cropped = image[
        start_y:start_y + min_dimension,
        start_x:start_x + min_dimension
    ]

    # Resize image to target dimensions
    resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))

    # Normalize pixel values
    normalizedImages = resized #/ 255.0

    return normalizedImages


def save_processed_image(image, output_path):
    """
    Save a processed image to disk.

    Parameters:
        image (numpy.ndarray):
            Normalized image array.
        output_path (str):
            Output file path.
    """
    # Convert normalized image back to uint8
    image_to_save = (image * 255).astype("uint8")
    cv2.imwrite(output_path, image_to_save)


def process_and_save_images(image_paths, output_folder):
    """
    Preprocess and save multiple images.

    Parameters:
        image_paths (list):
            List of input image paths.
        output_folder (str):
            Destination folder for processed images.
    """
    # Create directory
    create_directory(output_folder)
    # For each images folder path, process the images and save the new images to a new local path
    for image_path in image_paths:
        processed = preprocess_image(image_path)
        if processed is None:
            continue
        filename = os.path.basename(image_path)
        output_path = os.path.join(
            output_folder,
            filename
        )
        save_processed_image(processed, output_path)


def process_dataset(dataset):
    """
    Rebuild and preprocess the dataset.

    1. Remove previous processed dataset if it exists
    2. Combine old validation and testing sets
    3. Create new validation/testing split
    4. Preprocess all images
    5. Save processed images to local folders

    Generated folder structure:
        processed_data/
            train/
            val/
            test/

    Parameters:
        dataset (dict):
            Dictionary containing original dataset paths.
    """
    processed_root = "processed_data"

    # If old processed images exist
    # Remove old processed dataset
    if os.path.exists(processed_root):
        shutil.rmtree(processed_root)

    # Combine old testing and validation data
    combined_train_val_normal = (dataset["Train_Normal"] + dataset["Val_Normal"])
    combined_train_val_pneumonia = (dataset["Train_Pneumonia"] + dataset["Val_Pneumonia"])

    # Create new validation/testing split
    train_normal, val_normal = train_test_split(combined_train_val_normal, test_size=0.1, random_state=42)
    train_pneumonia, val_pneumonia = train_test_split(combined_train_val_pneumonia, test_size=0.1, random_state=42)

    # Dataset structure definition
    splits = {
        "train/NORMAL": train_normal,
        "train/PNEUMONIA": train_pneumonia,
        "val/NORMAL": val_normal,
        "val/PNEUMONIA": val_pneumonia,
        "test/NORMAL": dataset["Test_Normal"],
        "test/PNEUMONIA": dataset["Test_Pneumonia"]
    }

    # Process each dataset images split
    for relative_path, image_paths in splits.items():
        output_folder = os.path.join(processed_root, relative_path)
        print(f"Processing: {relative_path}")
        process_and_save_images(image_paths, output_folder)

    print()
    print("Dataset preprocessing completed.")


def load_processed_dataset(processed_root="processed_data"):
    """
    Load the processed dataset.

    Expected folder structure:
        processed_data/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/

    Parameters:
        processed_root (str):
            Root folder containing processed dataset.
    Returns:
        tuple:
            (
                complete_dataset,
                training_dataset,
                validation_dataset,
                testing_dataset
            )
    """
    complete_dataset = {}

    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]

    for split in splits:

        for class_name in classes:

            folder_path = os.path.join(
                processed_root,
                split,
                class_name
            )

            image_paths = []

            if os.path.exists(folder_path):

                for file_name in os.listdir(folder_path):

                    file_path = os.path.join(
                        folder_path,
                        file_name
                    )

                    if os.path.isfile(file_path):
                        image_paths.append(file_path)

            key = f"{split.capitalize()}_{class_name.capitalize()}"

            complete_dataset[key] = image_paths

    training_dataset = {
        "Normal":
            complete_dataset["Train_Normal"],

        "Pneumonia":
            complete_dataset["Train_Pneumonia"]
    }

    validation_dataset = {
        "Normal":
            complete_dataset["Val_Normal"],

        "Pneumonia":
            complete_dataset["Val_Pneumonia"]
    }

    testing_dataset = {
        "Normal":
            complete_dataset["Test_Normal"],

        "Pneumonia":
            complete_dataset["Test_Pneumonia"]
    }

    return (
        complete_dataset,
        training_dataset,
        validation_dataset,
        testing_dataset
    )


def get_labeled_datasets(training_dataset, validation_dataset, testing_dataset):
    """
    Convert training, validation, and testing datasets
    into image paths and numeric labels.

    Labels:
        0 -> Normal
        1 -> Pneumonia

    Parameters:
        training_dataset (dict):
            Training dataset dictionary.
        validation_dataset (dict):
            Validation dataset dictionary.
        testing_dataset (dict):
            Testing dataset dictionary.

    Returns:
        tuple:
            (
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test
            )
    """
    def process_split(dataset_split):
        image_paths = []
        labels = []
        # Normal images
        for path in dataset_split["Normal"]:
            image_paths.append(path)
            labels.append(0)
        # Pneumonia images
        for path in dataset_split["Pneumonia"]:
            image_paths.append(path)
            labels.append(1)

        return image_paths, labels

    X_train, y_train = process_split(training_dataset)

    X_val, y_val = process_split(validation_dataset)

    X_test, y_test = process_split(testing_dataset)

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )