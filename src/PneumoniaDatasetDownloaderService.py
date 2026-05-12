import os
import kagglehub

def download_Pneumonia_X_Ray_dataset():
    """
    Downloads the Chest X-Ray Pneumonia dataset if it does not exist,
    otherwise returns the existing cached path.
    """

    dataset_path = kagglehub.dataset_download(
        "paultimothymooney/chest-xray-pneumonia"
    )

    if os.path.exists(dataset_path):
        print("Dataset already exists.")
    else:
        print("Dataset downloaded.")

    return dataset_path
    

def get_split_path(dataset_path, split_name):
    """
    Returns path of train/test/val folder.
    """

    possible_paths = [
        os.path.join(dataset_path, "chest_xray", split_name),
        os.path.join(dataset_path, split_name)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"{split_name} folder not found")


def get_class_paths(split_path):
    """
    Returns paths for NORMAL and PNEUMONIA folders.
    """

    normal_path = os.path.join(split_path, "NORMAL")
    pneumonia_path = os.path.join(split_path, "PNEUMONIA")

    return {
        "NORMAL": normal_path,
        "PNEUMONIA": pneumonia_path
    }


def get_image_paths(folder_path):
    """
    Returns all image file paths in a folder.
    """

    valid_extensions = (".jpg", ".jpeg", ".png")

    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(valid_extensions)
    ]


def load_Pneumonia_X_Ray_dataset():
    """
    Returns all dataset image paths organized by:
    train/test/val -> NORMAL/PNEUMONIA
    """
    
    dataset_path = download_Pneumonia_X_Ray_dataset()

    result = {}

    for split in ["train", "test", "val"]:

        split_path = get_split_path(dataset_path, split)
        class_paths = get_class_paths(split_path)

        result[f"{split.capitalize()}_Normal"] = \
            get_image_paths(class_paths["NORMAL"])

        result[f"{split.capitalize()}_Pneumonia"] = \
            get_image_paths(class_paths["PNEUMONIA"])

    return result
