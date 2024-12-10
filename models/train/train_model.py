from ultralytics import YOLO
import shutil
import random
from pathlib import Path

images_input_dir = Path("dataset/images/")
labels_input_dir = Path("dataset/labels/")

images_output_dir = Path("datasets/images/")
labels_output_dir = Path("datasets/labels/")

train_images_dir = images_output_dir / "train"
val_images_dir = images_output_dir / "val"
train_labels_dir = labels_output_dir / "train"
val_labels_dir = labels_output_dir / "val"


def setup_directories():
    # overwrite output dirs if they exist
    for directory in [
        train_images_dir,
        val_images_dir,
        train_labels_dir,
        val_labels_dir,
    ]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def image_path_to_label_path(image_path):
    file_name = image_path.stem
    file_name += ".txt"
    return labels_input_dir / file_name


dataset = [
    (image_path, image_path_to_label_path(image_path))
    for image_path in images_input_dir.iterdir()
]

random.shuffle(dataset)

split_idx = int(len(dataset) * 0.8)
train_set = dataset[:split_idx]
val_set = dataset[split_idx:]


def move_files(file_list, image_dest, label_dest):
    for image_path, label_path in file_list:
        if image_path.exists():
            shutil.copy(str(image_path), image_dest / image_path.name)
        if label_path.exists():
            shutil.copy(str(label_path), label_dest / label_path.name)


setup_directories()

move_files(train_set, train_images_dir, train_labels_dir)
move_files(val_set, val_images_dir, val_labels_dir)

print(
    f"Dataset sorted successfully:\nNumber of training samples: {len(train_set)}\nNumber of validation samples: {len(val_set)}"
)

model = YOLO("yolo11l.pt")
model.train(data="dataset.yaml", epochs=175, batch=0.65, imgsz=(1280, 720))
model.export(imgsz=(1280, 720), half=True, dynamic=True, simplify=True)
