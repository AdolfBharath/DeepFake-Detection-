import os
from typing import Optional, Tuple

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def _build_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            # Light brightness jitter via Lambda to avoid extra deps
            tf.keras.layers.Lambda(lambda x: tf.clip_by_value(
                x + tf.random.uniform(tf.shape(x), minval=-0.05, maxval=0.05, dtype=x.dtype), 0.0, 1.0
            )),
        ],
        name="data_augmentation",
    )


def _preprocess_batch(image, label, image_size: Tuple[int, int]):
    # Ensure float32 [0,1]
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    # EfficientNet preprocess rescales to [-1, 1]
    image = tf.keras.applications.efficientnet.preprocess_input(image * 255.0)
    return image, label


def get_datasets(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = True,
    seed: int = 123,
):
    """
    Creates tf.data datasets from a directory with subfolders:
        train/real, train/fake, val/real, val/fake, [test/real, test/fake]

    Returns: train_ds, val_ds, test_ds (or None), class_names
    """
    data_dir = os.path.abspath(data_dir)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected 'train' and 'val' directories under: {data_dir}"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,  # initial load resize
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    class_names = train_ds.class_names

    test_ds = None
    if os.path.isdir(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )

    aug = _build_augmentation_layer() if augment else None

    def augment_then_preprocess(image, label):
        if aug is not None:
            image = aug(image, training=True)
        # image is in [0,1] after augmentation; convert for EfficientNet
        image = tf.keras.applications.efficientnet.preprocess_input(image * 255.0)
        return image, label

    # Apply augmentation only on train
    if augment:
        train_ds = train_ds.map(augment_then_preprocess, num_parallel_calls=AUTOTUNE)
    else:
        train_ds = train_ds.map(
            lambda x, y: _preprocess_batch(x, y, image_size),
            num_parallel_calls=AUTOTUNE,
        )

    val_ds = val_ds.map(
        lambda x, y: _preprocess_batch(x, y, image_size), num_parallel_calls=AUTOTUNE
    )
    if test_ds is not None:
        test_ds = test_ds.map(
            lambda x, y: _preprocess_batch(x, y, image_size), num_parallel_calls=AUTOTUNE
        )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
