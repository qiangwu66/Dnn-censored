
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os
from tqdm import tqdm

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)


def create_feature_extractor():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(1024, 1024, 3))
    test_input = tf.keras.Input(shape=(1024, 1024, 3))
    x = base_model(test_input)
    
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    return model


def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [1024, 1024])
    img = preprocess_input(img)
    return img


def create_dataset(image_paths, batch_size):
    """创建tf.data.Dataset数据管道"""
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_ds = image_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return image_ds


def extract_and_save_features(image_folder, output_prefix, batch_size=8):

    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    filenames = [os.path.basename(f) for f in image_files]
    
    model = create_feature_extractor()
    
    dataset = create_dataset(image_files, batch_size)
    
    all_features = []
    for batch_images in tqdm(dataset, desc="提取特征中", unit="batch"):
        batch_features = model.predict(batch_images, verbose=0)
        
        feature_dim = np.prod(batch_features.shape[1:])
        
        batch_features_flat = batch_features.reshape(batch_features.shape[0], feature_dim)
        all_features.append(batch_features_flat)
    
    features_array = np.vstack(all_features)
    
    features_output = f"{output_prefix}_features.npy"
    filenames_output = f"{output_prefix}_filenames.npy"
    
    np.save(features_output, features_array)
    np.save(filenames_output, np.array(filenames, dtype=object))
    

if __name__ == "__main__":

    IMAGE_FOLDER = "D:/~~model_free(new)/Sur_images_data (analysis)/images"
    OUTPUT_PREFIX = "histology_features"
    
    batch_size = 8
    
    extract_and_save_features(IMAGE_FOLDER, OUTPUT_PREFIX, batch_size)
