import os
from tqdm import tqdm
from pprint import pformat

import numpy as np
import cupy as cp
import cucim
import skimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 画像フォルダのパス
image_folder = '../images'

# クラスタ数
n_clusters = 10

# 画像を読み込み、リサイズ
def load_and_resize_images(folder_path, size=(256, 256)):
    images = []
    filenames = []
    total_loop = len(os.listdir(folder_path))
    progress_tqdm = tqdm(total=total_loop, unit='count', desc='Loading images')
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            img = skimage.io.imread(os.path.join(folder_path, filename))
            img_resized = skimage.transform.resize(img, size, anti_aliasing=True)
            images.append(img_resized.flatten())
            filenames.append(filename)
            progress_tqdm.update(1)
    progress_tqdm.close()
    return np.array(images), filenames


# クラスタリング結果の可視化
def visualize_clusters(images, clusters, num_clusters=5, num_images=5):
    plt.figure(figsize=(15, num_clusters * 3))
    
    for cluster in range(num_clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        selected_indices = cluster_indices[:num_images]
        
        for i, index in enumerate(selected_indices):
            plt.subplot(num_clusters, num_images, cluster * num_images + i + 1)
            plt.imshow(images[index].reshape(512, 512, -1))  # 形状に応じて変更
            plt.axis('off')
            plt.title(f'Cluster {cluster + 1}')

    plt.savefig('../clusters.png')


# 画像データの読み込み
data, filenames = load_and_resize_images(image_folder)

# KMeansクラスタリング
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(data, verbose=True)

# クラスタリング結果の表示
for i in range(n_clusters):
    cluster_indices = np.where(clusters == i)[0]
    print(f"Cluster {i+1}:")
    for index in cluster_indices:
        print(filenames[index])

# 画像データとクラスタリング結果を使って可視化
visualize_clusters(data, clusters)
