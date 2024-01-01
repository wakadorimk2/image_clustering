import os
from tqdm import tqdm
from pprint import pformat

import numpy as np
import skimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

# GPUの使用を設定（利用可能な場合）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 画像フォルダのパス
image_folder = '../images'

# 画像のサイズ
image_size = (512, 512, 3)

# 画像の枚数
num_images = len(os.listdir(image_folder)) // 100

# クラスタ数
n_clusters = 10

class Autoencoder(nn.Module):
    """Autoencoderのモデル"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(image_size[0] * image_size[1] * image_size[2], 128),
            nn.ReLU(True))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, image_size[0] * image_size[1] * image_size[2]),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_and_resize_images(folder_path, size=image_size[0:2]):
    """フォルダ内の画像を読み込み、リサイズして返す"""
    images = []
    filenames = []
    progress_tqdm = tqdm(total=num_images, unit='count', desc='Loading images')
    for filename in os.listdir(folder_path)[:num_images]:
        if filename.endswith('.jpg'):
            img = skimage.io.imread(os.path.join(folder_path, filename))
            img_resized = skimage.transform.resize(img, size, anti_aliasing=True)
            images.append(img_resized.flatten())
            filenames.append(filename)
            progress_tqdm.update(1)
    progress_tqdm.close()
    return np.array(images), filenames


def visualize_clusters(images, clusters, n_clusters=n_clusters, num_images=5):
    """クラスタリング結果を画像として表示する"""
    plt.figure(figsize=(15, n_clusters * 3))
    
    for cluster in range(n_clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        selected_indices = cluster_indices[:num_images]
        
        for i, index in enumerate(selected_indices):
            plt.subplot(n_clusters, num_images, cluster * num_images + i + 1)
            plt.imshow(images[index].reshape(512, 512, -1))  # 形状に応じて変更
            plt.axis('off')
            plt.title(f'Cluster {cluster + 1}')

    plt.savefig('../clusters.png')


# 画像データの読み込み
data, filenames = load_and_resize_images(image_folder)

# データをPyTorchテンソルに変換
tensor_data = torch.Tensor(data).to(device)

# データローダーの作成
dataloader = DataLoader(TensorDataset(tensor_data), batch_size=32, shuffle=True)

# Autoencoderのインスタンス化とGPUへの移動
autoencoder = Autoencoder().to(device)

# 訓練ループ（省略：ここに訓練コードを追加）

# エンコードされた特徴の抽出
encoded_features = autoencoder.encoder(tensor_data).cpu().detach().numpy()

# KMeansクラスタリング
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(encoded_features)

# クラスタリング結果の表示
for i in range(n_clusters):
    cluster_indices = np.where(clusters == i)[0]
    print(f"Cluster {i+1}:")
    for index in cluster_indices:
        print(filenames[index])

# 画像データとクラスタリング結果を使って可視化
visualize_clusters(data, clusters)