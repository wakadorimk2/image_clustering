from typing import Any, List
from pprint import pformat

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch

from openai import OpenAI

from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

from sklearn.cluster import KMeans

# GPUの使用を設定（利用可能な場合）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# クラスタ数
n_clusters = 10

# 画像フォルダのパス
image_folder = '../images'

# OpenAI APIのクライアントを作成
client = OpenAI()


def visualize_clusters(image_filenames: List[Any], clusters, n_clusters=n_clusters, num_columns=5):
    """クラスタリング結果を画像として表示する"""
    plt.figure(figsize=(15, n_clusters * 3))
    
    for cluster in range(n_clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        selected_indices = cluster_indices[:num_columns]
        
        for i, index in enumerate(selected_indices):
            plt.subplot(n_clusters, num_columns, cluster * num_columns + i + 1)
            image = Image.open(image_filenames[index])
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Cluster {cluster + 1}')

    plt.savefig('../clusters.png')


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


# 画像の総数とリスト取得
num_images = 0
image_filenames = []
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        num_images += 1
        image_filenames.append(os.path.join(image_folder, filename))

# タグ抽出モデルの構築
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained('FredZhang7/danbooru-tag-generator')
image_tag_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

# タグの抽出
tags_with_images = image_tag_generator(image_filenames)

# 前処理
tags_preprocessed = []
for tag_with_images in tags_with_images:
    for tag_with_image in tag_with_images:
        tags_preprocessed.append(tag_with_image['generated_text'])

# タグのベクトル化
tags_embeddings = []
for tag in tags_preprocessed:
    tags_embeddings.append(get_embedding(tag))

# KMeansクラスタリング
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(tags_embeddings)
print("KMeansクラスタリング完了！")
print("clusters:", clusters)

# 画像データとクラスタリング結果を使って可視化
visualize_clusters(image_filenames, clusters)