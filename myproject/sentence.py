import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import imageio
import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO  # 确保您已安装 ultralytics 库
from torchvision.transforms.functional import resize
from django.conf import settings


def detect_and_crop(image_path):
    # 固定 YOLO 模型路径和填充比例
    yolo_model_path = "yolov8x.pt"  # 请确保该路径正确，指向 YOLOv8 权重文件
    padding = 0.05  # 固定 5% 的填充比例

    # 加载 YOLO 模型
    model = YOLO(yolo_model_path)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法读取图像。")
        return None

    # 使用 YOLOv8 检测物体
    results = model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()
    if len(detections) == 0:
        print("未检测到任何物体。")
        return None

    # 找到检测到的最大物体区域
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in detections]
    largest_index = np.argmax(areas)
    x_min, y_min, x_max, y_max = map(int, detections[largest_index])

    # 对检测框应用填充
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, x_min - int(padding * width))
    y_min = max(0, y_min - int(padding * height))
    x_max = min(image.shape[1], x_max + int(padding * width))
    y_max = min(image.shape[0], y_max + int(padding * height))

    # 裁剪最大物体
    cropped_object = image[y_min:y_max, x_min:x_max]

    # 将图像从 BGR 转为 RGB
    cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)

    # 返回裁剪后的图像
    return cropped_object


def beam_search_decoder(image_path, beam_size=1, smooth=True, attention=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 固定的模型路径和词汇映射文件路径
    # 使用 BASE_DIR 构建文件路径
    checkpoint_path = os.path.join(settings.BASE_DIR,'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq_atten.pth.tar')
    word_map_file_path = os.path.join(settings.BASE_DIR, 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')

    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=str(device))

    encoder = checkpoint['encoder'].to(device).eval()
    decoder = checkpoint['decoder'].to(device).eval()

    # 加载词汇映射文件
    with open(word_map_file_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # 读取并预处理图片
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        img = np.concatenate([img[:, :, np.newaxis]] * 3, axis=2)
    img = np.array(Image.fromarray(img).resize((256, 256))).transpose(2, 0, 1) / 255.0
    img = torch.FloatTensor(img).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(img).unsqueeze(0)  # (1, 3, 256, 256)

    k = beam_size
    vocab_size = len(word_map)

    # 初始化解码所需的张量和变量
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    encoder_out = encoder(image)

    # 初始化解码器的 hidden 和 cell 状态
    if attention:
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim).expand(k, -1, -1)
        h, c = decoder.init_h(encoder_out.mean(dim=1)), decoder.init_c(encoder_out.mean(dim=1))
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)
    else:
        encoder_out = encoder_out.reshape(1, -1).expand(k, -1)
        h, c = decoder.init_h(encoder_out), decoder.init_c(encoder_out)

    # 解码主循环
    complete_seqs = []
    complete_seqs_scores = []
    step = 1
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        if attention:
            scores, alpha, h, c = decoder.one_step(embeddings, encoder_out, h, c)
            alpha = alpha.view(-1, enc_image_size, enc_image_size)
        else:
            scores, h, c = decoder.one_step(embeddings, h, c)
        scores = F.log_softmax(scores, dim=1)
        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        if attention:
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

        k -= len(complete_inds)
        if k == 0 or step > 50:
            break

        seqs = seqs[incomplete_inds]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        if attention:
            seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        step += 1

    # 选出分数最高的句子
    best_seq = complete_seqs[complete_seqs_scores.index(max(complete_seqs_scores))]
    generated_sentence = " ".join(
        [rev_word_map[idx] for idx in best_seq if idx not in {word_map['<start>'], word_map['<end>']}])

    return generated_sentence


import cv2
import numpy as np
import matplotlib.pyplot as plt


def edge_detection(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not read image.")
        return

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用直方图均衡化增强图像对比度
    equalized = cv2.equalizeHist(gray)

    # 使用自适应双边滤波平滑图像，保留更多细节和边缘
    blurred = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)

    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # 仅保留较大的边缘区域，去除噪声
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_edges = np.zeros_like(edges)
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # 仅保留面积较大的轮廓
            cv2.drawContours(filtered_edges, [contour], -1, 255, thickness=cv2.FILLED)

    # 将边缘检测结果缩放到40x60像素
    resized_edges = cv2.resize(filtered_edges, (60, 40), interpolation=cv2.INTER_AREA)

    # 创建一个40x60像素的画面，背景为白色
    white_image = np.ones((40, 60, 3), dtype=np.uint8) * 255

    # 将边缘检测结果叠加到白色背景画面中，边缘为黑色
    for i in range(3):
        white_image[:, :, i] = np.where(resized_edges > 0, 0, white_image[:, :, i])

    # 显示原始图像、边缘检测结果和叠加后的结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Detection")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
    plt.title("40x60 Image with Key Edges")
    plt.axis("off")

    plt.show()
