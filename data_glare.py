import os
import cv2
import numpy as np
import shutil
import math

def glare_map(center=(3, 3), radius=50, alpha=0.5, new_size=(256, 256)):
    w, h = new_size
    coordinate = [(x, y) for x in range(w) for y in range(h)]  # 生成坐标（x,y）
    fx = []
    for (x, y) in coordinate:
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        fx.append(alpha * np.e ** (-dist / radius))

    return np.array(fx).reshape(w, h)


def get_highlight_mask(y, thr=0.95):
    # 找到高亮区域mask，并且执行膨胀操作
    y = cv2.resize(y, (256, 256))  # 降采样，加速计算
    y = y / y.max()
    mask = (np.clip(y - thr, 0, 1) / (1 - thr) * 255).astype('uint8')
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def get_center(mask):
    # 寻找不规则mask区域的中心点
    centers = []
    contours, cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = mask[:, :, np.newaxis].repeat(3, axis=-1)
    for i in range(len(contours)):
        cnt_new = contours[i]
        M = cv2.moments(cnt_new)  # 计算轮廓的各阶矩,字典形式
        center_y = int(M["m10"] / M["m00"])
        center_x = int(M["m01"] / M["m00"])
        centers.append((center_x, center_y))
        cv2.circle(img, (int(center_x), int(center_y)), radius=1, color=(0, 255, 0), thickness=2)

    # print(centers)
    return centers


def get_glare(centers, radius):
    glare_list = []
    for i in centers:
        # glare += glare_map(center=(i[1], i[0])) ##多个光源的glare 直接叠加
        glare = glare_map(center=i, radius=radius)
        glare_list.append(glare[:, :, np.newaxis])
    glare = np.concatenate(glare_list, axis=-1)

    # softmax_w = np.exp(glare) / np.sum(np.exp(glare), axis=-1, keepdims=True)
    # glare1 = np.sum(glare * softmax_w, axis=-1, keepdims=True)
    # glare1 = glare1.mean(axis=-1, keepdims=True)  ##多个光源的glare 取均值
    glare = glare.max(axis=-1, keepdims=True)  ##多个光源的glare 取最大值
    # glare = np.concatenate([glare1, glare2], axis=-1).max(axis=-1)

    glare = cv2.GaussianBlur(glare, (13, 13), 0)
    return glare


def get_glare_img(im, ratio=0.3, radius=50):
    max_value = im.max()
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0:1]
    # 计算高光区域mask
    mask = get_highlight_mask(y)  # 降采样到256 256，加速计算
    if mask.max() <= 0:
        glare = np.zeros_like(y)
    else:
        # 计算mask区域的中心点
        centers = get_center(mask)
        # 根据每个中心点， 生成glare
        glare = get_glare(centers, radius=radius)
        # 叠加glare到原图
        h, w, c = im.shape
        glare = cv2.resize(glare, (w, h))
        glare = glare[:, :, np.newaxis]
    Y_fusion = y + ratio * glare
    # im_fusion = Y_fusion / y.clip(min=1e-7) * im
    yuv[:, :, 0:1] = Y_fusion / Y_fusion.max() * y.max()
    im_fusion = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    print(max_value, im_fusion.max())
    im_fusion = (im_fusion).clip(max=max_value)
    return im_fusion, glare, mask


if __name__ == "__main__":
    new_size = (512, 512)
    root = r'D:\DATA\aitoneData\ellip_data_x100'
    path = [i for i in os.listdir(root)]
    save_path = r'D:\DATA\aitoneData\glare_data_train_x100'
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    for p in path:
        name = os.path.basename(p)
        print(name)
        # if '20240103_163536' not in p:
        #     continue
        im = cv2.imread(os.path.join(root, p)).astype(np.float32)
        im = cv2.resize(im, (512, 512))
        # im = cv2.resize(im, new_size).astype(np.float32)
        im = im / 255.0
        for ratio in [0.5]:
            for radius in [50]:
                im_fusion, glare, mask = get_glare_img(im, ratio=ratio, radius=radius)
                cv2.imwrite(os.path.join(save_path, name[:-4] + '_mask.png'), mask)
                cv2.imwrite(os.path.join(save_path, name[:-4] + '_glare.png'), (glare.clip(min=0.0, max=1.0)) * 255)
                cv2.imwrite(os.path.join(save_path, name[:-4] + f'_im_fusion_ration-{ratio}_radius-{radius}.png'),
                            im_fusion.clip(min=0.0, max=1.0) * 255)
        cv2.imwrite(os.path.join(save_path, name[:-4] + '_im.png'), im.clip(min=0.0, max=1.0) * 255)

    # cv2.imshow('glare', glare)
    # cv2.imshow('im_fusion', im_fusion.clip(max=1))
    # cv2.imshow('im', im / 255)
    # cv2.imshow('mask', cv2.resize(mask, (512, 512)))
    # cv2.waitKey()
