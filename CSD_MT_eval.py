import os
import torch
import cv2
import os.path as osp
import numpy as np
from PIL import Image
from CSD_MT.options import Options
from CSD_MT.model import CSD_MT
from faceutils.face_parsing.model import BiSeNet
import torchvision.transforms as transforms
import faceutils as futils

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# load face_parsing model
n_classes = 19
face_paseing_model = BiSeNet(n_classes=n_classes)
save_pth = osp.join('faceutils/face_parsing/res/cp', '79999_iter.pth')
face_paseing_model.load_state_dict(torch.load(save_pth,map_location='cpu'))
face_paseing_model.eval()

# load makeup transfer model
parser = Options()
opts = parser.parse()
makeup_model = CSD_MT(opts)
ep0, total_it = makeup_model.resume('CSD_MT/weights/CSD_MT.pth')
makeup_model.eval()

def crop_image(image):
    up_ratio = 0.2 / 0.85  # delta_size / face_size
    down_ratio = 0.15 / 0.85  # delta_size / face_size
    width_ratio = 0.2 / 0.85  # delta_size / face_size

    image = Image.fromarray(image)
    face = futils.dlib.detect(image)

    if not face:
        raise ValueError("No face !")

    face_on_image = face[0]

    image, face, crop_face = futils.dlib.crop(image, face_on_image, up_ratio, down_ratio, width_ratio)
    np_image = np.array(image)
    return np_image

def get_face_parsing(x):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = Image.fromarray(x)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        out = face_paseing_model(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return parsing


def split_parse(opts,parse):
    h, w = parse.shape
    result = np.zeros([h, w, opts.semantic_dim])
    result[:, :, 0][np.where(parse == 0)] = 1
    result[:, :, 0][np.where(parse == 16)] = 1
    result[:, :, 0][np.where(parse == 17)] = 1
    result[:, :, 0][np.where(parse == 18)] = 1
    result[:, :, 0][np.where(parse == 9)] = 1
    result[:, :, 1][np.where(parse == 1)] = 1
    result[:, :, 2][np.where(parse == 2)] = 1
    result[:, :, 2][np.where(parse == 3)] = 1
    result[:, :, 3][np.where(parse == 4)] = 1
    result[:, :, 3][np.where(parse == 5)] = 1
    result[:, :, 1][np.where(parse == 6)] = 1
    result[:, :, 4][np.where(parse == 7)] = 1
    result[:, :, 4][np.where(parse == 8)] = 1
    result[:, :, 5][np.where(parse == 10)] = 1
    result[:, :, 6][np.where(parse == 11)] = 1
    result[:, :, 7][np.where(parse == 12)] = 1
    result[:, :, 8][np.where(parse == 13)] = 1
    result[:, :, 9][np.where(parse == 14)] = 1
    result[:, :, 9][np.where(parse == 15)] = 1
    result = np.array(result)
    return result


def local_masks(opts,split_parse):
    h, w, c = split_parse.shape
    all_mask = np.zeros([h, w])
    all_mask[np.where(split_parse[:, :, 0] == 0)] = 1
    all_mask[np.where(split_parse[:, :, 3] == 1)] = 0
    all_mask[np.where(split_parse[:, :, 6] == 1)] = 0
    all_mask = np.expand_dims(all_mask, axis=2)  # Expansion of the dimension
    all_mask = np.concatenate((all_mask, all_mask, all_mask), axis=2)
    return all_mask



def load_data_from_image(non_makeup_img, makeup_img,opts):
    non_makeup_img=crop_image(non_makeup_img)
    makeup_img = crop_image(makeup_img)
    non_makeup_img=cv2.resize(non_makeup_img,(opts.resize_size,opts.resize_size))
    makeup_img = cv2.resize(makeup_img, (opts.resize_size, opts.resize_size))
    non_makeup_parse = get_face_parsing(non_makeup_img)
    non_makeup_parse = cv2.resize(non_makeup_parse, (opts.resize_size, opts.resize_size),interpolation=cv2.INTER_NEAREST)
    makeup_parse = get_face_parsing(makeup_img)
    makeup_parse = cv2.resize(makeup_parse, (opts.resize_size, opts.resize_size),interpolation=cv2.INTER_NEAREST)

    non_makeup_split_parse = split_parse(opts,non_makeup_parse)
    makeup_split_parse = split_parse(opts,makeup_parse)

    non_makeup_all_mask = local_masks(opts,
        non_makeup_split_parse)
    makeup_all_mask = local_masks(opts,
        makeup_split_parse)

    non_makeup_img = non_makeup_img / 127.5 - 1
    non_makeup_img = np.transpose(non_makeup_img, (2, 0, 1))
    non_makeup_split_parse = np.transpose(non_makeup_split_parse, (2, 0, 1))

    makeup_img = makeup_img / 127.5 - 1
    makeup_img = np.transpose(makeup_img, (2, 0, 1))
    makeup_split_parse = np.transpose(makeup_split_parse, (2, 0, 1))

    non_makeup_img=torch.from_numpy(non_makeup_img).type(torch.FloatTensor)
    non_makeup_img = torch.unsqueeze(non_makeup_img, 0)
    non_makeup_split_parse = torch.from_numpy(non_makeup_split_parse).type(torch.FloatTensor)
    non_makeup_split_parse = torch.unsqueeze(non_makeup_split_parse, 0)
    non_makeup_all_mask = np.transpose(non_makeup_all_mask, (2, 0, 1))

    makeup_img = torch.from_numpy(makeup_img).type(torch.FloatTensor)
    makeup_img = torch.unsqueeze(makeup_img, 0)
    makeup_split_parse = torch.from_numpy(makeup_split_parse).type(torch.FloatTensor)
    makeup_split_parse = torch.unsqueeze(makeup_split_parse, 0)
    makeup_all_mask = np.transpose(makeup_all_mask, (2, 0, 1))

    data = {'non_makeup_color_img': non_makeup_img,
            'non_makeup_split_parse':non_makeup_split_parse,
            'non_makeup_all_mask': torch.unsqueeze(torch.from_numpy(non_makeup_all_mask).type(torch.FloatTensor), 0),

            'makeup_color_img': makeup_img,
            'makeup_split_parse': makeup_split_parse,
            'makeup_all_mask': torch.unsqueeze(torch.from_numpy(makeup_all_mask).type(torch.FloatTensor), 0)
            }
    return data


def extract_eye_mask(parsing, expansion=20, upward_bias=20, side_bias=20):
    # 눈 영역 마스크 생성
    
    eye_mask = np.zeros_like(parsing, dtype=np.uint8)
    eye_mask[np.where(parsing == 4)] = 1  # 왼쪽 눈
    eye_mask[np.where(parsing == 5)] = 1  # 오른쪽 눈

    # 기본 확장
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion, expansion))
    expanded_mask = cv2.dilate(eye_mask, kernel, iterations=1)

    upward_mask = np.zeros_like(expanded_mask)
    upward_mask[:-upward_bias, :] = expanded_mask[upward_bias:, :]

    left_mask = np.zeros_like(expanded_mask)
    left_mask[:, :-side_bias] = expanded_mask[:, side_bias:]

    right_mask = np.zeros_like(expanded_mask)
    right_mask[:, side_bias:] = expanded_mask[:, :-side_bias]

    final_mask = np.clip(expanded_mask + upward_mask + left_mask + right_mask, 0, 1)

    return final_mask

def extract_eyebrow_mask(parsing):
    # 눈썹 마스크 생성

    eyebrow_mask = np.zeros_like(parsing, dtype=np.uint8)
    eyebrow_mask[np.where(parsing == 2)] = 1  # 왼쪽 눈썹
    eyebrow_mask[np.where(parsing == 3)] = 1  # 오른쪽 눈썹
    return eyebrow_mask

def extract_lips_mask(parsing):
    # 입술 마스크 생성

    lips_mask = np.zeros_like(parsing, dtype=np.uint8)
    lips_mask[np.where(parsing == 12)] = 1  # 윗입술
    lips_mask[np.where(parsing == 13)] = 1  # 아랫입술
    return lips_mask


def makeup_transfer256(non_makeup_image, makeup_image, alpha_values, regions):
    """
    메이크업 전이 함수: 영역별로 다른 alpha 값을 사용하여 특정 영역에 필터 적용.
    """
    # 메이크업 전이 수행
    data = load_data_from_image(non_makeup_image, makeup_image, opts=opts)
    with torch.no_grad():
        transfer_tensor = makeup_model.test_pair(data)
        transfer_img = transfer_tensor[0].cpu().float().numpy()
        transfer_img = np.transpose((transfer_img / 2 + 0.5) * 255., (1, 2, 0))
        transfer_img = np.clip(transfer_img, 0, 255).astype(np.uint8)

    # 원본 이미지 크기에 맞게 리사이즈
    target_size = (non_makeup_image.shape[1], non_makeup_image.shape[0])
    transfer_img = cv2.resize(transfer_img, target_size, interpolation=cv2.INTER_LINEAR)

    # 얼굴 파싱 및 영역별 마스크 생성
    non_makeup_parse = get_face_parsing(non_makeup_image)
    masks = {
        "eye": extract_eye_mask(non_makeup_parse),
        "eyebrow": extract_eyebrow_mask(non_makeup_parse),
        "lip": extract_lips_mask(non_makeup_parse),
    }

    # 결과 이미지 생성
    result_image = non_makeup_image.astype(np.float32)
    transfer_img = transfer_img.astype(np.float32)

    # 선택된 영역에만 메이크업 적용
    for region in regions:
        mask = masks.get(region, None)
        if mask is not None:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0)
            mask = mask / mask.max()

            alpha = alpha_values.get(region, 1)  # 해당 영역의 alpha 값 가져오기
            for c in range(3):  # RGB 채널별 적용
                result_image[:, :, c] = result_image[:, :, c] * (1 - alpha * mask) + transfer_img[:, :, c] * (
                    alpha * mask
                )

    # 전체 영역에 대한 처리 (regions="all")
    if "all" in regions:
        alpha = alpha_values.get("all", 1)
        for c in range(3):
            result_image[:, :, c] = result_image[:, :, c] * (1 - alpha) + transfer_img[:, :, c] * alpha

    return result_image.astype(np.uint8)
