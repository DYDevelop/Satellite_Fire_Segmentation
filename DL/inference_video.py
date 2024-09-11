import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from torchvision import transforms as T
from tqdm.auto import tqdm
import torchvision
import segmentation_models_pytorch as smp

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 설정
# model = smp.UnetPlusPlus('efficientnet-b0', encoder_weights=None, classes=1, activation=None, in_channels=3)
model = smp.UnetPlusPlus('timm-resnest14d', encoder_weights=None, classes=1, activation=None, in_channels=3)
model.load_state_dict(torch.load('Best_Dice.pt'))
model.to(device)
model.eval()

# 전처리 함수
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.154, 0.199, 0.154], std=[0.065, 0.075, 0.061]),
])

# 배치 크기 조정
BATCH_SIZE = 16  # 더 큰 배치 크기로 설정하여 더 많은 타일을 한 번에 처리

# 타일 단위로 패딩 처리
def pad_to_nearest_multiple(image, tile_size):
    height, width = image.shape[:2]
    new_height = ((height + tile_size - 1) // tile_size) * tile_size
    new_width = ((width + tile_size - 1) // tile_size) * tile_size

    # 패딩
    padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    padded_image[:height, :width] = image
    return padded_image

# 이미지 처리를 위한 클래스 생성 (Dataset 사용)
class ImageTilesDataset(Dataset):
    def __init__(self, image, tile_size=256):
        self.image = pad_to_nearest_multiple(image, tile_size)
        self.tile_size = tile_size
        self.height, self.width = self.image.shape[:2]
        self.num_tiles_x = self.width // tile_size
        self.num_tiles_y = self.height // tile_size

    def __len__(self):
        return self.num_tiles_x * self.num_tiles_y

    def __getitem__(self, idx):
        i = idx // self.num_tiles_x
        j = idx % self.num_tiles_x
        y_start, x_start = i * self.tile_size, j * self.tile_size
        tile = self.image[y_start:y_start+self.tile_size, x_start:x_start+self.tile_size]
        tile = preprocess(tile)
        return tile, i, j

# 타일 처리 및 결과 병합
def process_large_image(image):
    tile_size = 256
    dataset = ImageTilesDataset(image, tile_size)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    padded_image = pad_to_nearest_multiple(image, tile_size)
    height, width = padded_image.shape[:2]

    full_mask = np.zeros((height, width), dtype=np.float32)

    # 배치로 추론
    for batch_imgs, i_vals, j_vals in tqdm(dataloader):
        batch_imgs = batch_imgs.to(device)

        with torch.no_grad():
            outputs = model(batch_imgs)
            outputs = torch.sigmoid(outputs)

        threshold = 0.5
        binary_masks = (outputs > threshold).float()
        pred_masks = binary_masks.cpu().numpy()

        # 타일 위치에 맞춰 병합
        for idx in range(len(i_vals)):
            i = i_vals[idx].item()  # 텐서를 정수로 변환
            j = j_vals[idx].item()  # 텐서를 정수로 변환
            y_start, x_start = i * tile_size, j * tile_size
            pred_mask = np.squeeze(pred_masks[idx])
            full_mask[y_start:y_start+tile_size, x_start:x_start+tile_size] = np.maximum(
                full_mask[y_start:y_start+tile_size, x_start:x_start+tile_size], pred_mask
            )

    # 원본 크기로 자르기
    full_mask = full_mask[:image.shape[0], :image.shape[1]]

    seg = (full_mask * 255).astype(np.uint8)

    # 윤곽선 찾기
    main = image.copy()
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(main, contours, -1, (0, 0, 255), 4)

    return full_mask, main

# 비디오 처리 함수
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_mask, contoured = process_large_image(frame_rgb)

        pred_mask_rgb = (pred_mask * 255).astype(np.uint8)
        pred_mask_rgb = cv2.cvtColor(pred_mask_rgb, cv2.COLOR_GRAY2RGB)

        frame_rgb = frame_rgb.astype(np.uint8)
        contoured = contoured.astype(np.uint8)

        result = np.hstack((frame_rgb, pred_mask_rgb, contoured))
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        cv2.imshow('Segmentation Result', result_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 비디오 처리 실행
process_video('Fire_demo.mp4')