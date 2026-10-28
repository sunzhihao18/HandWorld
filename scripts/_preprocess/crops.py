import json
import os

from typing import Tuple

import cv2
import numpy as np
import torch

from skimage.filters import gaussian
from torch.utils.data import Dataset


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    if target_aspect_ratio is None:
        return input_shape

    width, height = input_shape
    width_target, height_target = target_aspect_ratio

    if height / width < height_target / width_target:
        height_new = max(width * height_target / width_target, height)
        width_new = width
    else:
        height_new = height
        width_new = max(height * width_target / height_target, width)

    target_shape = np.array([width_new, height_new])

    return target_shape


def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def generate_trans_matrix(
    c_x: float,
    c_y: float,
    src_width: float,
    src_height: float,
    dst_width: float,
    dst_height: float,
    scale: float,
    rot: float,
) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    rot_rad = np.pi * rot / 180

    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def crop_patch(
    image: np.array,
    c_x: float,
    c_y: float,
    bb_width: float,
    bb_height: float,
    patch_width: float,
    patch_height: float,
    scale: float,
    rot: float,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0,
) -> Tuple[np.array, np.array]:
    """
    Crop the input image and return the crop and the corresponding transformation matrix.
    Args:
        image (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans_matrix (np.array): Transformation matrix.
    """
    trans_matrix = generate_trans_matrix(
        c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot
    )

    patch = cv2.warpAffine(
        image,
        trans_matrix,
        (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )

    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (image.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        patch[:, :, 3] = cv2.warpAffine(
            image[:, :, 3],
            trans_matrix,
            (int(patch_width), int(patch_height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    return patch, trans_matrix


def camera_patch_to_image(camera_patch, box_center, box_size, image_size, focal_length=5000.0):
    img_w, img_h = image_size[:, 0], image_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size[:, 0]
    w_2, h_2 = img_w / 2.0, img_h / 2.0

    bs = b * camera_patch[:, 0] + 1e-9

    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + camera_patch[:, 1]
    ty = (2 * (cy - h_2) / bs) + camera_patch[:, 2]

    camera_image = torch.stack([tx, ty, tz], dim=-1)
    return camera_image


class HamerDataset(Dataset):
    def __init__(self, frames, frames_det_result):
        assert len(frames) == len(frames_det_result)

        self.rescale_factor = 1
        self.box_aspect_ratio = [192, 256]
        self.patch_size = 256

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.frames_index = []
        self.frames = []
        self.boxes = []
        self.is_rhand = []

        for frame_index in range(len(frames)):
            frame = frames[frame_index]
            frame_det_result = frames_det_result[frame_index]

            for is_rhand in [0, 1]:
                if frame_det_result[str(is_rhand)] is not None:
                    box_with_conf = frame_det_result[str(is_rhand)]["box_with_conf"]
                    conf = box_with_conf[-1]
                    if conf < 0.6:
                        continue

                    self.frames.append(frame)
                    self.boxes.append(box_with_conf)
                    self.is_rhand.append(is_rhand)
                    self.frames_index.append(frame_index)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        box_with_conf = self.boxes[idx]
        is_rhand = self.is_rhand[idx]

        image_size = torch.tensor([frame.shape[1], frame.shape[0]])
        image_width = frame.shape[1]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not is_rhand:
            frame = cv2.flip(frame, 1)

            _box_with_conf = box_with_conf.copy()
            box_with_conf[0] = image_width - _box_with_conf[2]
            box_with_conf[2] = image_width - _box_with_conf[0]

        box_with_conf = torch.tensor(box_with_conf)

        box_center = (box_with_conf[2:4] + box_with_conf[0:2]) / 2.0
        box_scale = self.rescale_factor * (box_with_conf[2:4] - box_with_conf[0:2]) / 200.0
        box_size = expand_to_aspect_ratio(
            box_scale * 200, target_aspect_ratio=self.box_aspect_ratio
        ).max()

        downsample_factor = (box_size / self.patch_size) / 2.0
        if downsample_factor > 1.1:
            frame = gaussian(
                frame, sigma=(downsample_factor - 1) / 2, channel_axis=2, preserve_range=True
            )

        hand_patch, trans_matrix = crop_patch(
            frame,
            box_center[0],
            box_center[1],
            box_size,
            box_size,
            self.patch_size,
            self.patch_size,
            scale=1.0,
            rot=0,
            border_mode=cv2.BORDER_CONSTANT,
        )

        hand_patch = hand_patch.astype(np.float32) / 255.0
        hand_patch = (hand_patch - self.mean) / self.std
        hand_patch = torch.from_numpy(hand_patch.transpose(2, 0, 1)).to(torch.float32)

        return {
            "img": hand_patch,
            "box_center": box_center,
            "box_size": box_size,
            "img_size": image_size,
        }

    def __len__(self):
        return len(self.frames)
