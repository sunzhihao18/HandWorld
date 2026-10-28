import numpy as np
import torch

from ultralytics import YOLO


def track_hand(frames, hand_detection_model: YOLO, id_persist=False, track_conf=0.55):

    frames_det_result = []

    for frame_index, frame in enumerate(frames):
        frame_det_result = {"0": None, "1": None}

        with torch.no_grad():
            det_result = hand_detection_model.track(
                frame, conf=track_conf, persist=id_persist, verbose=False
            )
            det_result = det_result[0].boxes

            hand_sides = det_result.cls.cpu().numpy().tolist()
            boxes = det_result.xyxy.cpu().numpy()
            confidences = det_result.conf.cpu().numpy()

            if len(boxes) == 0 or det_result.id is None:
                frames_det_result.append(frame_det_result)
                continue

            boxes_with_conf = np.hstack([boxes, confidences[:, None]]).tolist()

            track_ids = det_result.id.cpu().numpy().tolist()
            assert len(boxes_with_conf) == len(track_ids) == len(hand_sides)

            for i in range(len(track_ids)):
                track_id = int(track_ids[i])
                box_with_conf = boxes_with_conf[i]
                is_rhand = str(int(hand_sides[i]))

                if not id_persist:
                    track_id = -1

                hand_result = {
                    "track_id": track_id,
                    "box_with_conf": box_with_conf,
                }

                frame_det_result[is_rhand] = hand_result

            frames_det_result.append(frame_det_result)

    return frames_det_result
