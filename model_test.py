import cv2
import torch
import numpy as np
from collections import deque
from transformers import DFineForObjectDetection, AutoImageProcessor
from ultralytics import YOLO
from PIL import Image
import subprocess
import os
from tqdm import tqdm
import time
import math

video_path = r"C:/Users/Amr Hekal/Desktop/Paddle test/B001_10282233_C003.mov"
processed_path = r"C:/Users/Amr Hekal/Desktop/Paddle test/B001_10282233_C003_match_processed_highlights_2.mp4"
final_path = r"C:/Users/Amr Hekal/Desktop/Paddle test/B001_10282233_C003_match_final_highlights.mp4"
#reference_path = r"C:/Users/Amr Hekal/Desktop/mahmoud test 4/points.csv"
audio_temp = "temp_audio.aac"
log_path = "highlights.log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco", use_fast=True)
dfine_model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
dfine_model.to(device)
dfine_model.eval()

# Use YOLOv11 medium pose model as requested (ensure the weight file/name is available)
pose_model = YOLO("yolo11m-pose.pt")
if device_str == "cuda":
    pose_model.to("cuda")

cap = cv2.VideoCapture(video_path)
# original capture dimensions
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# after rotating right 90deg, width/height swap
width = orig_height
height = orig_width

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_size = (width, height)
out = cv2.VideoWriter(processed_path, fourcc, fps, out_size)

# rotate mode (cv2.ROTATE_90_CLOCKWISE rotates to the right)
ROTATE_MODE = cv2.ROTATE_90_CLOCKWISE

detect_every_n = 20
pose_img_size = 640
bbox_min_score = 0.3
kp_min_conf = 0.25
highlight_seconds = 2.0
highlight_hold_frames = int(highlight_seconds * fps)

skeleton = [
    (5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)
]

def detect_dfine_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dfine_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=bbox_min_score)[0]
    persons = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = dfine_model.config.id2label[label_id.item()].lower()
        if label == "person":
            b = box.int().tolist()
            persons.append({"bbox": b, "score": float(score.cpu().item())})
    return persons

def crop_and_run_pose(frame, person):
    x1, y1, x2, y2 = person["bbox"]
    x1, x2 = max(0, int(x1)), min(width - 1, int(x2))
    y1, y2 = max(0, int(y1)), min(height - 1, int(y2))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad = int(0.05 * max(w, h))
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(width, x2 + pad)
    cy2 = min(height, y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return None

    results = pose_model.predict(source=crop, imgsz=pose_img_size, device=device_str, conf=0.25, verbose=False)
    if len(results) == 0:
        return None
    res = results[0]

    # Try multiple ways to extract keypoints tensor/array robustly
    kp_obj = None
    if hasattr(res, "keypoints") and res.keypoints is not None:
        kp_obj = res.keypoints
        if hasattr(kp_obj, "data"):
            try:
                kp_obj = kp_obj.data
            except Exception:
                pass

    if kp_obj is None:
        try:
            kp_obj = getattr(res, "keypoints", None)
        except Exception:
            kp_obj = None

    if kp_obj is None:
        return None

    try:
        if hasattr(kp_obj, "cpu") and hasattr(kp_obj, "numpy"):
            arr = kp_obj.cpu().numpy()
        else:
            arr = np.asarray(kp_obj)
    except Exception as e:
        print("Warning: couldn't convert keypoints object to numpy:", type(kp_obj), e)
        return None

    if arr.size == 0:
        return None

    parsed = None
    if arr.ndim == 3 and arr.shape[2] == 3:
        parsed = arr
    elif arr.ndim == 2:
        if arr.shape[1] == 3:
            parsed = arr[np.newaxis, :, :]
        elif arr.shape[1] % 3 == 0:
            nk = arr.shape[1] // 3
            parsed = arr.reshape(arr.shape[0], nk, 3)
        else:
            print("Warning: unexpected kp 2D shape", arr.shape)
            return None
    elif arr.ndim == 1:
        if arr.size % 3 == 0:
            parsed = arr.reshape(1, arr.size // 3, 3)
        else:
            print("Warning: unexpected flattened kp shape", arr.shape)
            return None
    else:
        try:
            tmp = []
            for inst in arr:
                inst_a = np.asarray(inst)
                if inst_a.ndim == 2 and inst_a.shape[1] == 3:
                    tmp.append(inst_a)
                elif inst_a.ndim == 1 and inst_a.size % 3 == 0:
                    tmp.append(inst_a.reshape(-1, 3))
                else:
                    tmp = []
                    break
            if len(tmp) > 0:
                parsed = np.stack(tmp, axis=0)
            else:
                print("Warning: couldn't parse object-array keypoints, dtype/object contents unexpected.")
                return None
        except Exception as e:
            print("Warning: exception parsing object-array keypoints:", e)
            return None

    if parsed is None or parsed.size == 0:
        return None

    mapped = []
    for inst in parsed:
        pts = []
        for elem in inst:
            if len(elem) < 3:
                continue
            x, y, c = float(elem[0]), float(elem[1]), float(elem[2])
            gx = x + cx1
            gy = y + cy1
            pts.append((gx, gy, c))
        if len(pts) > 0:
            mapped.append(pts)

    if len(mapped) == 0:
        return None

    if len(mapped) == 1:
        return mapped[0]
    else:
        person_cx = (x1 + x2) / 2.0
        person_cy = (y1 + y2) / 2.0
        best_idx = 0
        best_dist = float("inf")
        for i, inst in enumerate(mapped):
            xs = [p[0] for p in inst]
            ys = [p[1] for p in inst]
            if len(xs) == 0:
                continue
            cx_k = sum(xs) / len(xs)
            cy_k = sum(ys) / len(ys)
            d = math.hypot(cx_k - person_cx, cy_k - person_cy)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return mapped[best_idx]

def both_hands_raised(kp):
    if kp is None or len(kp) < 17:
        return False
    L_sh_y = kp[5][1]; R_sh_y = kp[6][1]
    L_w_y = kp[9][1]; R_w_y = kp[10][1]
    L_w_c = kp[9][2]; R_w_c = kp[10][2]
    if L_w_c < kp_min_conf or R_w_c < kp_min_conf:
        return False
    if L_w_y < L_sh_y - 5 and R_w_y < R_sh_y - 5:
        return True
    return False

person_id_counter = 0
active_people = {}
last_detection = {}
highlight_states = {}
led_state = {"frames_left": 0}
match_distance_threshold = 100

def match_persons(detections, prev_people):
    global person_id_counter
    assigned = {}
    cur_people = {}
    centers = [((d["bbox"][0]+d["bbox"][2])/2, (d["bbox"][1]+d["bbox"][3])/2) for d in detections]
    prev_ids = list(prev_people.keys())
    prev_centers = [((prev_people[i]["bbox"][0]+prev_people[i]["bbox"][2])/2, (prev_people[i]["bbox"][1]+prev_people[i]["bbox"][3])/2) for i in prev_ids] if prev_ids else []
    used_prev = set()
    for di, d in enumerate(detections):
        best_id = None; best_dist = 1e9
        for pi, pid in enumerate(prev_ids):
            if pid in used_prev:
                continue
            pc = prev_centers[pi]
            dist = math.hypot(pc[0]-centers[di][0], pc[1]-centers[di][1])
            if dist < best_dist:
                best_dist = dist; best_id = pid
        if best_id is not None and best_dist < match_distance_threshold:
            used_prev.add(best_id)
            cur_people[best_id] = {"bbox": d["bbox"], "score": d["score"]}
        else:
            person_id_counter += 1
            cur_people[person_id_counter] = {"bbox": d["bbox"], "score": d["score"]}
    return cur_people

for_frame_kp = {}
frame_count = 0

if os.path.exists(log_path):
    os.remove(log_path)

for _ in tqdm(range(total_frames), desc="Main loop"):
    ret, frame = cap.read()
    if not ret:
        break

    # rotate frame right by 90 degrees before any processing
    frame = cv2.rotate(frame, ROTATE_MODE)

    if frame_count % detect_every_n == 0:
        persons = detect_dfine_frame(frame)
        matched = match_persons(persons, active_people)
        active_people = matched
        for pid in list(active_people.keys()):
            p = active_people[pid]
            kp = crop_and_run_pose(frame, p)
            for_frame_kp[pid] = kp
            last_detection[pid] = {"bbox": p["bbox"], "kp": kp, "last_seen": frame_count}
    else:
        for pid in list(active_people.keys()):
            if pid in last_detection:
                active_people[pid]["bbox"] = last_detection[pid]["bbox"]
                for_frame_kp[pid] = last_detection[pid]["kp"]
    for pid in list(active_people.keys()):
        kp = for_frame_kp.get(pid, None)
        bbox = active_people[pid]["bbox"]
        triggered = both_hands_raised(kp)
        if triggered:
            highlight_states[pid] = highlight_hold_frames
            led_state["frames_left"] = max(led_state["frames_left"], highlight_hold_frames)
            with open(log_path, "a") as lf:
                tsec = frame_count / fps
                lf.write(f"frame,{frame_count},{tsec:.3f}s,person_id,{pid}")
        if pid in highlight_states and highlight_states[pid] > 0:
            highlight_states[pid] -= 1
    if led_state["frames_left"] > 0:
        led_state["frames_left"] -= 1
    vis = frame.copy()
    for pid, info in active_people.items():
        b = info["bbox"]
        x1,y1,x2,y2 = map(int, b)
        kp = for_frame_kp.get(pid, None)
        color = (0,0,255)
        if pid in highlight_states and highlight_states[pid] > 0:
            color = (0,255,0)
        thickness = 4
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, thickness)
        label = f"Player {pid}"
        ((tx, ty), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(vis, (x1, y1 - tx - 8), (x1 + tx + 6, y1), color, -1)
        cv2.putText(vis, label, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
        if kp is not None:
            for (i,j) in skeleton:
                if i < len(kp) and j < len(kp):
                    if kp[i][2] > kp_min_conf and kp[j][2] > kp_min_conf:
                        pt1 = (int(kp[i][0]), int(kp[i][1]))
                        pt2 = (int(kp[j][0]), int(kp[j][1]))
                        cv2.line(vis, pt1, pt2, color, thickness)
            for idx, (x,y,c) in enumerate(kp):
                if c > kp_min_conf:
                    cv2.circle(vis, (int(x), int(y)), 6, color, -1)
            if pid in highlight_states and highlight_states[pid] > 0:
                txt = "HIGHLIGHT CAPTURED"
                ts = 1.2
                cv2.putText(vis, txt, (x1, max(30, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, ts, (0,255,0), 5)
    led_sz = 36
    margin = 16
    led_x1 = width - margin - led_sz
    led_y1 = margin
    led_x2 = width - margin
    led_y2 = margin + led_sz
    led_color = (0,0,255)
    if led_state["frames_left"] > 0:
        led_color = (0,255,0)
    cv2.rectangle(vis, (led_x1, led_y1), (led_x2, led_y2), (50,50,50), -1)
    cv2.circle(vis, (led_x1 + led_sz//2, led_y1 + led_sz//2), led_sz//2 - 4, led_color, -1)
    out.write(vis)
    frame_count += 1

cap.release()
out.release()

subprocess.run([
    "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", audio_temp
], check=False)

subprocess.run([
    "ffmpeg", "-y", "-i", processed_path, "-i", audio_temp, "-c:v", "copy", "-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0", final_path
], check=False)

if os.path.exists(audio_temp):
    os.remove(audio_temp)
