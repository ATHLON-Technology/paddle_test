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
from matplotlib.path import Path

video_path = r"C:\Users\Amr Hekal\Desktop\Paddle test\B001_10292257_C005.mov"
processed_path = r"C:\Users\Amr Hekal\Desktop\Paddle test\B001_102922357_C005_match_processed_highlights.mp4"
final_path = r"C:\Users\Amr Hekal\Desktop\Paddle test\B001_10292257_C005_match_final_highlights.mp4"
reference_path = r"C:/Users/Amr Hekal/Desktop/Paddle test/points_3.csv"
audio_temp = "temp_audio.aac"
log_path = "highlights_2.log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco", use_fast=True)
dfine_model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
dfine_model.to(device)
dfine_model.eval()

# Use YOLO pose weights (user supplied file). Make sure file exists.
pose_model = YOLO("yolo11m-pose.pt")
if device_str == "cuda":
    pose_model.to("cuda")

cap = cv2.VideoCapture(video_path)
# original capture dimensions (before rotation)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# after rotating right 90deg, width/height swap
width = orig_width
height = orig_height

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_size = (width, height)
out = cv2.VideoWriter(processed_path, fourcc, fps, out_size)

# rotate mode (cv2.ROTATE_90_CLOCKWISE rotates to the right)
ROTATE_MODE = cv2.ROTATE_90_CLOCKWISE

# court polygon (will be loaded from CSV and rotated to match rotated frames)
court_polygon = None

# ---------- load court points from CSV (exact sample format) ----------
def load_court_points(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        raw = [line.rstrip('') for line in f]
    # find the line 'Green Points:' (exact match in sample)
    idx = None
    for i, line in enumerate(raw):
        if line.strip() == 'Green Points:':
            idx = i
            break
    if idx is None:
        return None
    pts = []
    i = idx + 1
    while len(pts) < 4 and i < len(raw):
        line = raw[i].strip()
        if line:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                try:
                    x = float(parts[0]); y = float(parts[1])
                    pts.append((x, y))
                except Exception:
                    pass
        i += 1
    return pts if len(pts) == 4 else None

raw_pts = load_court_points(reference_path)
court_polygon = Path(raw_pts)
#if raw_pts is not None:
    # rotate the court points the same way frames are rotated
    # mapping for clockwise 90deg: (x_old, y_old) -> (x_new, y_new) = (y_old, orig_width - x_old)
#    rotated_pts = [(y, orig_width - x) for (x, y) in raw_pts]
#    court_polygon = Path(rotated_pts)
#    print(f"Loaded court polygon (rotated): {rotated_pts}")
#else:
#    print("Warning: no court points loaded (reference_path missing or wrong format) -> no court filtering applied")

# ----------------- algorithm parameters -----------------
detect_every_n = 20
pose_img_size = 640
bbox_min_score = 0.3
kp_min_conf = 0.25
highlight_seconds = 2.0
highlight_hold_frames = int(highlight_seconds * fps)

# Monitoring parameters for pattern detection
check_interval = 5  # frames between pose checks during monitoring
monitor_seconds = 3.0
monitor_max_frames = int(monitor_seconds * fps)

skeleton = [
    (5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)
]

# ----------------- DFine detection ----------------------
from PIL import Image as PILImage

def detect_dfine_frame(frame):
    image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

# ----------------- pose + court filtering -----------------

def foot_point_from_kp(kp):
    # kp: list of (x,y,conf) in full-frame coords
    if kp is None or len(kp) < 17:
        return None
    la = kp[15]; ra = kp[16]
    la_ok = la[2] > kp_min_conf
    ra_ok = ra[2] > kp_min_conf
    if la_ok and ra_ok:
        return ((la[0] + ra[0]) / 2.0, (la[1] + ra[1]) / 2.0)
    if la_ok:
        return (la[0], la[1])
    if ra_ok:
        return (ra[0], ra[1])
    # fallback to hips
    lh = kp[11]; rh = kp[12]
    lh_ok = lh[2] > kp_min_conf
    rh_ok = rh[2] > kp_min_conf
    if lh_ok and rh_ok:
        return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    if lh_ok:
        return (lh[0], lh[1])
    if rh_ok:
        return (rh[0], rh[1])
    return None


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

    # robustly extract keypoints (same as before)
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
            return None
    elif arr.ndim == 1:
        if arr.size % 3 == 0:
            parsed = arr.reshape(1, arr.size // 3, 3)
        else:
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
                return None
        except Exception:
            return None
    if parsed is None or parsed.size == 0:
        return None

    # map crop coords back to full-frame coords
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
    # choose best instance
    if len(mapped) == 1:
        chosen = mapped[0]
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
        chosen = mapped[best_idx]

    # if we have a court polygon, make sure the foot point lies inside it
    if court_polygon is not None:
        foot = foot_point_from_kp(chosen)
        if foot is not None:
            # Path.contains_point expects (x,y)
            if not court_polygon.contains_point((foot[0], foot[1])):
                return None
    return chosen

# ----------------- highlight pattern state machine -----------------

def wrists_below_shoulders(kp):
    if kp is None or len(kp) < 17:
        return False
    L_sh_y = kp[5][1]; R_sh_y = kp[6][1]
    L_w_y = kp[9][1]; R_w_y = kp[10][1]
    L_w_c = kp[9][2]; R_w_c = kp[10][2]
    if L_w_c < kp_min_conf or R_w_c < kp_min_conf:
        return False
    return (L_w_y > L_sh_y + 5) and (R_w_y > R_sh_y + 5)

def both_hands_raised(kp):
    if kp is None or len(kp) < 17:
        return False
    L_sh_y = kp[5][1]; R_sh_y = kp[6][1]
    L_w_y = kp[9][1]; R_w_y = kp[10][1]
    L_w_c = kp[9][2]; R_w_c = kp[10][2]
    if L_w_c < kp_min_conf or R_w_c < kp_min_conf:
        return False
    # require wrists to be clearly above shoulders (small margin)
    return (L_w_y < L_sh_y - 5) and (R_w_y < R_sh_y - 5)

monitor_state = {}  # pid -> {frames_left, seen_below}
highlight_states = {}
led_state = {"frames_left": 0}

person_id_counter = 0
active_people = {}
last_detection = {}
match_distance_threshold = 100
for_frame_kp = {}

if os.path.exists(log_path):
    os.remove(log_path)

# ----------------- main loop -----------------
frame_count = 0
for _ in tqdm(range(total_frames), desc="Main loop"):
    ret, frame = cap.read()
    if not ret:
        break

    # rotate frame right by 90 degrees before any processing
    #frame = cv2.rotate(frame, ROTATE_MODE)

    # Run DFine every detect_every_n frames
    if frame_count % detect_every_n == 0:
        persons = detect_dfine_frame(frame)
        matched = (lambda detections, prev: __import__('math') or [] )
        # reuse original match_persons inline (keeps file self-contained)
        def match_persons(detections, prev_people):
            global person_id_counter
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

        active_people = match_persons(persons, active_people)
        # run pose for each detection and eliminate if foot outside court (crop_and_run_pose returns None)
        for pid in list(active_people.keys()):
            p = active_people[pid]
            kp = crop_and_run_pose(frame, p)
            if kp is None:
                del active_people[pid]
                if pid in last_detection: del last_detection[pid]
                if pid in for_frame_kp: del for_frame_kp[pid]
                if pid in monitor_state: del monitor_state[pid]
                continue
            for_frame_kp[pid] = kp
            last_detection[pid] = {"bbox": p["bbox"], "kp": kp, "last_seen": frame_count}
            # if initial raise detected, start monitoring
            if both_hands_raised(kp) and pid not in monitor_state:
                monitor_state[pid] = {"frames_left": monitor_max_frames, "seen_below": False}
    else:
        # between DFine runs: restore last bboxes/kps (holds drawing)
        for pid in list(active_people.keys()):
            if pid in last_detection:
                active_people[pid]["bbox"] = last_detection[pid]["bbox"]
                for_frame_kp[pid] = last_detection[pid]["kp"]

    # periodic monitoring checks every `check_interval` frames
    if frame_count % check_interval == 0 and len(monitor_state) > 0:
        for pid in list(monitor_state.keys()):
            if pid not in active_people:
                monitor_state.pop(pid, None)
                continue
            p = active_people[pid]
            kp = crop_and_run_pose(frame, p)
            if kp is None:
                # if we can't get kp this check, skip but continue monitoring until timeout
                continue
            for_frame_kp[pid] = kp
            # if we already saw a below -> wait for above again
            if monitor_state[pid]["seen_below"]:
                if both_hands_raised(kp):
                    # pattern matched: below then above -> trigger highlight
                    highlight_states[pid] = highlight_hold_frames
                    led_state["frames_left"] = max(led_state["frames_left"], highlight_hold_frames)
                    with open(log_path, "a") as lf:
                        tsec = frame_count / fps
                        lf.write(f"frame,{frame_count},{tsec:.3f}s,person_id,{pid}")
                    monitor_state.pop(pid, None)
            else:
                # waiting to observe wrist under shoulders
                if wrists_below_shoulders(kp):
                    monitor_state[pid]["seen_below"] = True

    # decrement monitor timers and remove expired monitors
    expired = []
    for pid in list(monitor_state.keys()):
        monitor_state[pid]["frames_left"] -= 1
        if monitor_state[pid]["frames_left"] <= 0:
            expired.append(pid)
    for pid in expired:
        monitor_state.pop(pid, None)

    # drawing + highlight decay
    vis = frame.copy()
    # update highlights based on manual triggers (also handled above)
    for pid in list(active_people.keys()):
        kp = for_frame_kp.get(pid, None)
        # If a player has no kp (maybe filtered by court), skip drawing but keep id until next detection
        b = active_people[pid]["bbox"]
        x1,y1,x2,y2 = map(int, b)
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
    # decrement all active highlight timers (avoid using loop variable from earlier)
    for pid in list(highlight_states.keys()):
        if highlight_states[pid] > 0:
            highlight_states[pid] -= 1
    if led_state["frames_left"] > 0:
        led_state["frames_left"] -= 1

    # LED
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

# re-attach audio
subprocess.run([
    "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", audio_temp
], check=False)
subprocess.run([
    "ffmpeg", "-y", "-i", processed_path, "-i", audio_temp, "-c:v", "copy", "-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0", final_path
], check=False)
if os.path.exists(audio_temp):
    os.remove(audio_temp)
