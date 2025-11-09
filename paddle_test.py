import cv2
import torch
import numpy as np
from transformers import DFineForObjectDetection, AutoImageProcessor
from ultralytics import YOLO
from PIL import Image
import subprocess
import os
from tqdm import tqdm
import math
from matplotlib.path import Path
import shutil
import sys

video_path = r"C:\Users\Amr Hekal\Desktop\Paddle test\B001_10292257_C005.mov"
final_path = r"C:\Users\Amr Hekal\Desktop\Paddle test\B001_10292257_C005_match_actual3_highlights.mp4"
reference_path = r"C:/Users/Amr Hekal/Desktop/Paddle test/points_3.csv"
log_path = "highlights_2.log"
tmp_dir = "highlights_tmp"

if shutil.which("ffmpeg") is None:
    print("ffmpeg not found in PATH. Install ffmpeg and make sure it is on PATH.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco", use_fast=True)
dfine_model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
dfine_model.to(device)
dfine_model.eval()

pose_model = YOLO("yolo11m-pose.pt")
if device_str == "cuda":
    pose_model.to("cuda")

cap = cv2.VideoCapture(video_path)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = orig_width
height = orig_height

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
video_duration = total_frames / fps if fps > 0 else 0.0

def load_court_points(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        raw = [line.rstrip('') for line in f]
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
court_polygon = Path(raw_pts) if raw_pts is not None else None

detect_every_n = 20
pose_img_size = 640
bbox_min_score = 0.3
kp_min_conf = 0.25
highlight_seconds = 2.0
highlight_hold_frames = int(highlight_seconds * fps)

check_interval = 5
monitor_seconds = 3.0
monitor_max_frames = int(monitor_seconds * fps)

skeleton = [
    (5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)
]

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

def foot_point_from_kp(kp):
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
    except Exception:
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
    if court_polygon is not None:
        foot = foot_point_from_kp(chosen)
        if foot is not None:
            if not court_polygon.contains_point((foot[0], foot[1])):
                return None
    return chosen

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
    return (L_w_y < L_sh_y - 5) and (R_w_y < R_sh_y - 5)

monitor_state = {}
person_id_counter = 0
active_people = {}
last_detection = {}
match_distance_threshold = 100
for_frame_kp = {}

if os.path.exists(log_path):
    os.remove(log_path)

frame_count = 0
highlight_triggers = []

for _ in tqdm(range(total_frames), desc="Main loop"):
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % detect_every_n == 0:
        persons = detect_dfine_frame(frame)
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
            if both_hands_raised(kp) and pid not in monitor_state:
                monitor_state[pid] = {"frames_left": monitor_max_frames, "seen_below": False}
    else:
        for pid in list(active_people.keys()):
            if pid in last_detection:
                active_people[pid]["bbox"] = last_detection[pid]["bbox"]
                for_frame_kp[pid] = last_detection[pid]["kp"]
    if frame_count % check_interval == 0 and len(monitor_state) > 0:
        for pid in list(monitor_state.keys()):
            if pid not in active_people:
                monitor_state.pop(pid, None)
                continue
            p = active_people[pid]
            kp = crop_and_run_pose(frame, p)
            if kp is None:
                continue
            for_frame_kp[pid] = kp
            if monitor_state[pid]["seen_below"]:
                if both_hands_raised(kp):
                    highlight_triggers.append(frame_count)
                    monitor_state.pop(pid, None)
            else:
                if wrists_below_shoulders(kp):
                    monitor_state[pid]["seen_below"] = True
    expired = []
    for pid in list(monitor_state.keys()):
        monitor_state[pid]["frames_left"] -= 1
        if monitor_state[pid]["frames_left"] <= 0:
            expired.append(pid)
    for pid in expired:
        monitor_state.pop(pid, None)
    frame_count += 1

cap.release()

if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir, exist_ok=True)

if len(highlight_triggers) == 0:
    print("No highlights triggered.")
    sys.exit(0)

trigger_frames_sorted = sorted(set(highlight_triggers))
trigger_times = [f / fps for f in trigger_frames_sorted]
merge_threshold = 3.0
segments = []
for t in trigger_times:
    start = max(0.0, t - 10.0)
    end = t
    if not segments:
        segments.append([start, end])
    else:
        prev = segments[-1]
        if start <= prev[1] + merge_threshold:
            prev[1] = max(prev[1], end)
        else:
            segments.append([start, end])

seg_files = []
mapping_triggers = []
font_candidates = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\Tahoma.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
]
fontfile = None
for p in font_candidates:
    if os.path.exists(p):
        fontfile = p
        break

for i, (start, end) in enumerate(segments):
    dur = end - start
    if dur <= 0.01:
        mapping_triggers.append([])
        continue
    triggers_in_seg = [f for f in trigger_frames_sorted if (f / fps) >= start - 1e-6 and (f / fps) <= end + 1e-6]
    mapping_triggers.append(triggers_in_seg)
    title_file = os.path.join(tmp_dir, f"title_{i+1:03d}.mp4")
    clip_file = os.path.join(tmp_dir, f"clip_{i+1:03d}.mp4")
    title_text = f"Highlight_{i+1}"
    if fontfile is not None:
        font_arg = fontfile.replace("\\", "/")
        vf = f"drawtext=fontfile={font_arg}:text={title_text}:fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2"
    else:
        vf = f"drawtext=text={title_text}:fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2"
    ff_title_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:d=3",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-t", "3",
        "-r", str(int(round(fps))),
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-pix_fmt", "yuv420p",
        title_file
    ]
    try:
        subprocess.run(ff_title_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("ffmpeg title creation failed for", title_file)
        print("ffmpeg stdout:", e.stdout)
        print("ffmpeg stderr:", e.stderr)
        raise
    ff_extract_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", f"{start:.6f}",
        "-t", f"{dur:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-r", str(int(round(fps))),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-pix_fmt", "yuv420p",
        "-avoid_negative_ts", "make_zero",
        clip_file
    ]
    try:
        subprocess.run(ff_extract_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("ffmpeg clip extraction failed for", clip_file)
        print("ffmpeg stdout:", e.stdout)
        print("ffmpeg stderr:", e.stderr)
        raise
    seg_files.append(title_file)
    seg_files.append(clip_file)

list_file = os.path.join(tmp_dir, "concat_list.txt")
with open(list_file, "w", encoding="utf-8") as lf:
    for f in seg_files:
        lf.write(f"file '{os.path.abspath(f)}'\n")

ff_concat_cmd = [
    "ffmpeg", "-y",
    "-f", "concat", "-safe", "0",
    "-i", list_file,
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
    "-pix_fmt", "yuv420p",
    final_path
]
subprocess.run(ff_concat_cmd, check=True)

with open(log_path, "w", encoding="utf-8") as lf:
    for i, (start, end) in enumerate(segments):
        triggers = mapping_triggers[i] if i < len(mapping_triggers) else []
        triggers_str = ",".join([str(int(t)) for t in triggers])
        lf.write(f"highlight_{i+1},start_s,{start:.3f},end_s,{end:.3f},triggers,{triggers_str}\n")

print(f"Created final highlights video: {final_path}")
print(f"Log written to: {log_path}")
print(f"Temporary files in: {tmp_dir} (delete if not needed)")
