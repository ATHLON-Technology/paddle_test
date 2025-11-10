import os
import math
import subprocess
from pathlib import Path
import cv2
import numpy as np
import torch
from matplotlib.path import Path as MplPath
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import DFineForObjectDetection, AutoImageProcessor
from ultralytics import YOLO
import re

VIDEO_PATH = r"C:\Users\Administrator\Downloads\Paddle Test\B001_11062109_C026.mov"
OUT_DIR = r"C:\Users\Administrator\Downloads\Paddle Test\B001_11062109_C026_highlights"
REFERENCE_PATH = r"C:\Users\Administrator\Downloads\Paddle Test\c26.csv"

MANUAL_CHECK = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECT_EVERY_N = 20
POSE_IMG_SIZE = 640
BBOX_MIN_SCORE = 0.3
KP_MIN_CONF = 0.25

MONITOR_SECONDS = 3.0
CHECK_INTERVAL = 5

PRE_HIGHLIGHT_SECONDS = 30.0
BANNER_SECONDS = 3.0
IGNORE_TRIGGERS_WITHIN_SECONDS = 10.0

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
VIDEO_CODEC = "h264_nvenc"
ENC_PRESET = "fast"
CRF = 18
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_FINAL = os.path.join(OUT_DIR, "final_highlights.mp4")
LOG_PATH = os.path.join(OUT_DIR, "highlights.log")

def sec_to_mmss(s):
    m = int(s) // 60
    sec = int(s) % 60
    return f"{m:02d}:{sec:02d}"

def find_font_file():
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        r"/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco", use_fast=True)
dfine_model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
dfine_model.to(DEVICE)
dfine_model.eval()

pose_model = YOLO("yolo11m-pose.pt")
if DEVICE.type == "cuda":
    pose_model.to("cuda")

def load_court_points(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        raw = [line.rstrip("\n") for line in f]
    idx = None
    for i, line in enumerate(raw):
        if line.strip() == "Green Points:":
            idx = i
            break
    if idx is None:
        return None
    pts = []
    i = idx + 1
    while len(pts) < 4 and i < len(raw):
        line = raw[i].strip()
        if line:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    x = float(parts[0]); y = float(parts[1])
                    pts.append((x, y))
                except Exception:
                    pass
        i += 1
    return pts if len(pts) == 4 else None

raw_pts = load_court_points(REFERENCE_PATH)
court_polygon = None
if raw_pts is not None:
    court_polygon = MplPath(raw_pts)

def detect_dfine_frame(frame):
    image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dfine_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=BBOX_MIN_SCORE)[0]
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
    la_ok = la[2] > KP_MIN_CONF
    ra_ok = ra[2] > KP_MIN_CONF
    if la_ok and ra_ok:
        return ((la[0] + ra[0]) / 2.0, (la[1] + ra[1]) / 2.0)
    if la_ok:
        return (la[0], la[1])
    if ra_ok:
        return (ra[0], ra[1])
    lh = kp[11]; rh = kp[12]
    lh_ok = lh[2] > KP_MIN_CONF
    rh_ok = rh[2] > KP_MIN_CONF
    if lh_ok and rh_ok:
        return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    if lh_ok:
        return (lh[0], lh[1])
    if rh_ok:
        return (rh[0], rh[1])
    return None

def crop_and_run_pose(frame, person, width, height):
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
    results = pose_model.predict(source=crop, imgsz=POSE_IMG_SIZE, device=DEVICE.type, conf=0.25, verbose=False)
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
    if L_w_c < KP_MIN_CONF or R_w_c < KP_MIN_CONF:
        return False
    return (L_w_y > L_sh_y + 5) and (R_w_y > R_sh_y + 5)

def both_hands_raised(kp):
    if kp is None or len(kp) < 17:
        return False
    L_sh_y = kp[5][1]; R_sh_y = kp[6][1]
    L_w_y = kp[9][1]; R_w_y = kp[10][1]
    L_w_c = kp[9][2]; R_w_c = kp[10][2]
    if L_w_c < KP_MIN_CONF or R_w_c < KP_MIN_CONF:
        return False
    return (L_w_y < L_sh_y - 5) and (R_w_y < R_sh_y - 5)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video: " + VIDEO_PATH)

orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = orig_width
height = orig_height
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
video_duration = total_frames / fps if fps > 0 else 0

print(f"Video: {VIDEO_PATH} | {width}x{height} @ {fps:.2f}fps | {video_duration:.1f}s")

person_id_counter = 0
active_people = {}
last_detection = {}
for_frame_kp = {}
monitor_state = {}
last_accepted_highlight_time = -9999.0
highlights = []

frame_count = 0
MATCH_DISTANCE_THRESHOLD = 100.0

def match_persons(detections, prev_people):
    global person_id_counter
    cur_people = {}
    centers = [((d["bbox"][0] + d["bbox"][2]) / 2.0, (d["bbox"][1] + d["bbox"][3]) / 2.0) for d in detections]
    prev_ids = list(prev_people.keys())
    prev_centers = [((prev_people[i]["bbox"][0] + prev_people[i]["bbox"][2]) / 2.0, (prev_people[i]["bbox"][1] + prev_people[i]["bbox"][3]) / 2.0) for i in prev_ids] if prev_ids else []
    used_prev = set()
    for di, d in enumerate(detections):
        best_id = None; best_dist = 1e9
        for pi, pid in enumerate(prev_ids):
            if pid in used_prev:
                continue
            pc = prev_centers[pi]
            dist = math.hypot(pc[0] - centers[di][0], pc[1] - centers[di][1])
            if dist < best_dist:
                best_dist = dist; best_id = pid
        if best_id is not None and best_dist < MATCH_DISTANCE_THRESHOLD:
            used_prev.add(best_id)
            cur_people[best_id] = {"bbox": d["bbox"], "score": d["score"]}
        else:
            person_id_counter += 1
            cur_people[person_id_counter] = {"bbox": d["bbox"], "score": d["score"]}
    return cur_people

print("Starting main detection loop...")
for _ in tqdm(range(total_frames if total_frames > 0 else 1), desc="Main loop"):
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % DETECT_EVERY_N == 0:
        persons = detect_dfine_frame(frame)
        active_people = match_persons(persons, active_people)
        for pid in list(active_people.keys()):
            p = active_people[pid]
            kp = crop_and_run_pose(frame, p, width, height)
            if kp is None:
                del active_people[pid]
                last_detection.pop(pid, None)
                for_frame_kp.pop(pid, None)
                monitor_state.pop(pid, None)
                continue
            for_frame_kp[pid] = kp
            last_detection[pid] = {"bbox": p["bbox"], "kp": kp, "last_seen": frame_count}
            if both_hands_raised(kp) and pid not in monitor_state:
                monitor_state[pid] = {"frames_left": int(MONITOR_SECONDS * fps), "seen_below": False}
    else:
        for pid in list(active_people.keys()):
            if pid in last_detection:
                active_people[pid]["bbox"] = last_detection[pid]["bbox"]
                for_frame_kp[pid] = last_detection[pid]["kp"]
    if frame_count % CHECK_INTERVAL == 0 and len(monitor_state) > 0:
        for pid in list(monitor_state.keys()):
            if pid not in active_people:
                monitor_state.pop(pid, None)
                continue
            p = active_people[pid]
            kp = crop_and_run_pose(frame, p, width, height)
            if kp is None:
                continue
            for_frame_kp[pid] = kp
            if monitor_state[pid]["seen_below"]:
                if both_hands_raised(kp):
                    trigger_time = frame_count / fps
                    if trigger_time - last_accepted_highlight_time >= IGNORE_TRIGGERS_WITHIN_SECONDS:
                        clip_start = max(0.0, trigger_time - PRE_HIGHLIGHT_SECONDS)
                        clip_end = min(video_duration, trigger_time)
                        highlights.append({"trigger_time": trigger_time, "clip_start": clip_start, "clip_end": clip_end})
                        last_accepted_highlight_time = trigger_time
                        print(f"Accepted highlight at {trigger_time:.2f}s -> clip {clip_start:.2f}s..{clip_end:.2f}s")
                    else:
                        print(f"Ignored close trigger at {trigger_time:.2f}s (last accepted {last_accepted_highlight_time:.2f}s)")
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

if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)
with open(LOG_PATH, "w") as lf:
    if len(highlights) == 0:
        lf.write("No highlights detected")
    else:
        for i, h in enumerate(highlights, start=1):
            lf.write(f"Highlight_{i} {sec_to_mmss(h['clip_start'])} - {sec_to_mmss(h['clip_end'])}\n")

if len(highlights) == 0 and not MANUAL_CHECK:
    print("No highlights found. Exiting.")
    raise SystemExit(0)

def parse_time_to_seconds(t):
    t = t.strip()
    parts = t.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return None
    return None

def load_highlights_from_log(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    for ln in lines:
        if not ln:
            continue
        if ln.lower().startswith("no highlights"):
            continue
        m = re.search(r'Highlight_\d+\s+([0-9:\.]+)\s*-\s*([0-9:\.]+)', ln)
        if not m:
            m2 = re.search(r'([0-9:\.]+)\s*-\s*([0-9:\.]+)', ln)
            if not m2:
                continue
            a,b = m2.group(1), m2.group(2)
        else:
            a,b = m.group(1), m.group(2)
        s = parse_time_to_seconds(a)
        e = parse_time_to_seconds(b)
        if s is None or e is None:
            continue
        trigger = e
        s = max(0.0, min(s, video_duration))
        e = max(0.0, min(e, video_duration))
        if e - s <= 0:
            continue
        out.append({"trigger_time": trigger, "clip_start": s, "clip_end": e})
    out.sort(key=lambda x: x["clip_start"])
    return out

if MANUAL_CHECK:
    print(f"Manual check enabled. Edit the log at: {LOG_PATH}")
    print("After editing, type any non-empty text to proceed, 'reload' to reload without proceeding, or 'abort' to exit.")
    while True:
        cmd = input("cmd> ").strip()
        if cmd == "":
            continue
        if cmd.lower() in ("abort", "exit", "quit"):
            print("Aborted by user.")
            raise SystemExit(1)
        if cmd.lower() == "reload":
            highlights = load_highlights_from_log(LOG_PATH)
            print(f"Reloaded {len(highlights)} highlights from log.")
            continue
        highlights = load_highlights_from_log(LOG_PATH)
        print(f"Proceeding with {len(highlights)} highlights.")
        break

if len(highlights) == 0:
    print("No highlights found. Exiting.")
    raise SystemExit(0)

try:
    probe_cmd = [FFPROBE_BIN, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels", "-of", "default=noprint_wrappers=1:nokey=1", VIDEO_PATH]
    out = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().splitlines()
    if len(out) >= 2:
        AUDIO_SAMPLE_RATE = int(out[0])
        AUDIO_CHANNELS = int(out[1])
    elif len(out) == 1:
        AUDIO_SAMPLE_RATE = int(out[0])
        AUDIO_CHANNELS = 2
    else:
        AUDIO_SAMPLE_RATE = 48000
        AUDIO_CHANNELS = 2
except Exception:
    AUDIO_SAMPLE_RATE = 48000
    AUDIO_CHANNELS = 2

print(f"Audio sample rate: {AUDIO_SAMPLE_RATE}, channels: {AUDIO_CHANNELS}")

if "nvenc" in VIDEO_CODEC:
    v_enc_opts = ["-c:v", VIDEO_CODEC, "-preset", ENC_PRESET, "-rc", "vbr", "-cq", "19"]
else:
    v_enc_opts = ["-c:v", VIDEO_CODEC, "-preset", ENC_PRESET, "-crf", str(CRF)]

fontfile = find_font_file()
concat_inputs = []
filter_parts = []

for i, h in enumerate(highlights, start=1):
    banner_path = os.path.join(OUT_DIR, f"banner_{i:03d}.mp4")
    clip_path = os.path.join(OUT_DIR, f"clip_{i:03d}.mp4")
    audio_start = max(0.0, h["trigger_time"] - BANNER_SECONDS)
    banner_audio_tmp = os.path.join(OUT_DIR, f"banner_audio_{i:03d}.wav")
    extract_audio_cmd = [
        FFMPEG_BIN, "-y", "-i", VIDEO_PATH, "-ss", f"{audio_start:.3f}", "-t", f"{BANNER_SECONDS:.3f}", "-vn",
        "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS), banner_audio_tmp
    ]
    subprocess.run(extract_audio_cmd, check=True)
    if fontfile is not None:
        ff_font = fontfile.replace('\\', '/')
        draw_expr = f"fontfile='{ff_font}':text='Highlight_{i}':fontcolor=white:fontsize={int(min(width,height)/10)}:x=(w-text_w)/2:y=(h-text_h)/2"
    else:
        draw_expr = f"text='Highlight_{i}':fontcolor=white:fontsize={int(min(width,height)/10)}:x=(w-text_w)/2:y=(h-text_h)/2"
    color_input = f"color=size={width}x{height}:duration={BANNER_SECONDS}:color=black"
    banner_cmd = [
        FFMPEG_BIN, "-y",
        "-f", "lavfi", "-i", color_input,
        "-i", banner_audio_tmp,
        "-filter_complex", f"[0:v]drawtext={draw_expr}[v]",
        "-map", "[v]", "-map", "1:a",
    ] + v_enc_opts + [
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
        banner_path
    ]
    subprocess.run(banner_cmd, check=True)
    start = h["clip_start"]
    duration = max(0.1, h["clip_end"] - h["clip_start"])
    clip_cmd = [
        FFMPEG_BIN, "-y", "-i", VIDEO_PATH, "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
    ] + v_enc_opts + [
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
        clip_path
    ]
    subprocess.run(clip_cmd, check=True)
    concat_inputs.append((banner_path, clip_path))

input_args = []
for pair in concat_inputs:
    for p in pair:
        input_args += ["-i", p]

n_inputs = len(concat_inputs) * 2
filter_fragments = []
for i in range(n_inputs):
    filter_fragments.append(f"[{i}:v]fps=fps={int(round(fps))},setpts=PTS-STARTPTS[vid{i}]")
    filter_fragments.append(f"[{i}:a]aresample={AUDIO_SAMPLE_RATE},asetpts=PTS-STARTPTS[aud{i}]")

concat_inputs_labels = "".join(f"[vid{i}][aud{i}]" for i in range(n_inputs))
filter_complex = ";".join(filter_fragments) + ";" + f"{concat_inputs_labels}concat=n={n_inputs}:v=1:a=1[v][a]"

final_cmd = [FFMPEG_BIN, "-y"] + input_args + [
    "-filter_complex", filter_complex,
    "-map", "[v]", "-map", "[a]",
    "-vsync", "2", "-r", str(int(round(fps))),
] + v_enc_opts + [
    "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", AUDIO_BITRATE,
    "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
    OUTPUT_FINAL
]

print("Running final concat with normalized filter_complex (strict A/V sync)...")
subprocess.run(final_cmd, check=True)
