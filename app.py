import os
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from pydub import AudioSegment
import numpy as np
import cv2
import moviepy.editor as mpy
import librosa
import soundfile as sf
import mediapipe as mp

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_FILES = 50

app = Flask(__name__, static_folder="static", template_folder="templates")
JOBS = {}
mp_face = mp.solutions.face_mesh

# -------------------
def cleanup_old_files():
    all_files = list(UPLOAD_DIR.glob("*")) + list(OUTPUT_DIR.glob("*"))
    if len(all_files) <= MAX_FILES:
        return
    all_files.sort(key=lambda x: x.stat().st_mtime)
    for f in all_files[:-MAX_FILES]:
        try:
            f.unlink()
        except: pass

# Cortar áudio
def trim_audio(in_path, out_path, start_ms=0, end_ms=None):
    audio = AudioSegment.from_file(in_path)
    if end_ms is None:
        trimmed = audio[start_ms:]
    else:
        trimmed = audio[start_ms:end_ms]
    trimmed.export(out_path, format="wav")
    return out_path

# Envelope de áudio
def amplitude_envelope(wav_path, sr=16000, hop_length=512):
    y, sr = librosa.load(wav_path, sr=sr)
    env = np.array([np.max(np.abs(y[i:i+hop_length])) for i in range(0, len(y), hop_length)])
    if env.max() > 0:
        env = env / env.max()
    return env, sr

# Geração de vídeo com animações faciais
def generate_animation(image_path, audio_path, out_path, fps=25,
                       mouth_amp=0.6, head_amp=3.0, blink_speed=30,
                       eyebrow_amp=2.0, smile_amp=0.2):
    cleanup_old_files()
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # Boca
            mouth_inds = [61, 291, 13, 14, 78, 308, 82, 312]
            xs = [int(lm[i].x*w) for i in mouth_inds]
            ys = [int(lm[i].y*h) for i in mouth_inds]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            pad_x = max(6, int((x2-x1)*0.3))
            pad_y = max(6, int((y2-y1)*0.6))
            mouth_box = (max(0,x1-pad_x), max(0,y1-pad_y), min(w,x2+pad_x), min(h,y2+pad_y))
        else:
            cx, cy = w//2, h//2
            mouth_box = (int(cx- w*0.15), int(cy + h*0.05), int(cx + w*0.15), int(cy + h*0.25))

    env, sr = amplitude_envelope(audio_path, sr=16000)
    audio_info = sf.info(audio_path)
    duration = audio_info.duration
    total_frames = max(1, int(duration*fps))
    frames = []
    env_frame = np.interp(np.linspace(0,len(env)-1,total_frames), np.arange(len(env)), env)

    for i in range(total_frames):
        frame = img.copy()
        # Cabeça
        theta = np.deg2rad(head_amp * np.sin(2*np.pi*(i/total_frames)*1.2))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        M_rot = np.array([[cos_t,-sin_t,0],[sin_t,cos_t,0]],dtype=np.float32)
        cx, cy = w//2, h//2
        M_rot[0,2] = (1 - cos_t)*cx - sin_t*cy
        M_rot[1,2] = sin_t*cx + (1 - cos_t)*cy
        frame = cv2.warpAffine(frame, M_rot, (w,h), borderMode=cv2.BORDER_REFLECT)

        # Boca
        x1,y1,x2,y2 = mouth_box
        mouth_roi = frame[y1:y2, x1:x2].copy()
        if mouth_roi.size !=0:
            mh, mw = mouth_roi.shape[:2]
            scale = 1.0 + mouth_amp*float(env_frame[i])
            new_h = max(2,int(mh*scale))
            resized = cv2.resize(mouth_roi,(mw,new_h),interpolation=cv2.INTER_LINEAR)
            ystart = max(0, y1 - (new_h-mh)//2)
            yend = ystart + new_h
            if yend <= frame.shape[0]:
                overlay = frame.copy()
                overlay[ystart:yend, x1:x2] = resized
                frame = cv2.addWeighted(overlay, 0.95, frame, 0.05,0)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    clip = mpy.ImageSequenceClip(frames,fps=fps)
    clip = clip.set_audio(mpy.AudioFileClip(str(audio_path)))
    clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac", verbose=False, logger=None)

# -----------------
# Rotas
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error":"Nenhum arquivo enviado"}),400
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    path = UPLOAD_DIR / fname
    file.save(path)
    return jsonify({"img_path": str(path)})

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    file = request.files.get("audio")
    if not file:
        return jsonify({"error":"Nenhum arquivo enviado"}),400
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    path = UPLOAD_DIR / fname
    file.save(path)
    return jsonify({"audio_path": str(path)})

@app.route("/generate", methods=["POST"])
def generate():
    image_path = request.form.get("image_path")
    audio_path = request.form.get("audio_path")
    start = float(request.form.get("start",0))
    end = float(request.form.get("end",0))
    job_id = f"{uuid.uuid4().hex}.mp4"
    out_path = OUTPUT_DIR / job_id

    def worker():
        temp_audio = UPLOAD_DIR / f"temp_{uuid.uuid4().hex}.wav"
        trim_audio(audio_path,temp_audio,int(start*1000),int(end*1000))
        generate_animation(image_path, temp_audio, out_path)
        JOBS[job_id] = {"status":"done"}

    JOBS[job_id] = {"status":"processing"}
    threading.Thread(target=worker).start()
    return jsonify({"job": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    return jsonify(JOBS.get(job_id, {"status":"unknown"}))

@app.route("/output/<job_id>")
def output(job_id):
    path = OUTPUT_DIR / job_id
    if path.exists():
        return send_file(path, as_attachment=True)
    return "Arquivo não encontrado",404

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
