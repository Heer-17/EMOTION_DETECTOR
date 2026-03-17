import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from deepface import DeepFace

st.set_page_config(page_title="AI Emotion Analytics", page_icon="🧠", layout="wide")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_EMOJIS = {"happy": "😄", "neutral": "😐", "sad": "😢", "angry": "😡", "fear": "😨", "surprise": "😲", "disgust": "🤢"}
EMOTION_COLORS = {"happy": "#00f5d4", "neutral": "#7b61ff", "sad": "#4a9eff", "angry": "#ff4d6d", "fear": "#ff9f1c", "surprise": "#f7d716", "disgust": "#a8ff78"}

if "running" not in st.session_state:
    st.session_state.running = False
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "last_scores" not in st.session_state:
    st.session_state.last_scores = {e: 0.0 for e in EMOTIONS}
if "dominant_emotion" not in st.session_state:
    st.session_state.dominant_emotion = "N/A"

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ START", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹ STOP", use_container_width=True)
    if start_btn:
        st.session_state.running = True
        st.session_state.emotion_history = []
        st.session_state.frame_count = 0
    if stop_btn:
        st.session_state.running = False
    frame_skip = st.slider("Analyze every N frames", 1, 10, 3)
    confidence_threshold = st.slider("Confidence threshold (%)", 0, 50, 10)
    enforce_detection = st.checkbox("Enforce face detection", value=False)
    st.markdown("---")
    frames_ph = st.empty()
    detections_ph = st.empty()
    top_emotion_ph = st.empty()

st.title("🧠 AI Emotion Analytics")
st.caption("Real-Time Facial Emotion Detection — Minor Project 3")
st.markdown("🔴 **LIVE**" if st.session_state.running else "⚫ STOPPED")
st.markdown("---")

left_col, right_col = st.columns([1.2, 1], gap="large")
with left_col:
    st.markdown("### 📷 Live Camera Feed")
    video_ph = st.empty()
    face_status_ph = st.empty()
with right_col:
    st.markdown("### 📊 Emotion Confidence Scores")
    chart_ph = st.empty()
    st.markdown("### 🎯 Dominant Emotion")
    dominant_ph = st.empty()

st.markdown("---")
st.markdown("### 📈 Emotion Timeline")
timeline_ph = st.empty()

def draw_box(frame, region, emotion):
    x, y, w, h = region.get("x",0), region.get("y",0), region.get("w",0), region.get("h",0)
    hex_c = EMOTION_COLORS.get(emotion, "#00f5d4")
    r, g, b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
    color = (b, g, r)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame, (px, py), (px+dx*18, py), color, 3)
        cv2.line(frame, (px, py), (px, py+dy*18), color, 3)
    label = f"{EMOTION_EMOJIS.get(emotion,'')} {emotion.upper()}"
    cv2.rectangle(frame, (x, y-25), (x+150, y), color, -1)
    cv2.putText(frame, emotion.upper(), (x+5, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    return frame

def show_chart(scores, ph):
    df = pd.DataFrame({
        "Emotion": [f"{EMOTION_EMOJIS.get(e,'')} {e.capitalize()}" for e in EMOTIONS],
        "Confidence (%)": [scores.get(e, 0.0) for e in EMOTIONS]
    }).set_index("Emotion")
    ph.bar_chart(df, use_container_width=True, height=280)

def show_timeline(history, ph):
    if not history:
        ph.info("No data yet.")
        return
    total = len(history)
    df = pd.DataFrame({
        "Emotion": [f"{EMOTION_EMOJIS.get(e,'')} {e.capitalize()}" for e in EMOTIONS],
        "Frequency (%)": [history.count(e)/total*100 for e in EMOTIONS]
    }).set_index("Emotion")
    ph.bar_chart(df, use_container_width=True, height=200)

def update_sidebar():
    frames_ph.metric("Frames Analyzed", st.session_state.frame_count)
    detections_ph.metric("Face Detections", len(st.session_state.emotion_history))
    if st.session_state.emotion_history:
        top = max(set(st.session_state.emotion_history), key=st.session_state.emotion_history.count)
        top_emotion_ph.metric("Most Frequent", f"{EMOTION_EMOJIS.get(top,'')} {top.upper()}")

show_chart(st.session_state.last_scores, chart_ph)
show_timeline(st.session_state.emotion_history, timeline_ph)
update_sidebar()

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        st.session_state.running = False
    else:
        face_status_ph.info("🔍 Scanning...")
        local_count = 0
        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break
                local_count += 1
                st.session_state.frame_count += 1
                if local_count % frame_skip == 0:
                    try:
                        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=enforce_detection, silent=True)
                        if isinstance(result, list):
                            result = result[0]
                        scores = result.get("emotion", {})
                        dominant = result.get("dominant_emotion", "neutral").lower()
                        region = result.get("region", {})
                        if scores.get(dominant, 0) < confidence_threshold:
                            dominant = "neutral"
                        total = sum(scores.values())
                        norm = {k: (v/total)*100 for k, v in scores.items()} if total > 0 else {e: 0.0 for e in EMOTIONS}
                        st.session_state.last_scores = norm
                        st.session_state.dominant_emotion = dominant
                        st.session_state.emotion_history.append(dominant)
                        frame = draw_box(frame, region, dominant)
                        face_status_ph.success("✅ Face detected")
                        show_chart(norm, chart_ph)
                        color = EMOTION_COLORS.get(dominant, "#7b61ff")
                        emoji = EMOTION_EMOJIS.get(dominant, "")
                        dominant_ph.markdown(f"## {emoji} {dominant.upper()}")
                        if len(st.session_state.emotion_history) % 5 == 0:
                            show_timeline(st.session_state.emotion_history, timeline_ph)
                    except Exception:
                        face_status_ph.warning("⚠️ No face detected")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_ph.image(rgb, channels="RGB", use_container_width=True)
                update_sidebar()
                time.sleep(0.03)
        finally:
            cap.release()

if not st.session_state.running and st.session_state.emotion_history:
    st.markdown("---")
    st.markdown("### 📋 Session Summary")
    history = st.session_state.emotion_history
    total = len(history)
    counts = {e: history.count(e) for e in EMOTIONS}
    top = max(counts, key=counts.get)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Readings", total)
    c2.metric("Dominant Emotion", f"{EMOTION_EMOJIS.get(top,'')} {top.upper()}")
    c3.metric("Unique Emotions", len([e for e in EMOTIONS if counts[e] > 0]))
    df = pd.DataFrame([{"Emotion": f"{EMOTION_EMOJIS.get(e,'')} {e.capitalize()}", "Count": counts[e], "%": f"{counts[e]/total*100:.1f}%"} for e in EMOTIONS if counts[e] > 0])
    st.dataframe(df.sort_values("Count", ascending=False).reset_index(drop=True), use_container_width=True)
    show_timeline(history, timeline_ph)