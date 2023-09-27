#coding=utf-8
import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path, "emotion_model"))
import json

from emotion_model import audio_emotion_model


wav_dir = "/mnt/work/sda/video_labeling/emotion/sample_wav_clips"
wav_clip_list = []

for wav_file in os.listdir(wav_dir):
    if wav_file.endswith(".wav"):
        wav_path = os.path.join(wav_dir, wav_file)
        wav_clip_list.append(wav_path)

result_str = audio_emotion_model.predict(wav_clip_list)
result_dict = json.loads(result_str)

print(result_dict)