from moviepy import *

def merge_audio_video(video_path, audio_path, output_path):

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    video = video.with_audio(audio)

    video.write_videofile(output_path, codec='rawvideo', audio_codec='aac')
