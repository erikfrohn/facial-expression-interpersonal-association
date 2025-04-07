import moviepy.editor as mp

def merge_audio_video(video_path, audio_path, output_path):

    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(audio_path)

    video = video.set_audio(audio)

    video.write_videofile(output_path, codec='rawvideo', audio_codec='aac')
