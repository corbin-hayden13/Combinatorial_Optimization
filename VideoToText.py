import os
import glob
from pydub import AudioSegment
import speech_recognition as sr


def main():
    video_folder = "videos\\"
    audio_folder = "audio_files\\"

    files = glob.glob(os.path.join(video_folder, "*"))

    # filter video files
    video_files = [f for f in files if f.endswith((".mp4", ".avi", ".mkv", ".mov"))]

    for video in video_files:
        print(video)
        extension = os.path.splitext(video)[1]
        file_name = os.path.splitext(video)[0].split("\\")[1]
        print(file_name, extension)
        given_file = AudioSegment.from_file(video_folder + file_name + extension, format=extension[1:])
        print("Got File")
        given_file.export("audio_files\\" + file_name + extension, format="mp3")


if __name__ == "__main__":
    main()

