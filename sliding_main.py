import cv2
import numpy as np
from matplotlib import pyplot as plt

import eulerian
import heartrate
import preprocessing
import pyramids

# Frequency range for Fast-Fourier Transform
freq_min = 50 / 60
freq_max = 180 / 60

# Preprocessing phase
print("Reading + preprocessing video...")
video_frames, frame_ct, fps = preprocessing.read_video(
    "/run/media/davidyz/NIKON Z 5/DCIM/183RAW__/VID_0274.MP4"
)

# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
x, y = [], []
sc = ax.scatter(x, y)
plt.ylabel("Heart Rate (bpm)")
plt.xlabel("Time (frame)")
plt.title("Real-time Heart Rate Plot")


# Define window size and step size
window_size = fps * 2  # Number of frames in each window
step_size = 1  # Number of frames to slide the window

# Process video in a sliding window
for start_frame in range(0, frame_ct - window_size + 1, step_size):
    end_frame = start_frame + window_size
    window_frames = video_frames[start_frame:end_frame]

    # Build Laplacian video pyramid for the window
    lap_video_window = pyramids.build_video_pyramid(window_frames)

    for i, video in enumerate(lap_video_window):
        if i == 0 or i == len(lap_video_window) - 1:
            continue

        # Eulerian magnification with temporal FFT filtering
        result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
        lap_video_window[i] += result

        # Calculate heart rate for the window
        heart_rate = heartrate.find_heart_rate_segment(
            fft, frequencies, freq_min, freq_max
        )
        print(f"Heart rate for window {start_frame}-{end_frame}: {heart_rate} bpm")

        # Update plot
        x.append(start_frame)
        y.append(heart_rate)
        sc.set_offsets(np.c_[x, y])
        ax.set_xlim(left=max(0, start_frame - 100), right=start_frame + 100)
        ax.set_ylim(bottom=max(0, min(y) - 10), top=max(y) + 10)
        plt.draw()
        plt.pause(0.1)

plt.ioff()
plt.show()
