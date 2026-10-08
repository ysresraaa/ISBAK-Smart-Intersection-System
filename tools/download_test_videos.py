"""
Downloads the test videos into the videos/ folder using yt-dlp.
The videos are not stored in the repository for copyright reasons and are
intended for personal testing only.

Usage:  python tools/download_test_videos.py
Edit the VIDEOS list to add your own videos.
"""
import os

import yt_dlp

VIDEOS = [
    {"url": "https://youtu.be/i1JfZpiypAk", "name": "test_police.mp4"},
    {"url": "https://youtu.be/ixPibKV8Kbc", "name": "test_fire_truck.mp4"},
    {"url": "https://youtu.be/CJZXN0EHOTg", "name": "istanbul_traffic.mp4"},
]

os.makedirs("videos", exist_ok=True)

for v in VIDEOS:
    opts = {
        # Single-file (progressive) format: avoids merge errors from separate audio/video streams
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join("videos", v["name"]),
        "retries": 10,
        "fragment_retries": 10,
        "nooverwrites": True,
    }
    print(f"Downloading: {v['name']}")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([v["url"]])
    except Exception as e:
        print(f"WARNING: could not download {v['name']} ({e}). Please try again.")

print("Done. Run: python smart_intersection.py")
