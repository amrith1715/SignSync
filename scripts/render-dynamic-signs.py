"""Render the generated J/Z hand artwork into slow, browser-compatible clips.

Rendering dependencies: Pillow, numpy, opencv-python and an FFmpeg executable.
Example: python scripts/render-dynamic-signs.py --ffmpeg /path/to/ffmpeg
These are animated visual references, not recordings or recognition test data.
"""

import argparse
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dynamic_sign_videos"
SIZE = 640
FPS = 30
DURATION = 6.0
START_HOLD = 0.9
MOTION_DURATION = 3.4
FADE_START = 5.25
FADE_END = 5.65
J_TIMES = np.array([0, .19, .38, .58, .8, 1.0])
CROP = (130, 80, 510, 620)
VIDEO_WIDTH, VIDEO_HEIGHT = CROP[2] - CROP[0], CROP[3] - CROP[1]


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def rgba_stage(image, x, y):
    stage = Image.new("RGBA", (SIZE, SIZE))
    stage.alpha_composite(image, (round(x), round(y)))
    rgba = np.asarray(stage).astype(np.float32) / 255.0
    rgba[:, :, :3] *= rgba[:, :, 3:4]
    return rgba


def over_white(rgba):
    return np.clip((rgba[:, :, :3] + 1 - rgba[:, :, 3:4]) * 255, 0, 255).astype(np.uint8)


def load_j_poses():
    atlas = Image.open(OUTPUT / "source" / "J-poses.png").convert("RGBA")
    cell_w, cell_h = atlas.width / 3, atlas.height / 2
    # Pinky-tip landmarks in the six 512px cells, palm through edge to back.
    tips = [(198, 18), (190, 17), (247, 17), (264, 11), (337, 21), (363, 13)]
    stages = []
    scale = 340 / 512
    for index in range(6):
        col, row = index % 3, index // 3
        pose = atlas.crop((round(col * cell_w), round(row * cell_h),
                           round((col + 1) * cell_w), round((row + 1) * cell_h)))
        pose = pose.resize((340, 340), Image.Resampling.LANCZOS)
        # Align the fingertip before optical flow. Translation is rendered
        # separately so blending cannot produce a second moving fingertip.
        point = j_position(0)
        x = point[0] - tips[index][0] * scale
        y = point[1] - tips[index][1] * scale
        stages.append(rgba_stage(pose, x, y))
    return stages


def pose_flow(stages):
    gray = [cv2.cvtColor(over_white(stage), cv2.COLOR_RGB2GRAY) for stage in stages]
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    flows = []
    for left, right in zip(gray, gray[1:]):
        flows.append((estimator.calc(left, right, None), estimator.calc(right, left, None)))
    return flows


def interpolate_pose(stages, flows, progress):
    if progress >= 1:
        blended = stages[-1]
    else:
        index = max(0, min(len(stages) - 2, np.searchsorted(J_TIMES, progress, side="right") - 1))
        fraction = float((progress - J_TIMES[index]) / (J_TIMES[index + 1] - J_TIMES[index]))
        fraction = float(smoothstep(fraction))
        forward, backward = flows[index]
        yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
        left = cv2.remap(stages[index], xx - forward[:, :, 0] * fraction,
                         yy - forward[:, :, 1] * fraction, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        right = cv2.remap(stages[index + 1], xx - backward[:, :, 0] * (1 - fraction),
                          yy - backward[:, :, 1] * (1 - fraction), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
        blended = np.clip(left * (1 - fraction) + right * fraction, 0, 1)
    shift = j_position(progress) - j_position(0)
    matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    return cv2.warpAffine(blended, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def z_position(progress):
    points = np.array([(420, 120), (270, 120), (420, 240), (270, 240)], dtype=float)
    segment = min(2, int(progress * 3))
    fraction = float(smoothstep(progress * 3 - segment))
    return points[segment] * (1 - fraction) + points[segment + 1] * fraction


def j_position(progress):
    if progress < .4:
        return np.array([270, 120 + 115 * progress / .4])
    t = (progress - .4) / .6
    a, b, c, d = np.array([(270, 235), (270, 315), (385, 315), (385, 235)])
    return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d


def path_for(letter, progress):
    position = z_position if letter == "Z" else j_position
    return [position(float(p)) for p in np.linspace(0, progress, max(2, int(progress * 100)))]


def render_frame(letter, seconds, stages, flows, z_art):
    progress = float(np.clip((seconds - START_HOLD) / MOTION_DURATION, 0, 1))
    if letter == "J":
        rgba = interpolate_pose(stages, flows, progress)
    else:
        fingertip = z_position(progress)
        # Fingertip measured from source; artwork uniformly scaled, never mirrored.
        rgba = rgba_stage(z_art, fingertip[0] - 670 * .24, fingertip[1] - 54 * .24)
    background = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    if progress > .01:
        path = np.rint(path_for(letter, progress)).astype(np.int32)
        cv2.polylines(background, [path], False, (157, 187, 199), 3, cv2.LINE_AA)
    foreground = rgba[:, :, :3] * 255
    frame = foreground + background.astype(np.float32) * (1 - rgba[:, :, 3:4])
    # Fade to white before the loop resets; never demonstrate a reversed sign.
    fade = float(smoothstep((seconds - FADE_START) / (FADE_END - FADE_START)))
    frame = np.clip(frame * (1 - fade) + 255 * fade, 0, 255).astype(np.uint8)
    # Present the motion from the learner's viewpoint, matching the chart.
    return cv2.flip(frame[CROP[1]:CROP[3], CROP[0]:CROP[2]], 1)


def encode(letter, ffmpeg, stages, flows, z_art, preview_dir):
    output = OUTPUT / f"{letter}.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
               "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    samples = []
    try:
        for index in range(round(DURATION * FPS)):
            frame = render_frame(letter, index / FPS, stages, flows, z_art)
            if index == 12:
                Image.fromarray(frame).save(OUTPUT / f"{letter}.jpg", quality=94)
            if index in [12, 35, 55, 75, 95, 115, 135, 153]:
                samples.append(Image.fromarray(frame))
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait():
            raise RuntimeError(f"FFmpeg failed for {letter}: {error}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (1040, 740), "white")
        for index, sample in enumerate(samples):
            sheet.paste(sample.resize((260, 370), Image.Resampling.LANCZOS),
                        ((index % 4) * 260, (index // 4) * 370))
        sheet.save(preview_dir / f"{letter}-review.jpg", quality=93)
    print(f"Created {output.name}: {DURATION:.1f}s, {VIDEO_WIDTH}x{VIDEO_HEIGHT}, {FPS}fps, {output.stat().st_size:,} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("Pass --ffmpeg with the path to an FFmpeg executable.")
    stages = load_j_poses()
    print("Preparing J orientation transitions...", flush=True)
    flows = pose_flow(stages)
    z_art = Image.open(OUTPUT / "source" / "Z-hand.png").convert("RGBA")
    z_art = z_art.resize((round(z_art.width * .24), round(z_art.height * .24)), Image.Resampling.LANCZOS)
    for letter in ["J", "Z"]:
        encode(letter, args.ffmpeg, stages, flows, z_art, args.preview_dir)


if __name__ == "__main__":
    main()
