#!/usr/bin/env bash
# Shared by vlcsync.sh (system install), setup_venv.sh (venv install), and
# the Dockerfile (image build): clones joonson/syncnet_python and fetches
# its pretrained weights from Oxford VGG's official public release. Meant
# to be sourced, not executed directly.

# run_pipeline.py's scene_detect() opens its intermediate video.avi (an
# ffmpeg-produced mpeg4-in-avi file) directly via OpenCV/PySceneDetect. The
# PyPI opencv wheel on macOS ships AVFoundation as its only video backend
# (no bundled ffmpeg), and AVFoundation cannot decode that file at all --
# every other step in the pipeline shells out to real ffmpeg or reads JPGs,
# so this is the one call site that breaks. Patch it to hand PySceneDetect a
# throwaway H.264 transcode instead. Harmless on platforms where the
# original would have worked (Linux/Windows opencv wheels do bundle
# ffmpeg) -- just an extra small transcode there.
patch_syncnet_avi_scenedetect() {
  local target_dir="$1"
  local f="$target_dir/run_pipeline.py"

  if [ ! -f "$f" ]; then
    echo "error: $f not found -- can't apply the macOS scenedetect patch." >&2
    return 1
  fi
  if grep -q "video_for_scenedetect" "$f"; then
    echo "==> run_pipeline.py already patched for scene-detect compatibility"
    return 0
  fi

  echo "==> patching run_pipeline.py (scene_detect: macOS opencv/AVFoundation can't read its .avi)"
  python3 - "$f" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as fh:
    src = fh.read()

old = """def scene_detect(opt):

  video_path = os.path.join(opt.avi_dir,opt.reference,'video.avi')
  video = open_video(video_path)"""

new = """def scene_detect(opt):

  video_path = os.path.join(opt.avi_dir,opt.reference,'video.avi')

  # OpenCV's macOS wheel ships AVFoundation only (no bundled ffmpeg), and
  # AVFoundation can't decode the mpeg4-in-avi file produced above. Make a
  # throwaway H.264 transcode that it can actually open, used only here.
  scenedetect_path = os.path.join(opt.avi_dir,opt.reference,'video_for_scenedetect.mp4')
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                  "-c:v", "libx264", "-an", scenedetect_path], check=True)
  video = open_video(scenedetect_path)"""

if old not in src:
    sys.exit("error: expected scene_detect() source not found -- upstream repo changed; patch skipped")

src = src.replace(old, new, 1)
with open(path, "w") as fh:
    fh.write(src)
print("patched.")
PYEOF
}

# crop_video() writes each face-track crop with cv2.VideoWriter using the
# 'XVID' fourcc into a .avi container. On macOS, OpenCV's video *writer*
# has the same AVFoundation-only limitation as its reader: XVID/.avi opens
# silently as not-ready (isOpened() is False, but the code never checks)
# and writes nothing, so the next ffmpeg step fails with "No such file".
# 'avc1' (H.264) into .mp4 is a combination AVFoundation actually writes.
patch_syncnet_crop_writer() {
  local target_dir="$1"
  local f="$target_dir/run_pipeline.py"

  if [ ! -f "$f" ]; then
    echo "error: $f not found -- can't apply the macOS crop-writer patch." >&2
    return 1
  fi
  if grep -q "avc1" "$f"; then
    echo "==> run_pipeline.py already patched for crop-writer compatibility"
    return 0
  fi

  echo "==> patching run_pipeline.py (crop_video: macOS opencv can't write XVID/.avi)"
  python3 - "$f" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as fh:
    src = fh.read()

replacements = [
    (
        "  fourcc = cv2.VideoWriter_fourcc(*'XVID')\n"
        "  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))",
        "  # 'XVID'/.avi silently fails to open for writing on macOS (AVFoundation-only\n"
        "  # OpenCV build); 'avc1'/.mp4 is a combination it can actually write.\n"
        "  fourcc = cv2.VideoWriter_fourcc(*'avc1')\n"
        "  vOut = cv2.VideoWriter(cropfile+'t.mp4', fourcc, opt.frame_rate, (224,224))",
    ),
    (
        'command = ["ffmpeg", "-y", "-loglevel", "error", "-i", cropfile+\'t.avi\', "-i", audiotmp,',
        'command = ["ffmpeg", "-y", "-loglevel", "error", "-i", cropfile+\'t.mp4\', "-i", audiotmp,',
    ),
    (
        "os.remove(cropfile+'t.avi')",
        "os.remove(cropfile+'t.mp4')",
    ),
]

missing = [old for old, _ in replacements if old not in src]
if missing:
    sys.exit(f"error: expected crop_video() source not found (upstream repo changed?): {missing!r}")

for old, new in replacements:
    src = src.replace(old, new, 1)

with open(path, "w") as fh:
    fh.write(src)
print("patched.")
PYEOF
}

clone_and_fetch_syncnet_weights() {
  local target_dir="$1"

  if ! command -v git >/dev/null 2>&1; then
    echo "error: git not found." >&2
    return 1
  fi

  if [ -d "$target_dir" ]; then
    echo "==> $target_dir already exists, skipping clone"
  else
    echo "==> cloning joonson/syncnet_python..."
    git clone --depth 1 https://github.com/joonson/syncnet_python "$target_dir"
  fi

  patch_syncnet_avi_scenedetect "$target_dir"
  patch_syncnet_crop_writer "$target_dir"

  mkdir -p "$target_dir/data" "$target_dir/detectors/s3fd/weights"

  local model_path="$target_dir/data/syncnet_v2.model"
  local face_path="$target_dir/detectors/s3fd/weights/sfd_face.pth"

  if [ -f "$model_path" ]; then
    echo "==> syncnet_v2.model already present, skipping download"
  else
    echo "==> downloading syncnet_v2.model (Oxford VGG)..."
    curl -L --fail -o "$model_path" "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
  fi

  if [ -f "$face_path" ]; then
    echo "==> sfd_face.pth already present, skipping download"
  else
    echo "==> downloading sfd_face.pth (S3FD face detector, Oxford VGG)..."
    curl -L --fail -o "$face_path" "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth"
  fi
}
