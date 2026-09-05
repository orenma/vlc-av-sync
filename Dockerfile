# Fully self-contained image: ffmpeg, both detection backends, and the
# pretrained SyncNet weights baked in. No brew, no touching host Python.
#
# NOTE: containers have no display, so this image can only do the compute
# side -- detect the offset and/or remux a corrected file. It cannot launch
# VLC for you; open the resulting file (or the original with --audio-desync)
# in the VLC on your Mac. vlcsync.sh's `fix` still handles this gracefully
# (prints the command to run on your Mac instead of failing).
#
# Build:
#   docker build -t vlcsync .
# Use (mount the directory holding your video as /data):
#   docker run --rm -v "$PWD":/data vlcsync detect /data/movie.mp4 --model syncnet --start 300 --duration 20
#   docker run --rm -v "$PWD":/data vlcsync fix /data/movie.mp4 --model syncnet --start 300 --duration 20 --apply
#   docker run --rm -v "$PWD":/data vlcsync subs /data/movie.mp4 /data/subs.srt

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vlcsync

COPY requirements.txt requirements-syncnet.txt lib_syncnet_install.sh ./

# CPU-only torch build first, from PyTorch's own index -- the default PyPI
# wheel pulls CUDA libs that are dead weight on a CPU-only image.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt -r requirements-syncnet.txt

RUN bash -c "source lib_syncnet_install.sh && clone_and_fetch_syncnet_weights /opt/vlcsync/third_party/syncnet_python"

COPY av_sync_detect.py syncnet_detect.py vlcsync.sh ./
RUN chmod +x vlcsync.sh

ENTRYPOINT ["./vlcsync.sh"]
CMD ["--help"]
