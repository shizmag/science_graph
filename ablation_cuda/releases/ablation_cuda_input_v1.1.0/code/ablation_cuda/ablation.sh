#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
  echo "Usage: ./ablation.sh {doctor|install|smoke|run|resume|status|pack|verify} [options]" >&2
  exit 2
fi
shift || true

VERSION="1.1.0"
IMAGE="biomedical-rag-cuda:${VERSION}"
BASE_IMAGE_DIGEST="sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2"
STATE="${ABLA_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ablation_cuda}"
MODELS="$STATE/models"
DATABASE="$STATE/database"
LOGS="$STATE/logs"
OUTPUTS="$STATE/outputs"
RETURN_DIR="${ABLA_RETURN_DIR:-$ROOT/return}"
mkdir -p "$MODELS" "$DATABASE" "$LOGS" "$OUTPUTS" "$RETURN_DIR"

fail() { echo "ERROR: $*" >&2; exit 2; }

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing '$1'. Install Docker Engine with the NVIDIA Container Toolkit, then rerun ./ablation.sh doctor."
}

host_preflight() {
  [[ "$(uname -m)" == "x86_64" ]] || fail "This CUDA runtime is built for linux/amd64. Use a 64-bit x86_64 Linux host with an RTX 5090; detected '$(uname -m)'."
  require_command docker
  docker buildx version >/dev/null 2>&1 || fail "Docker Buildx is required for a local image digest. Update Docker Engine/Docker Desktop to a version that includes Buildx."
  local os_type
  os_type="$(docker info --format '{{.OSType}}' 2>/dev/null || true)"
  [[ "$os_type" == "linux" ]] || fail "Docker Engine must be running Linux containers (docker info reported '${os_type:-unavailable}'). Start the Linux Docker Engine and rerun."
  require_command nvidia-smi
  require_command nvidia-ctk
  nvidia-ctk --version >/dev/null 2>&1 || fail "NVIDIA Container Toolkit CLI is installed but not responding. Reinstall/configure the toolkit for Docker Engine using NVIDIA's official instructions."
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || fail "NVIDIA driver is not responding. Install a current NVIDIA Linux driver (580.65.06 or newer for CUDA 13 / Blackwell), reboot only through your normal maintenance process, then rerun. This script will not replace drivers or restart the host."
  local driver
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
  if [[ "$(printf '%s\n' 580.65.06 "$driver" | sort -V | head -n 1)" != "580.65.06" ]]; then
    fail "NVIDIA driver $driver is below the CUDA 13 Blackwell minimum 580.65.06. Update the host driver through your approved maintenance process."
  fi
}

ensure_image() {
  local metadata_file="$STATE/image-metadata.json"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1 || [[ ! -f "$metadata_file" ]]; then
    echo "Building the pinned linux/amd64 benchmark image (first build downloads locked runtime wheels)."
    docker buildx build --platform linux/amd64 --pull --load --metadata-file "$metadata_file" \
      -t "$IMAGE" -f "$ROOT/Dockerfile" "$ROOT"
  fi
  IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
  [[ "$IMAGE_ID" == sha256:* ]] || fail "Docker did not return an immutable local image ID."
  IMAGE_DIGEST="$(sed -n 's/.*"containerimage.digest":"\(sha256:[0-9a-f]*\)".*/\1/p' "$metadata_file")"
  [[ "$IMAGE_DIGEST" == sha256:* ]] || fail "Docker Buildx did not return an OCI image digest in $metadata_file."
}

container() {
  local command="$1"
  shift
  local gpu_args=()
  local mount_args=()
  case "$command" in
    doctor|install|smoke|run|resume) gpu_args+=(--gpus all) ;;
  esac
  if [[ "$command" == "verify" ]]; then
    [[ -n "${VERIFY_ARCHIVE:-}" ]] || fail "Internal verify archive path was not set."
    mount_args+=(--mount "type=bind,source=$VERIFY_ARCHIVE,target=/verify/archive.tar.gz,readonly")
    if [[ -f "$VERIFY_ARCHIVE.sha256" ]]; then
      mount_args+=(--mount "type=bind,source=$VERIFY_ARCHIVE.sha256,target=/verify/archive.tar.gz.sha256,readonly")
    fi
  fi
  docker run --rm --platform linux/amd64 "${gpu_args[@]}" \
    -v "$ROOT:/app/release:ro" \
    -v "$MODELS:/models" -v "$DATABASE:/db" -v "$LOGS:/logs" \
    -v "$OUTPUTS:/outputs" -v "$RETURN_DIR:/return" \
    "${mount_args[@]}" \
    -e ABLA_IMAGE_ID="$IMAGE_ID" -e ABLA_IMAGE_DIGEST="$IMAGE_DIGEST" \
    -e ABLA_BASE_IMAGE_DIGEST="$BASE_IMAGE_DIGEST" -e ABLA_UV_VERSION=0.10.9 -e HOME=/state/home \
    -e ABLA_HOST_OUTPUT_ROOT="$OUTPUTS" -e ABLA_HOST_RETURN_DIR="$RETURN_DIR" \
    "$IMAGE" "$command" "$@"
}

case "$COMMAND" in
  doctor)
    host_preflight
    ensure_image
    container doctor || fail "NVIDIA Container Toolkit is not exposing the GPU to this container. Configure Docker Engine with NVIDIA Container Toolkit, then rerun ./ablation.sh doctor."
    ;;
  install)
    host_preflight
    ensure_image
    container install
    ;;
  smoke)
    host_preflight
    ensure_image
    container doctor || fail "CUDA operations are unavailable inside the container. Resolve the host NVIDIA driver/Container Toolkit prerequisite before downloading model assets."
    # Runtime smoke runs installation/asset verification itself before starting
    # the two-arm query, so this is a single repeatable collaborator command.
    container smoke
    ;;
  run)
    host_preflight
    ensure_image
    container run "$@"
    ;;
  resume)
    host_preflight
    ensure_image
    if [[ $# -gt 1 ]]; then fail "Usage: ./ablation.sh resume [run_directory_name]"; fi
    if [[ $# -eq 1 ]]; then
      container resume --run-id "$1"
    else
      container resume
    fi
    ;;
  status)
    require_command docker
    ensure_image
    container status
    ;;
  pack)
    require_command docker
    ensure_image
    container pack --export-dir /return "$@"
    ;;
  verify)
    [[ $# -ge 1 ]] || fail "Usage: ./ablation.sh verify <results.tar.gz>"
    require_command docker
    ensure_image
    archive_path="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    [[ -f "$archive_path" ]] || fail "Results archive does not exist: $archive_path"
    VERIFY_ARCHIVE="$archive_path"
    export VERIFY_ARCHIVE
    archive_name="$(basename "$archive_path")"
    archive_id="${archive_name%.tar.gz}"
    import_destination="$RETURN_DIR/imported/$archive_id"
    container verify --archive /verify/archive.tar.gz --destination "/return/imported/$archive_id"
    ;;
  *)
    fail "Unknown command '$COMMAND'. Choose doctor, install, smoke, run, resume, status, pack or verify."
    ;;
esac
