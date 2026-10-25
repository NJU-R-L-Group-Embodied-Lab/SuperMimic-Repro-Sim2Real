#!/usr/bin/env bash
# Video clip(s) -> 30 fps transcode -> [cut-aware person boxes] -> GVHMR (static camera, world-frame SMPL-X)
#   -> GMR retarget to Unitree G1 29-DoF -> 36-column 30 fps CSV (drift floorfix + constant-floor variant) -> preview.
# See docs/VIDEO_TO_MOTION.md#environment-setup for installation (GVHMR / GMR / SMPL-X are external and user-installed).
#
#   GVHMR_ROOT=~/v2m/GVHMR GMR_ROOT=~/v2m/GMR tools/video2motion/run_pipeline.sh clip1.mp4 [clip2.mp4 ...]
#
# Environment variables (defaults in brackets):
#   GVHMR_ROOT   GVHMR checkout, pinned + patched as in docs/VIDEO_TO_MOTION.md       [required]
#   GMR_ROOT     GMR checkout; SMPL-X model at $GMR_ROOT/assets/body_models/smplx/       [required]
#   SMPLX_BODY_MODELS  folder that contains smplx/SMPLX_NEUTRAL.npz                    [$GMR_ROOT/assets/body_models]
#   V2M_WORK     work/output directory                                                  [<repo>/work/video2motion]
#   GVHMR_PY     python of the GVHMR env; if unset: conda run -n $GVHMR_ENV python      []
#   GVHMR_ENV    conda env name for GVHMR                                               [gvhmr]
#   GMR_PY       python of the GMR env; if unset: conda run -n $GMR_ENV python          []
#   GMR_ENV      conda env name for GMR                                                 [gmr]
#   G1_XML       G1 29-DoF MJCF for joint limits / feet / rendering; if unset, it is
#                resolved from this repo's uv environment (mjlab's g1.xml)               []
#   GPU          CUDA device for GVHMR / YOLO                                           [0]
#   SHOTS        1 = run shot_bbx.py first (clips with hard camera cuts)                [0]
#   RETARGET_ARGS  extra gvhmr_to_supermimic.py flags                                   [--init_from_human --restart_tol 0.1]
#   PREVIEW      0 = skip the side-by-side preview render                               [1]
#   MUJOCO_GL    headless GL backend for the preview: egl or osmesa                     [egl]
#
# Outputs per clip <c> (file stem of the input video):
#   $V2M_WORK/video30/<c>.mp4               30 fps copy of the input (what GVHMR sees; row t = frame t)
#   $V2M_WORK/gvhmr/<c>/hmr4d_results.pt    GVHMR result (+ <c>.log)
#   $V2M_WORK/csv_drift/<c>.csv             recommended CSV (floor drift removed) + <c>_gmr.pkl
#   $V2M_WORK/csv_constant/<c>.csv          one constant floor offset per clip + <c>_gmr.pkl
#   $V2M_WORK/preview/<c>.mp4               source | G1 side by side (drift CSV)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
: "${GVHMR_ROOT:?set GVHMR_ROOT to your GVHMR checkout (see docs/VIDEO_TO_MOTION.md#environment-setup)}"
: "${GMR_ROOT:?set GMR_ROOT to your GMR checkout (see docs/VIDEO_TO_MOTION.md#environment-setup)}"
GVHMR_ROOT=$(cd "$GVHMR_ROOT" && pwd)
GMR_ROOT=$(cd "$GMR_ROOT" && pwd)
export GMR_ROOT
V2M_WORK=${V2M_WORK:-$REPO/work/video2motion}
GVHMR_ENV=${GVHMR_ENV:-gvhmr}
GMR_ENV=${GMR_ENV:-gmr}
GPU=${GPU:-0}
SHOTS=${SHOTS:-0}
PREVIEW=${PREVIEW:-1}
RETARGET_ARGS=${RETARGET_ARGS---init_from_human --restart_tol 0.1}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}

py_gvhmr() {
  if [ -n "${GVHMR_PY:-}" ]; then "$GVHMR_PY" "$@"; else conda run --no-capture-output -n "$GVHMR_ENV" python "$@"; fi
}
py_gmr() {
  if [ -n "${GMR_PY:-}" ]; then "$GMR_PY" "$@"; else conda run --no-capture-output -n "$GMR_ENV" python "$@"; fi
}

if [ -z "${G1_XML:-}" ]; then
  G1_XML=$(cd "$REPO" && uv run --quiet python -c \
    'from supermimic.data.motion_convert import g1_xml_path; print(g1_xml_path())' 2>/dev/null || true)
  [ -n "$G1_XML" ] || { echo "set G1_XML (could not resolve mjlab's g1.xml via 'uv run' in $REPO)" >&2; exit 1; }
fi
export G1_XML
SMPLX_BODY_MODELS=${SMPLX_BODY_MODELS:-$GMR_ROOT/assets/body_models}
export SMPLX_BODY_MODELS
[ -f "$SMPLX_BODY_MODELS/smplx/SMPLX_NEUTRAL.npz" ] || {
  echo "missing $SMPLX_BODY_MODELS/smplx/SMPLX_NEUTRAL.npz (download SMPL-X yourself, see docs/VIDEO_TO_MOTION.md#environment-setup)" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
[ $# -ge 1 ] || { sed -n '2,6p' "$0"; exit 1; }

mkdir -p "$V2M_WORK/video30" "$V2M_WORK/gvhmr" "$V2M_WORK/preview"
for v in "$@"; do
  v=$(cd "$(dirname "$v")" && pwd)/$(basename "$v")
  c=$(basename "${v%.*}")
  v30=$V2M_WORK/video30/$c.mp4
  echo "== $c"
  # GVHMR and GMR assume 30 fps without resampling: a 25/60 fps source would play at the wrong speed
  ffmpeg -nostdin -loglevel error -y -i "$v" -vf fps=30 -an -c:v libx264 -crf 17 -pix_fmt yuv420p "$v30"
  if [ "$SHOTS" = 1 ]; then
    (cd "$V2M_WORK" && CUDA_VISIBLE_DEVICES=$GPU py_gvhmr "$HERE/shot_bbx.py" "$v30" "$V2M_WORK/gvhmr/$c/preprocess" \
      --gvhmr_root "$GVHMR_ROOT")
  fi
  # GVHMR's demo resolves its checkpoints relative to the checkout, so run it from there
  (cd "$GVHMR_ROOT" && CUDA_VISIBLE_DEVICES=$GPU py_gvhmr tools/demo/demo.py --video "$v30" \
    --output_root "$V2M_WORK/gvhmr" -s > "$V2M_WORK/gvhmr/$c.log" 2>&1) \
    || { echo "GVHMR failed, see $V2M_WORK/gvhmr/$c.log" >&2; exit 1; }
  for mode in drift constant; do
    # shellcheck disable=SC2086
    (cd "$V2M_WORK" && py_gmr "$HERE/gvhmr_to_supermimic.py" --gvhmr_pred "$V2M_WORK/gvhmr/$c/hmr4d_results.pt" \
      --name "$c" --floor_mode $mode --out_dir "$V2M_WORK/csv_$mode" --robot_xml "$G1_XML" $RETARGET_ARGS)
  done
  if [ "$PREVIEW" = 1 ]; then
    (cd "$V2M_WORK" && py_gmr "$HERE/render_preview.py" "$V2M_WORK/csv_drift/$c.csv" "$v30" "$V2M_WORK/preview/$c.mp4" \
      --robot_xml "$G1_XML")
  fi
done
