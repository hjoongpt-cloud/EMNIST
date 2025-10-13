#!/usr/bin/env bash
# scripts/run_train_base.sh
# Usage: bash run_train_base.sh stage out_dir [additional config files]

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 stage_name output_dir [config1.yaml config2.yaml ...]"
  exit 1
fi

STAGE=$1
OUT_DIR=$2
shift 2
CONFIGS=($@)

# Ensure scripts directory
mkdir -p "$OUT_DIR"

# Build --config args
CONFIG_ARGS=()
for cfg in "${CONFIGS[@]}"; do
  CONFIG_ARGS+=("--config" "$cfg")
done

# For C-stage: sweep through preset strengths
# Preset strengths defined in configs/c.yaml under augment.strength
PRESETS=(light medium strong)

for strength in "${PRESETS[@]}"; do
  STAGE_C="stage_C_${strength}"
  echo "
*** Running C-stage with strength: $strength ***"
  # create temporary override config
  TMP_CFG="configs/c_${strength}.yaml"
  yq eval ".augment.strength=\"$strength\"" configs/c.yaml > "$TMP_CFG"

  python -m src.training.train_base \
    --config "$TMP_CFG" \
    --out_dir "$OUT_DIR/$STAGE_C" "$OUT_DIR/$STAGE_C"

  rm "$TMP_CFG"
done

echo "C-stage sweep completed. Results in $OUT_DIR" "Training completed for stage '$STAGE'. Results saved in $OUT_DIR/$STAGE"
