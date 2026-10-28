#!/usr/bin/env bash
# Run the PlantDx pipeline stages in dependency order.
# Stages are stubbed until their milestone lands (they exit non-zero with a
# milestone message); this script documents the canonical invocation order.
set -euo pipefail

CONFIG="${1:-configs/config.yaml}"

echo "== PlantDx pipeline (config: ${CONFIG}) =="

# Milestone 2
plantdx ontology build   --config "${CONFIG}" || true
plantdx vocabulary build --config "${CONFIG}" || true

# Milestone 3
plantdx generate         --config "${CONFIG}" || true
plantdx validate         --config "${CONFIG}" || true

# Milestone 4
plantdx dataset build    --config "${CONFIG}" || true
for model in qwen2_5_vl qwen3_vl internvl3 gemma3 mlx_vlm; do
  plantdx dataset convert --model "${model}" || true
done
plantdx qa sample        --config "${CONFIG}" || true

echo "== Done (stubbed stages skipped) =="
