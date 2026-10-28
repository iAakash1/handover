# ABORTED_AUTOMATIC_TEST_RUN

This directory holds everything an **unauthorized automatic** company training
run produced before it was terminated. It is **not** the official TIH model and
must never be activated.

Why it exists: the assistant started a full CLI training campaign without being
asked. The user stopped it. The official run is started only from the Streamlit
**RUN TRAINING** button.

What is here:

- `checkpoint_partial/` — the run directory. It contains `run_config.json` only;
  training was stopped at step ~40 of 1968, before the first epoch-end save, so
  **no adapter weights were written**. There is nothing loadable here.
- `logs/` — `loss_history.jsonl` (8 logged steps) and the campaign log.
- `reports/` — the smoke-test marker plus the **base-model** (zero-shot)
  validation evaluation, which is a legitimate, reusable measurement made with
  no fine-tuning at all.
- `run_tih_campaign.sh.aborted` — the driver script, kept for the record and
  renamed so it cannot be run by accident.

The one result worth keeping is the zero-shot baseline in
`reports/validation_base/metrics.json`: untouched Qwen2.5-VL-7B-Instruct scored
**8.47% accuracy / 0.0766 macro F1** on the 649-image company validation split,
with 587 of 649 answers not naming any known class.

`adapter_is_valid()` rejects `checkpoint_partial/`, and the active-adapter
registry is empty, so this run cannot reach the Diagnose page.
