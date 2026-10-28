"""
scripts/judge_outputs.py

Scores every (model, image) raw output from benchmark.py using an
LLM-as-a-judge, against the filename-derived ground truth from
extract_ground_truth.py. The judge is given ONLY the ground-truth label
and the model's full text response — never the image itself — and asked
to return a strict-JSON rubric score (see config.RUBRIC_WEIGHTS).

Provider is selected via config.JUDGE_PROVIDER ("openai" or "gemini"),
controlled by the VLM_JUDGE_PROVIDER environment variable. The
corresponding API key (OPENAI_API_KEY or GEMINI_API_KEY) must be set.

If no key is configured, this script does NOT fail the pipeline — it
writes an empty judge_scores.json and logs how to enable judging.
Downstream scripts (evaluate.py, generate_csv.py, create_plots.py,
generate_report.py) all treat missing judge scores as null/unavailable
rather than crashing, so you can still get accuracy/latency/memory
results without ever calling a judge API.

Output: json_results/judge_scores.json
"""

import sys
import json
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import utils

# These point ranges must stay in sync with config.RUBRIC_WEIGHTS.
assert config.RUBRIC_WEIGHTS == {
    "plant_species": 10, "symptom_quality": 20, "disease_quality": 40,
    "hallucination": 20, "confidence_calibration": 10,
}, "JUDGE_INSTRUCTIONS below hardcodes these point values — update both together."

JUDGE_INSTRUCTIONS = """You are an expert plant pathologist acting as an impartial evaluator of an AI model's crop-disease diagnosis. You will be given the ground-truth label for an image (inferred independently from dataset metadata, NOT shown to the model being judged) and the model's full written response.

Score the model's response using this exact rubric. Each component score must be an integer within its stated range.

1. plant_species_score (0-10): Did the model correctly identify the plant species?
   10 = correct, 5 = plausible/partially correct (e.g. right family, wrong species), 0 = wrong or not stated.
2. symptom_quality_score (0-20): How accurate, specific, and visually grounded is the symptom description (color, texture, lesions)?
   20 = highly specific and accurate, 0 = vague, generic, or fabricated.
3. disease_quality_score (0-40): Is the "Most likely disease" correct (matches or is a reasonable synonym of the ground truth)? Award partial credit (10-25) if the correct disease appears only among "Alternative diagnoses".
   40 = exact correct primary diagnosis, 0 = no correct disease mentioned anywhere.
4. hallucination_score (0-20): Did the model invent symptoms, diseases, or details not plausible from a real leaf image, or state claims with false certainty?
   20 = fully grounded and appropriately hedged where uncertain, 0 = severe fabrication.
   (This is a GROUNDEDNESS score: HIGHER is better, i.e. LESS hallucination.)
5. confidence_calibration_score (0-10): Does the model's stated confidence (0-100) match how correct/well-grounded its answer actually is?
   10 = well calibrated, 0 = badly overconfident given a wrong/fabricated answer, or unjustifiably underconfident given a correct one.

Respond with ONLY a single valid JSON object, no prose, no markdown code fences, in exactly this shape:
{
  "plant_species_score": <int 0-10>,
  "symptom_quality_score": <int 0-20>,
  "disease_quality_score": <int 0-40>,
  "hallucination_score": <int 0-20>,
  "confidence_calibration_score": <int 0-10>,
  "justification": "<one or two sentence explanation>"
}
"""


def build_judge_user_prompt(ground_truth_disease, ground_truth_species, raw_model_output):
    return (
        "GROUND TRUTH (not shown to the model being judged):\n"
        f"  Plant species : {ground_truth_species}\n"
        f"  Disease       : {ground_truth_disease}\n\n"
        "MODEL'S FULL RESPONSE:\n"
        f"\"\"\"\n{raw_model_output}\n\"\"\"\n\n"
        "Score this response now."
    )


def call_openai_judge(user_prompt):
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.OPENAI_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


def call_gemini_judge(user_prompt):
    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        config.GEMINI_JUDGE_MODEL, system_instruction=JUDGE_INSTRUCTIONS
    )
    resp = model.generate_content(user_prompt, generation_config={"temperature": 0})
    return resp.text


def parse_judge_json(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = text.rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return None
    return None


def judge_one(ground_truth_disease, ground_truth_species, raw_model_output):
    user_prompt = build_judge_user_prompt(ground_truth_disease, ground_truth_species, raw_model_output)
    last_error = None
    for attempt in range(config.JUDGE_MAX_RETRIES):
        try:
            if config.JUDGE_PROVIDER == "gemini":
                raw = call_gemini_judge(user_prompt)
            else:
                raw = call_openai_judge(user_prompt)
            parsed = parse_judge_json(raw)
            if parsed is not None:
                total = sum(
                    parsed.get(k, 0) or 0 for k in [
                        "plant_species_score", "symptom_quality_score",
                        "disease_quality_score", "hallucination_score",
                        "confidence_calibration_score",
                    ]
                )
                parsed["total_score"] = total
                parsed["judge_error"] = None
                return parsed
            last_error = f"Judge returned unparseable output: {(raw or '')[:200]}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < config.JUDGE_MAX_RETRIES - 1:
            time.sleep(config.JUDGE_RETRY_DELAY)
    return {
        "plant_species_score": None, "symptom_quality_score": None,
        "disease_quality_score": None, "hallucination_score": None,
        "confidence_calibration_score": None, "total_score": None,
        "justification": None, "judge_error": last_error,
    }


def judge_is_configured():
    if config.JUDGE_PROVIDER == "gemini":
        return bool(config.GEMINI_API_KEY)
    return bool(config.OPENAI_API_KEY)


def main():
    config.ensure_directories()
    raw_path = config.JSON_RESULTS_DIR / "all_raw_results.json"
    gt_path  = config.JSON_RESULTS_DIR / "ground_truth.json"

    if not raw_path.exists():
        utils.log(f"Missing {raw_path}. Run benchmark.py first.")
        return
    if not gt_path.exists():
        utils.log(f"Missing {gt_path}. Run extract_ground_truth.py first.")
        return

    raw_results  = utils.load_json(raw_path)
    ground_truth = utils.load_json(gt_path)

    if not judge_is_configured():
        utils.log(
            f"No API key found for judge provider '{config.JUDGE_PROVIDER}'.\n"
            "  Set OPENAI_API_KEY (provider=openai, default) or GEMINI_API_KEY\n"
            "  (provider=gemini, set VLM_JUDGE_PROVIDER=gemini) and re-run.\n"
            "  Skipping LLM-judge scoring — downstream scripts will treat all\n"
            "  judge scores as null/unavailable and still produce accuracy,\n"
            "  latency, and memory results."
        )
        utils.save_json([], config.JSON_RESULTS_DIR / "judge_scores.json")
        return

    utils.log(f"Judging {len(raw_results)} model outputs with provider='{config.JUDGE_PROVIDER}'...")

    judged = []
    for i, rec in enumerate(raw_results, 1):
        gt = ground_truth.get(rec["image"], {})
        gt_disease = gt.get("ground_truth_disease", "Unknown")
        gt_species = gt.get("ground_truth_species", "Unknown")

        if rec.get("error") or not rec.get("raw_output"):
            score = {
                "plant_species_score": None, "symptom_quality_score": None,
                "disease_quality_score": None, "hallucination_score": None,
                "confidence_calibration_score": None, "total_score": None,
                "justification": "Skipped: model generation failed for this image.",
                "judge_error": rec.get("error"),
            }
        elif gt_disease == "Unknown":
            score = {
                "plant_species_score": None, "symptom_quality_score": None,
                "disease_quality_score": None, "hallucination_score": None,
                "confidence_calibration_score": None, "total_score": None,
                "justification": "Skipped: no resolvable ground-truth label for this image.",
                "judge_error": "ground_truth_unknown",
            }
        else:
            score = judge_one(gt_disease, gt_species, rec["raw_output"])

        judged.append({
            "model": rec["model"],
            "image": rec["image"],
            "ground_truth_disease": gt_disease,
            "ground_truth_species": gt_species,
            **score,
        })

        if i % 10 == 0 or i == len(raw_results):
            utils.log(f"  Judged {i}/{len(raw_results)}...")

    out_path = config.JSON_RESULTS_DIR / "judge_scores.json"
    utils.save_json(judged, out_path)
    utils.log(f"Saved judge scores -> {out_path}")


if __name__ == "__main__":
    main()
