"""
caption_synthesizer.py
======================
Fuses Part A (agronomy KB record) and Part B (visual-scene dict) into FIVE
diverse captions per image.

Design goal (your supervisor's brief):
    every caption must demonstrate BOTH
      (1) agricultural understanding -- crop, disease, pathogen, severity, symptoms
      (2) image-pattern understanding -- framing, lighting, hand/object, background

To get genuine diversity (and avoid the loss-collapse you saw when captions were
near-identical), the five captions use five different *registers* drawn from
the VLM-captioning literature -- a clinical report, a photograph description, a
symptom walk-through, a field-scout note, and a tight integrated label -- each
backed by its own pool of phrasings. A per-image seeded RNG keeps the output
reproducible while varying wording across images.
"""

from __future__ import annotations

import random
from typing import Dict, List

import config


# --------------------------------------------------------------------------- #
# Phrase renderers: turn coded scene attributes into natural language
# --------------------------------------------------------------------------- #
_SHOT = {
    "extreme_closeup": ["an extreme close-up", "a tightly cropped macro shot", "a very close view"],
    "single_leaf_closeup": ["a close-up of a single leaf", "a close shot of one detached leaf",
                            "a leaf-level close-up"],
    "multiple_leaves": ["a frame showing several leaves", "a view of multiple leaves together",
                        "a shot of a cluster of leaves"],
    "whole_plant": ["a view of the whole plant", "a shot of the entire plant",
                    "a full-plant photograph"],
    "field_wide": ["a wide field view", "a broad shot of the crop in the field",
                   "a wide canopy view"],
}
_LIGHT = {
    "bright_daylight": ["under bright daylight", "in strong natural light", "in clear sunlight"],
    "diffuse_overcast": ["under soft overcast light", "in flat diffuse light", "under a cloudy sky"],
    "indoor_artificial": ["under indoor artificial lighting", "in artificial light",
                          "under lab lighting"],
    "camera_flash": ["lit by a camera flash", "with hard flash lighting", "under direct flash"],
    "low_light_dark": ["in dim low-light conditions", "in a dark, poorly lit setting",
                       "under low, shadowed light"],
    "backlit": ["backlit against the light", "with the light coming from behind",
                "in back-lit conditions"],
}
_BG = {
    "soil_or_ground": ["against bare soil", "over a soil background", "with ground visible behind"],
    "dense_foliage": ["against a backdrop of other foliage", "with surrounding greenery behind",
                      "against dense leaves"],
    "sky": ["against an open sky", "with sky behind the leaf"],
    "plain_or_studio": ["against a plain studio background", "on a plain neutral background",
                        "against a clean backdrop"],
    "blurred_natural": ["against a softly blurred natural background", "with the background thrown out of focus",
                        "against a blurred outdoor scene"],
    "indoor_surface": ["resting on an indoor surface", "on a table-top surface"],
    "other": ["against a mixed background", "against an unremarkable background"],
}
_FOCUS = {
    "sharp": ["sharp and well-focused", "crisp and in focus", "cleanly focused"],
    "slightly_soft": ["a little soft in focus", "slightly soft", "marginally out of focus"],
    "blurred": ["noticeably blurred", "out of focus", "soft and hard to read in places"],
}
_WET = {
    "dew_or_droplets": ["with dew droplets on the surface", "speckled with water droplets"],
    "wet_glossy": ["wet and glossy", "with a wet sheen"],
}


def _pick(rng: random.Random, options: List[str]) -> str:
    return rng.choice(options)


def _scene_phrase(rng: random.Random, vis: Dict) -> str:
    """A compact phrase combining shot type, lighting and (if present) background."""
    shot = _pick(rng, _SHOT.get(vis["shot_type"], _SHOT["single_leaf_closeup"]))
    light = _pick(rng, _LIGHT.get(vis["lighting"], _LIGHT["bright_daylight"]))
    bg = _pick(rng, _BG.get(vis["background"], _BG["other"]))
    return f"{shot} {light}, {bg}"


def _hand_clause(rng: random.Random, vis: Dict) -> str:
    """Return a clause about a hand/object if one is present, else empty."""
    h = str(vis.get("hand_or_object", "none")).strip().lower()
    if h in ("", "none", "no", "n/a"):
        return ""
    raw = str(vis["hand_or_object"]).strip()
    templates = [
        f"; {raw} holds it in frame",
        f", and {raw} is visible holding the leaf",
        f", held by {raw}",
    ]
    return _pick(rng, templates)


def _quality_clause(rng: random.Random, vis: Dict) -> str:
    notes = str(vis.get("image_quality_notes", "none")).strip().lower()
    if notes in ("", "none", "no", "n/a"):
        return ""
    return f" The image shows {vis['image_quality_notes']}."


def severity_from_pct(pct: int) -> str:
    for band, (lo, hi) in config.SEVERITY_BANDS.items():
        if lo <= pct < hi:
            return band
    return "moderate"


def _symptom_sample(rng: random.Random, agro: Dict, k: int = 2) -> str:
    """Pick k canonical symptoms and join them naturally."""
    syms = list(agro.get("symptoms", []))
    rng.shuffle(syms)
    chosen = syms[:max(1, min(k, len(syms)))]
    if len(chosen) == 1:
        return chosen[0]
    return ", ".join(chosen[:-1]) + " and " + chosen[-1]


# --------------------------------------------------------------------------- #
# The five caption archetypes
# --------------------------------------------------------------------------- #
def _cap_diagnostic(rng, agro, vis, sev) -> str:
    crop = agro["crop"]; disease = agro["disease"]
    patho = agro["pathogen"]; ptype = agro["pathogen_type"]
    scene = _scene_phrase(rng, vis)
    syms = _symptom_sample(rng, agro, 2)
    if ptype == "none":
        return (f"This is {scene}, showing a {crop} leaf with no disease present. "
                f"The foliage displays {syms}, consistent with a healthy plant.")
    cause = (f"caused by the {ptype} {patho}" if ptype not in ("pest", "abiotic")
             else f"caused by {patho}")
    return (f"This is {scene}. The image shows {crop} affected by {disease}, "
            f"{cause}. Severity appears {sev}, with {syms}.")


def _cap_photo_first(rng, agro, vis, sev) -> str:
    crop = agro["crop"]; disease = agro["disease"]
    scene = _scene_phrase(rng, vis)
    focus = _pick(rng, _FOCUS.get(vis["focus_quality"], _FOCUS["sharp"]))
    hand = _hand_clause(rng, vis)
    obs = vis.get("observed_symptoms", "discolouration on the leaf")
    if agro["pathogen_type"] == "none":
        return (f"The photograph is {scene}{hand}; it is {focus}. The {crop} leaf "
                f"looks clean and healthy, with {obs} but no signs of disease.")
    return (f"The photograph is {scene}{hand}; it is {focus}. The {crop} leaf shows "
            f"{obs}, which points to {disease} at a {sev} stage.")


def _cap_symptom_walk(rng, agro, vis, sev) -> str:
    crop = agro["crop"]; disease = agro["disease"]
    syms = _symptom_sample(rng, agro, 3)
    distinguishing = agro.get("distinguishing", "")
    light = _pick(rng, _LIGHT.get(vis["lighting"], _LIGHT["bright_daylight"]))
    if agro["pathogen_type"] == "none":
        return (f"Seen {light}, the {crop} leaf shows {syms}. There are no lesions "
                f"or fungal growth, indicating healthy tissue.")
    tail = f" {distinguishing[0].upper() + distinguishing[1:]}, which identifies it as {disease}." \
        if distinguishing else f" These signs identify {disease}."
    return (f"Seen {light}, the {crop} leaf shows {syms}.{tail} The damage is {sev}.")


def _cap_field_scout(rng, agro, vis, sev) -> str:
    crop = agro["crop"]; disease = agro["disease"]
    shot = _pick(rng, _SHOT.get(vis["shot_type"], _SHOT["single_leaf_closeup"]))
    mgmt = agro.get("management", "")
    syms = _symptom_sample(rng, agro, 1)
    if agro["pathogen_type"] == "none":
        return (f"Scouting note: {shot} of {crop} foliage. Leaves are healthy with "
                f"{syms}; no action needed beyond routine monitoring.")
    advice = f" Recommended action: {mgmt}." if mgmt else ""
    return (f"Scouting note: {shot} of {crop}. The crop is showing {disease} "
            f"({sev}), recognisable by {syms}.{advice}")


def _cap_integrated(rng, agro, vis, sev) -> str:
    crop = agro["crop"]; disease = agro["disease"]
    shot = _pick(rng, _SHOT.get(vis["shot_type"], _SHOT["single_leaf_closeup"]))
    light = _pick(rng, _LIGHT.get(vis["lighting"], _LIGHT["bright_daylight"]))
    sym = _symptom_sample(rng, agro, 1)
    if agro["pathogen_type"] == "none":
        return (f"{shot.capitalize()} of a healthy {crop} leaf {light}, with {sym} "
                f"and no disease symptoms.")
    return (f"{shot.capitalize()} of a {crop} leaf {light}, showing {sym} "
            f"typical of {sev} {disease}.")


_ARCHETYPES = [
    ("diagnostic_report", _cap_diagnostic),
    ("photograph_description", _cap_photo_first),
    ("symptom_walkthrough", _cap_symptom_walk),
    ("field_scout_note", _cap_field_scout),
    ("integrated_label", _cap_integrated),
]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def synthesize(agro: Dict, vis: Dict, image_id: str) -> Dict:
    """
    Build the full record for one image: the derived severity, and the five
    captions (each tagged with its archetype).
    """
    rng = random.Random(f"{config.RANDOM_SEED}:{image_id}")
    pct = int(vis.get("estimated_pct_affected", 20))
    sev = "healthy" if agro["pathogen_type"] == "none" else severity_from_pct(pct)
    if sev == "healthy" and agro["pathogen_type"] != "none":
        sev = "mild"  # a diseased leaf is at least mild even if % is tiny

    captions = []
    for name, fn in _ARCHETYPES:
        text = fn(rng, agro, vis, sev)
        # collapse any accidental double spaces / spaces before punctuation
        text = " ".join(text.split()).replace(" ,", ",").replace(" .", ".").replace(" ;", ";")
        captions.append({"archetype": name, "text": text})

    return {
        "severity": sev,
        "severity_pct": pct,
        "captions": captions,
    }


if __name__ == "__main__":
    # Demonstrate on one diseased and one healthy example with a mock scene.
    import agronomy_kb, visual_analyzer
    from PIL import Image
    Image.new("RGB", (256, 256), (90, 110, 60)).save("/tmp/_c.jpg")
    vis = visual_analyzer.analyze_image("/tmp/_c.jpg", backend="mock")
    vis["hand_or_object"] = "a hand holding the leaf by the petiole"
    for label in ["Cotton___Bacterial_blight", "Tomato___healthy"]:
        agro = agronomy_kb.parse_label(label)
        rec = synthesize(agro, vis, label)
        print(f"\n=== {label}  (severity: {rec['severity']}) ===")
        for c in rec["captions"]:
            print(f"  [{c['archetype']}]\n    {c['text']}")
