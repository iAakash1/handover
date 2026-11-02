"""
agronomy_kb.py
==============
Part A of the model: the AGRICULTURAL UNDERSTANDING.

This module holds a curated knowledge base of crop diseases and a robust parser
that turns a folder name or file name (the ground-truth label) into a structured
disease record.

Why ground-truth labels instead of letting the VLM guess the disease?
---------------------------------------------------------------------
For new crops (cotton, sugarcane, maize) a zero-shot VLM will hallucinate the
pathogen and confuse look-alike diseases. By anchoring the *diagnosis* to the
known label and using the vision model only for the *visual scene* (Part B), the
captions stay agronomically correct -- which is exactly what you need to show
your supervisor. The KB below is the single source of truth for that accuracy.

Each disease record contains:
    crop            human-readable crop name
    disease         human-readable disease name
    pathogen        scientific name of the causal agent
    pathogen_type   fungus / bacterium / virus / oomycete / pest / abiotic / none
    symptoms        list of canonical visual symptoms (used to enrich captions)
    distinguishing  one short phrase that separates it from look-alikes
    management      one short control recommendation
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# KNOWLEDGE BASE
# Keys are normalised "crop|disease" slugs. Add new entries freely.
# --------------------------------------------------------------------------- #
KB: Dict[str, Dict] = {

    # ------------------------------- COTTON -------------------------------- #
    "cotton|bacterial_blight": {
        "crop": "cotton", "disease": "bacterial blight (angular leaf spot)",
        "pathogen": "Xanthomonas citri pv. malvacearum", "pathogen_type": "bacterium",
        "symptoms": [
            "small water-soaked angular spots bounded by leaf veins",
            "lesions turning dark brown to black with age",
            "a yellow halo sometimes ringing the spots",
            "black streaks along veins and petioles ('black arm') in advanced cases",
        ],
        "distinguishing": "spots are angular because veins confine them, unlike round fungal spots",
        "management": "use acid-delinted certified seed and copper-based bactericides",
    },
    "cotton|alternaria_leaf_spot": {
        "crop": "cotton", "disease": "Alternaria leaf spot",
        "pathogen": "Alternaria macrospora", "pathogen_type": "fungus",
        "symptoms": [
            "small brown circular spots with concentric rings",
            "a pale grey centre that may crack and fall out",
            "spots coalescing into large blighted areas",
            "premature yellowing and defoliation",
        ],
        "distinguishing": "concentric target-like rings with a brittle grey centre",
        "management": "balanced potassium nutrition and mancozeb sprays",
    },
    "cotton|grey_mildew": {
        "crop": "cotton", "disease": "grey mildew (areolate mildew)",
        "pathogen": "Ramularia areola", "pathogen_type": "fungus",
        "symptoms": [
            "pale angular spots on the upper leaf surface",
            "a frosty white-to-grey fungal growth on the underside",
            "spots limited by veins giving a blocky pattern",
            "heavy infections causing leaf drop",
        ],
        "distinguishing": "white powdery growth on the lower surface within angular blocks",
        "management": "remove crop residue and apply wettable sulphur",
    },
    "cotton|leaf_curl_virus": {
        "crop": "cotton", "disease": "cotton leaf curl virus disease",
        "pathogen": "Cotton leaf curl virus (begomovirus, whitefly-transmitted)",
        "pathogen_type": "virus",
        "symptoms": [
            "upward or downward curling of leaf margins",
            "thickened, darkened veins visible from beneath",
            "leaf-like enations on the underside of veins",
            "stunted bushy growth with reduced boll set",
        ],
        "distinguishing": "vein thickening and cup-shaped curling rather than spotting",
        "management": "control whitefly vectors and grow resistant varieties",
    },

    # ------------------------------ SUGARCANE ------------------------------ #
    "sugarcane|red_rot": {
        "crop": "sugarcane", "disease": "red rot",
        "pathogen": "Colletotrichum falcatum", "pathogen_type": "fungus",
        "symptoms": [
            "reddening of the internal stalk tissue with crosswise white patches",
            "drooping and yellowing of the upper leaves",
            "a sour alcoholic smell from split canes",
            "reddish lesions with dark margins on the midrib",
        ],
        "distinguishing": "red internal pith broken by white cross-bands, plus a vinegar smell",
        "management": "plant resistant clones and use disease-free setts",
    },
    "sugarcane|smut": {
        "crop": "sugarcane", "disease": "whip smut",
        "pathogen": "Sporisorium scitamineum", "pathogen_type": "fungus",
        "symptoms": [
            "a long black whip-like structure emerging from the growing point",
            "the whip covered by a thin silvery membrane that ruptures into black spores",
            "thin, grassy tillering of the stool",
            "stunted, narrow leaves",
        ],
        "distinguishing": "the curved black 'whip' is unmistakable",
        "management": "rogue infected stools and hot-water treat setts",
    },
    "sugarcane|rust": {
        "crop": "sugarcane", "disease": "common rust",
        "pathogen": "Puccinia melanocephala", "pathogen_type": "fungus",
        "symptoms": [
            "small elongated yellowish flecks on the leaf",
            "flecks enlarging into orange-brown to reddish-brown pustules",
            "pustules mostly on the lower leaf surface",
            "drying of heavily infected leaves",
        ],
        "distinguishing": "rusty pustules that rub off as orange powder",
        "management": "grow resistant varieties and apply triazole fungicides if severe",
    },
    "sugarcane|mosaic": {
        "crop": "sugarcane", "disease": "mosaic virus disease",
        "pathogen": "Sugarcane mosaic virus (aphid-transmitted potyvirus)",
        "pathogen_type": "virus",
        "symptoms": [
            "a contrasting pattern of pale green to yellow stripes against normal green",
            "the mottling most visible at the leaf base",
            "occasional leaf narrowing and stunting",
            "no spots or pustules, only colour mosaic",
        ],
        "distinguishing": "light-and-dark green mosaic with no fungal fruiting bodies",
        "management": "use virus-free seed cane and control aphids",
    },

    # ------------------------------- MAIZE --------------------------------- #
    "maize|common_rust": {
        "crop": "maize", "disease": "common rust",
        "pathogen": "Puccinia sorghi", "pathogen_type": "fungus",
        "symptoms": [
            "small cinnamon-brown powdery pustules on both leaf surfaces",
            "pustules scattered or in bands across the blade",
            "pustules turning brownish-black as the plant matures",
            "chlorosis around heavy pustule clusters",
        ],
        "distinguishing": "circular-to-elongate pustules erupting through the epidermis on both sides",
        "management": "plant resistant hybrids; fungicide only under high pressure",
    },
    "maize|northern_leaf_blight": {
        "crop": "maize", "disease": "northern leaf blight",
        "pathogen": "Exserohilum turcicum", "pathogen_type": "fungus",
        "symptoms": [
            "long elliptical grey-green to tan cigar-shaped lesions",
            "lesions running parallel to the leaf margins",
            "lesions enlarging and merging to blight whole leaves",
            "a dull green-grey sporulation in the lesion centre when humid",
        ],
        "distinguishing": "large cigar-shaped lesions, much longer than wide",
        "management": "rotate crops, bury residue, use resistant hybrids",
    },
    "maize|gray_leaf_spot": {
        "crop": "maize", "disease": "gray leaf spot",
        "pathogen": "Cercospora zeae-maydis", "pathogen_type": "fungus",
        "symptoms": [
            "narrow rectangular tan-to-grey lesions confined by the veins",
            "lesions giving a striped blocky appearance",
            "lesions coalescing to grey-brown blighted bands",
            "lower leaves affected first",
        ],
        "distinguishing": "rectangular lesions with sharp parallel sides set by the veins",
        "management": "residue management and timely triazole/strobilurin sprays",
    },
    "maize|leaf_blight": {
        "crop": "maize", "disease": "leaf blight",
        "pathogen": "Bipolaris / Exserohilum spp.", "pathogen_type": "fungus",
        "symptoms": [
            "tan to brown elongated lesions along the blade",
            "lesions with darker borders",
            "progressive drying from the lower leaves upward",
            "blighted, scorched leaf tips in severe cases",
        ],
        "distinguishing": "elongated necrotic blotches that scorch the leaf",
        "management": "resistant hybrids and protectant fungicides",
    },

    # ------------------------------- TOMATO -------------------------------- #
    "tomato|early_blight": {
        "crop": "tomato", "disease": "early blight",
        "pathogen": "Alternaria solani", "pathogen_type": "fungus",
        "symptoms": [
            "dark brown spots with concentric rings forming a target pattern",
            "a yellow halo surrounding the lesions",
            "lesions starting on the older, lower leaves",
            "affected leaves yellowing and dropping",
        ],
        "distinguishing": "concentric 'bull's-eye' rings inside each spot",
        "management": "stake plants, rotate, and apply chlorothalonil or mancozeb",
    },
    "tomato|late_blight": {
        "crop": "tomato", "disease": "late blight",
        "pathogen": "Phytophthora infestans", "pathogen_type": "oomycete",
        "symptoms": [
            "large irregular greasy grey-green blotches",
            "a pale fuzzy white mould on the underside in humid weather",
            "lesions turning brown-black and spreading rapidly",
            "dark greasy patches on stems and fruit",
        ],
        "distinguishing": "water-soaked greasy blotches with white sporulation underneath",
        "management": "destroy volunteers and apply protectant oomycete fungicides",
    },
    "tomato|septoria_leaf_spot": {
        "crop": "tomato", "disease": "Septoria leaf spot",
        "pathogen": "Septoria lycopersici", "pathogen_type": "fungus",
        "symptoms": [
            "numerous small circular spots with dark borders and tan centres",
            "tiny black fruiting bodies (pycnidia) dotting the spot centres",
            "spots most dense on lower leaves",
            "heavy speckling causing leaf yellowing and drop",
        ],
        "distinguishing": "many small spots with pin-prick black pycnidia in the centre",
        "management": "remove lower leaves and apply protectant fungicides",
    },
    "tomato|bacterial_spot": {
        "crop": "tomato", "disease": "bacterial spot",
        "pathogen": "Xanthomonas spp.", "pathogen_type": "bacterium",
        "symptoms": [
            "small dark greasy water-soaked spots",
            "spots with a faint yellow halo",
            "spots remaining small and angular, sometimes tearing",
            "scabby raised spots on green fruit",
        ],
        "distinguishing": "tiny greasy dark spots, often torn, with yellow halos",
        "management": "use clean seed and copper plus mancozeb sprays",
    },
    "tomato|yellow_leaf_curl_virus": {
        "crop": "tomato", "disease": "yellow leaf curl virus disease",
        "pathogen": "Tomato yellow leaf curl virus (whitefly-transmitted begomovirus)",
        "pathogen_type": "virus",
        "symptoms": [
            "upward cupping and curling of leaflets",
            "marked yellowing of leaf margins between the veins",
            "reduced leaf size and bushy stunted growth",
            "heavy flower drop and poor fruit set",
        ],
        "distinguishing": "cupped, yellow-margined small leaves with no spotting",
        "management": "manage whiteflies and grow tolerant hybrids",
    },
    "tomato|leaf_mold": {
        "crop": "tomato", "disease": "leaf mould",
        "pathogen": "Passalora fulva", "pathogen_type": "fungus",
        "symptoms": [
            "pale yellow-green diffuse patches on the upper surface",
            "an olive-green to brown velvety mould on the underside",
            "patches matching mould position when the leaf is flipped",
            "leaves curling and withering under high humidity",
        ],
        "distinguishing": "olive velvety mould on the underside beneath yellow upper patches",
        "management": "improve greenhouse ventilation and lower humidity",
    },
    "tomato|spider_mites": {
        "crop": "tomato", "disease": "two-spotted spider mite damage",
        "pathogen": "Tetranychus urticae", "pathogen_type": "pest",
        "symptoms": [
            "fine pale stippling speckled across the leaf",
            "a bronzed or sandblasted look on heavily fed leaves",
            "delicate webbing on the underside and between leaflets",
            "leaves drying and dropping in hot dry conditions",
        ],
        "distinguishing": "fine stippling plus silken webbing, not discrete spots",
        "management": "raise humidity and release predatory mites or use miticides",
    },
    "tomato|healthy": {
        "crop": "tomato", "disease": "no disease (healthy)",
        "pathogen": "none", "pathogen_type": "none",
        "symptoms": [
            "uniformly green leaflets with intact margins",
            "no spots, lesions, mould or curling",
            "healthy turgid posture",
            "even colour across the blade",
        ],
        "distinguishing": "clean, evenly green foliage with no symptoms",
        "management": "maintain balanced nutrition and routine monitoring",
    },

    # ------------------- POTATO / PEPPER (PlantVillage staples) ------------ #
    "potato|early_blight": {
        "crop": "potato", "disease": "early blight",
        "pathogen": "Alternaria solani", "pathogen_type": "fungus",
        "symptoms": [
            "brown spots with concentric target rings",
            "a yellow zone bordering the lesions",
            "lesions on older lower leaves first",
            "leaves yellowing and dying back",
        ],
        "distinguishing": "concentric-ringed target spots on lower leaves",
        "management": "rotate and apply protectant fungicides",
    },
    "potato|late_blight": {
        "crop": "potato", "disease": "late blight",
        "pathogen": "Phytophthora infestans", "pathogen_type": "oomycete",
        "symptoms": [
            "large greasy grey-green to brown-black blotches",
            "a white downy growth on the underside in humid weather",
            "rapidly expanding water-soaked lesions",
            "a dark dry rot on infected tubers",
        ],
        "distinguishing": "greasy spreading blotches with white sporulation underneath",
        "management": "destroy infected foliage and spray oomycete fungicides",
    },
    "potato|healthy": {
        "crop": "potato", "disease": "no disease (healthy)",
        "pathogen": "none", "pathogen_type": "none",
        "symptoms": [
            "deep green compound leaves with clean surfaces",
            "no lesions, mould or chlorosis",
            "firm upright leaflets",
            "uniform colour",
        ],
        "distinguishing": "clean green foliage with no symptoms",
        "management": "maintain good agronomy and scouting",
    },
    "pepper_bell|bacterial_spot": {
        "crop": "bell pepper", "disease": "bacterial spot",
        "pathogen": "Xanthomonas spp.", "pathogen_type": "bacterium",
        "symptoms": [
            "small water-soaked spots that turn dark brown",
            "angular lesions with slightly raised scabby surfaces",
            "a faint yellow halo around some spots",
            "spots tearing to leave ragged holes",
        ],
        "distinguishing": "raised scabby dark spots, sometimes shot-holed",
        "management": "copper sprays and clean certified seed",
    },
    "pepper_bell|healthy": {
        "crop": "bell pepper", "disease": "no disease (healthy)",
        "pathogen": "none", "pathogen_type": "none",
        "symptoms": [
            "glossy uniformly green leaves",
            "smooth intact margins",
            "no spotting or curling",
            "even healthy colour",
        ],
        "distinguishing": "clean glossy green foliage",
        "management": "routine monitoring and balanced feeding",
    },
}


# --------------------------------------------------------------------------- #
# LABEL PARSING
# --------------------------------------------------------------------------- #
# Aliases let messy folder names map onto canonical KB keys.
_CROP_ALIASES = {
    "corn": "maize", "maize": "maize", "zea": "maize",
    "cotton": "cotton", "gossypium": "cotton",
    "sugarcane": "sugarcane", "sugar_cane": "sugarcane", "cane": "sugarcane",
    "tomato": "tomato",
    "potato": "potato",
    "pepper": "pepper_bell", "pepperbell": "pepper_bell",
    "pepper_bell": "pepper_bell", "bell_pepper": "pepper_bell", "capsicum": "pepper_bell",
}

_DISEASE_ALIASES = {
    "bacterial_blight": "bacterial_blight", "angular_leaf_spot": "bacterial_blight",
    "alternaria": "alternaria_leaf_spot", "alternaria_leaf_spot": "alternaria_leaf_spot",
    "grey_mildew": "grey_mildew", "gray_mildew": "grey_mildew", "areolate_mildew": "grey_mildew",
    "leaf_curl": "leaf_curl_virus", "leaf_curl_virus": "leaf_curl_virus", "clcuv": "leaf_curl_virus",
    "red_rot": "red_rot", "redrot": "red_rot",
    "smut": "smut", "whip_smut": "smut",
    "rust": "rust", "common_rust": "common_rust",
    "mosaic": "mosaic", "mosaic_virus": "mosaic", "scmv": "mosaic",
    "northern_leaf_blight": "northern_leaf_blight", "nlb": "northern_leaf_blight",
    "turcicum": "northern_leaf_blight",
    "gray_leaf_spot": "gray_leaf_spot", "grey_leaf_spot": "gray_leaf_spot",
    "cercospora": "gray_leaf_spot", "cercospora_leaf_spot": "gray_leaf_spot",
    "leaf_blight": "leaf_blight", "blight": "leaf_blight",
    "early_blight": "early_blight",
    "late_blight": "late_blight",
    "septoria": "septoria_leaf_spot", "septoria_leaf_spot": "septoria_leaf_spot",
    "bacterial_spot": "bacterial_spot",
    "yellow_leaf_curl": "yellow_leaf_curl_virus",
    "yellow_leaf_curl_virus": "yellow_leaf_curl_virus", "tylcv": "yellow_leaf_curl_virus",    "leaf_mold": "leaf_mold", "leaf_mould": "leaf_mold",
    "spider_mites": "spider_mites", "spider_mite": "spider_mites",
    "two_spotted_spider_mite": "spider_mites", "target_spot": "spider_mites",
    "healthy": "healthy", "normal": "healthy",
}


def _normalise(text: str) -> str:
    """Lower-case, strip accents, and collapse separators to single underscores."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", text)  # split camelCase: YellowLeaf -> Yellow_Leaf
    text = text.lower()
    text = re.sub(r"[\s\-]+", "_", text)          # spaces & hyphens -> underscore
    text = re.sub(r"_{2,}", "_", text)            # collapse repeats (handles ___)
    text = re.sub(r"[^a-z0-9_]", "", text)        # drop anything else
    return text.strip("_")


def _match_alias(tokens: List[str], alias_map: Dict[str, str]) -> Optional[str]:
    """
    Try progressively shorter multi-word joins of the tokens against the alias
    map, so 'northern_leaf_blight' matches before 'leaf_blight' or 'blight'.
    """
    n = len(tokens)
    for size in range(n, 0, -1):
        for start in range(0, n - size + 1):
            candidate = "_".join(tokens[start:start + size])
            if candidate in alias_map:
                return alias_map[candidate]
    return None


def parse_label(raw_label: str) -> Optional[Dict]:
    """
    Turn a raw label such as 'Cotton___Bacterial_blight',
    'corn_(maize)___Common_rust' or 'tomato early blight' into a KB record.

    Returns the disease record dict (with 'kb_key', 'crop_slug', 'disease_slug'
    added) or None if it cannot be confidently matched.
    """
    norm = _normalise(raw_label)
    tokens = [t for t in norm.split("_") if t]
    if not tokens:
        return None

    # Drop common noise tokens that appear in PlantVillage-style names.
    noise = {"leaf", "disease", "plant", "img", "image", "the", "of", "and"}
    # We keep 'leaf' for now because some disease aliases need it; handle below.

    crop_slug = _match_alias(tokens, _CROP_ALIASES)
    disease_slug = _match_alias(tokens, _DISEASE_ALIASES)

    if crop_slug is None or disease_slug is None:
        # Second pass: strip a copy without bare noise tokens and retry disease.
        stripped = [t for t in tokens if t not in noise]
        if disease_slug is None:
            disease_slug = _match_alias(stripped, _DISEASE_ALIASES)
        if crop_slug is None:
            crop_slug = _match_alias(stripped, _CROP_ALIASES)

    if crop_slug is None or disease_slug is None:
        return None

    key = f"{crop_slug}|{disease_slug}"
    record = KB.get(key)
    if record is None:
        return None

    enriched = dict(record)
    enriched["kb_key"] = key
    enriched["crop_slug"] = crop_slug
    enriched["disease_slug"] = disease_slug
    enriched["raw_label"] = raw_label
    return enriched


def list_supported() -> List[str]:
    """Return all crop|disease keys the KB currently understands."""
    return sorted(KB.keys())


if __name__ == "__main__":
    # Quick self-test of the parser against messy real-world style labels.
    tests = [
        "Cotton___Bacterial_blight",
        "corn_(maize)___Common_rust",
        "Sugarcane - Red Rot",
        "tomato___Tomato_YellowLeaf__Curl_Virus",
        "Potato___Late_blight",
        "Pepper,_bell___healthy",
        "maize northern leaf blight",
        "garbage_label_xyz",
    ]
    for t in tests:
        rec = parse_label(t)
        if rec:
            print(f"OK   {t:45s} -> {rec['crop']} / {rec['disease']}")
        else:
            print(f"MISS {t:45s} -> (no match)")
