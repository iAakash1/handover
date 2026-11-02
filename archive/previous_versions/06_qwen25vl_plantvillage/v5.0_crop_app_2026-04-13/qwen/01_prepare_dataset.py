# 01_prepare_dataset.py
# Prepares PlantVillage captions for Qwen2.5-VL-7B fine-tuning.
# Outputs captions.json with train/val split.
# Save this file to: C:\Users\Admin\Desktop\VLM\qwen\

import json
import random
import re
from pathlib import Path

DATASET_ROOT = r"C:\Users\Admin\Desktop\VLM\PlantVillage"
OUTPUT_FILE  = "captions.json"
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ── Instruction variants (8 options, cycled by image index) ───────────────
INSTRUCTION_VARIANTS = [
    "Analyze this plant leaf image and provide a detailed diagnosis.",
    "Examine the leaf carefully and describe its condition.",
    "You are an expert plant pathologist. Diagnose this leaf.",
    "Identify the plant, its health status, and any visible symptoms.",
    "Provide a structured plant disease diagnosis for this image.",
    "What plant disease or condition is shown in this leaf image?",
    "Describe this plant leaf in terms of species, health, and symptoms.",
    "As a plant disease expert, analyze and describe this leaf image.",
]

# ── 12 structured caption templates per disease class ─────────────────────
TEMPLATES = {
    "late_blight": [
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. The leaf shows large water-soaked lesions with dark brown necrotic centers and pale green halos, indicating rapid tissue collapse.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Dark irregular greasy blotches with water-soaked margins are spreading across the leaf surface.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Significant necrotic patches with pale chlorotic borders dominate the leaf; white sporulation may be visible on the underside.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Rapidly expanding brown-black lesions with faint green halos and collapsed tissue indicate advanced late blight.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Greasy dark lesions with irregular edges surrounded by chlorotic tissue are spreading across the {plant} leaf.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Water-soaked spots transitioning to dark necrotic tissue with pale margins dominate the leaf.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Large coalescing dark patches with pale to yellow borders indicate severe Phytophthora infection.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Irregular black-brown lesions spreading inward from the leaf margins typify P. infestans infection.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Collapsed water-soaked tissue forming dark patches with pale surrounding zones and wilting leaf edges indicate active late blight.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Progressive dark lesions with pale halos and wilted tissue edges show moderate to severe late blight progression.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. Dark necrotic regions with a greasy sheen and indistinct margins are spreading rapidly across this diseased {plant} leaf.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Pathogen: Phytophthora infestans. The leaf shows irregular water-soaked to necrotic lesions with a characteristic oil-stained appearance from late blight.",
    ],
    "early_blight": [
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. The leaf shows circular brown lesions with distinctive concentric rings forming a target-like pattern surrounded by yellow chlorotic zones.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Target-shaped dark brown spots with concentric rings and yellow halos are scattered across the {plant} leaf surface.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Bull-eye patterned lesions with dark centers and lighter concentric bands surrounded by yellowing tissue indicate Alternaria early blight.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Multiple circular spots with dark brown concentric rings and chlorotic margins are visible on this {plant} leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Classic early blight presents as target-like spots with angular dark centers and surrounding chlorotic tissue on the leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. The leaf displays several necrotic spots with concentric ring patterns typical of Alternaria solani infection.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Dark brown lesions with concentric banding and yellow margins characterize this early blight infection on the {plant} leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Irregular brown spots with lighter concentric zones and yellow surrounding tissue distributed across the leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Target-pattern lesions with dark brown rings and chlorotic halos are hallmark early blight symptoms on this leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Circular necrotic areas with concentric dark and light banding surrounded by chlorosis indicate early blight.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Angular to circular lesions with target-like ring patterns and yellowing surrounding tissue indicate Alternaria infection.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Pathogen: Alternaria solani. Dry brown lesions with concentric zonation and yellow borders scattered across the leaf mark this as early blight.",
    ],
    "bacterial_spot": [
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Small water-soaked lesions have turned dark brown with distinct yellow halos scattered irregularly across the leaf surface.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. The leaf displays numerous irregular brown spots surrounded by yellow margins, consistent with Xanthomonas infection.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Multiple dark lesions with a water-soaked greasy appearance and chlorotic halos are distributed across this {plant} leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Angular brown spots bounded by leaf veins with surrounding yellowing indicate bacterial spot disease.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Dark necrotic spots with yellow borders and irregular margins are characteristic of bacterial spot on this {plant} leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. The leaf shows small raised dark lesions with water-soaked centers and chlorotic halos caused by Xanthomonas infection.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Scattered dark brown spots with yellow surrounding tissue and irregular shapes cover portions of this {plant} leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Water-soaked spots transitioning to brown necrotic lesions with yellow halos indicate bacterial spot across the leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Multiple small angular lesions with dark centers and chlorotic margins indicate bacterial spot infection on this leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. The {plant} leaf exhibits irregularly shaped dark spots with greasy margins and surrounding yellowing from bacterial infection.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Small to medium raised lesions with dark necrotic cores and yellow halos indicate moderate Xanthomonas infection.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Pathogen: Xanthomonas spp. Angular vein-delimited lesions with dark centers and chlorotic halos distributed across this infected {plant} leaf.",
    ],
    "healthy": [
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. The leaf displays uniform vibrant green coloration with no visible lesions, spots, discoloration, or any signs of disease or stress.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. This {plant} leaf appears completely healthy showing consistent green pigmentation, smooth texture, and no abnormalities whatsoever.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. No disease symptoms are visible. Normal healthy growth is indicated by even green color and intact, undamaged leaf structure.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. The {plant} leaf shows excellent health with rich green coloration, no spots or lesions, and normal leaf morphology throughout.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. Uniform green coloration and absence of any lesions, spots, or discoloration confirm this {plant} leaf is in a healthy condition.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. The leaf surface is clean and uniformly green with no visible fungal growth, bacterial lesions, or viral symptoms present.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. Healthy leaf tissue is indicated by consistent green pigmentation and a smooth surface with no necrotic or chlorotic areas visible.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. This {plant} leaf is free from any disease symptoms, showing normal venation, deep green pigmentation, and intact margins.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. No pathological symptoms observed. The leaf exhibits vigorous healthy appearance with uniform coloration and no structural damage.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. The {plant} leaf is in optimal health showing deep green color, no discoloration, and complete absence of disease indicators.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. Full leaf blade with uniform chlorophyll-rich coloration, intact margins, and no evidence of pathogen activity or mechanical damage.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Pathogen: None. Well-developed healthy {plant} leaf with symmetrical structure, rich green color, and absence of any disease-associated visual abnormalities.",
    ],
    "leaf_mold": [
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Pale yellow spots on the upper leaf surface correspond to olive-green velvety fungal growth on the underside.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. The leaf shows chlorotic patches above with a distinctive olive-colored mold coating beneath, indicating Passalora leaf mold.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Yellow to pale green discoloration on the upper surface with corresponding fungal sporulation below characterizes this leaf mold infection.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Irregular yellow blotches on the upper leaf surface and velvety olive-green growth on the lower surface indicate leaf mold.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. The {plant} leaf exhibits chlorotic spots above with a felt-like olive-brown fungal coating on the corresponding lower surface.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Pale yellow patches visible from above and dense fungal sporulation beneath indicate moderate leaf mold infection.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Upper leaf surface shows faint yellowing while the underside displays characteristic olive-green mold growth from Passalora fulva.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Chlorotic areas on the adaxial surface with corresponding abaxial fungal sporulation are visible on this {plant} leaf.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. The leaf displays yellow irregular spots on top and velvety dark sporulation beneath, consistent with leaf mold disease.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Diffuse yellowing on the upper surface and olive-green mold patches on the underside characterize this Passalora infection.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Prominent chlorotic lesions on the leaf top with a velvet-textured grayish-green fungal coating on the reverse indicate leaf mold.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pathogen: Passalora fulva. Irregular pale green to yellow patches on the upper leaf and densely sporulating olive mold on the lower surface.",
    ],
    "septoria": [
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Small circular spots with dark brown borders, lighter tan centers, and tiny black pycnidia dots are scattered across the leaf.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Numerous small lesions with dark margins and lighter centers containing visible black fruiting bodies characterize this infection.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Multiple circular spots with pycnidia at their centers are spread across the leaf, consistent with Septoria lycopersici.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. The leaf shows many small round spots with dark borders, gray-tan centers, and minute black pycnidia within each lesion.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Tiny circular lesions with dark edges and pale tan centers containing black pycnidia distributed across this {plant} leaf.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Characteristic septoria spots appear as small circles with dark brown margins and gray centers dotted with pycnidia.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. The {plant} leaf displays numerous small round lesions with dark borders and lighter centers showing black fruiting bodies.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Small circular necrotic spots with dark halos and tan interiors containing pycnidia indicate septoria leaf spot.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Dense spots with dark rings and pale centers scattered across the leaf surface indicate Septoria lycopersici infection.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Round lesions with dark brown edges and grayish centers with visible black pycnidia spread across this leaf.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Abundant small lesions with sooty black pycnidia embedded in tan interiors and dark raised margins cover the leaf.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Pathogen: Septoria lycopersici. Circular spots with very dark margins, pale tan centers, and prominent pycnidia densely distributed across the {plant} leaf.",
    ],
    "spider_mites": [
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. Fine stippling and yellow speckling cover the leaf surface with delicate webbing possibly visible between leaf structures.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. The leaf shows a bronzed dusty appearance with tiny yellow feeding dots from two-spotted spider mite damage.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. Extensive stippling creates a silvery mottled pattern across the {plant} leaf from sustained spider mite feeding.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. Thousands of tiny puncture marks cause a speckled yellow-bronze discoloration across the upper leaf surface.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. The upper leaf surface appears pale and stippled with yellow dots while fine silk webbing may be present beneath.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. Mite feeding has caused a characteristic salt-and-pepper stippled pattern with overall leaf bronzing visible.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. The {plant} leaf shows chlorotic stippling and bronzing from Tetranychus urticae feeding across the leaf surface.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. Diffuse yellow speckling and a dusty bronzed appearance across the leaf indicate significant spider mite infestation.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. Fine yellow dots and possible webbing from two-spotted spider mites distributed across this {plant} leaf.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. The leaf surface displays pale stippling and slight bronzing consistent with moderate to heavy spider mite damage.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. Uniform bronzed stippling from mite punctures spread across the blade with chlorotic speckles and slight crinkling.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Pathogen: Tetranychus urticae. Heavy stippling producing a sand-blasted appearance on the leaf upper surface indicates severe T. urticae infestation.",
    ],
    "target_spot": [
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Brown circular lesions with concentric rings and yellow surrounding tissue form a distinct target pattern on the leaf.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Ringed brown spots with chlorotic margins indicate target spot disease caused by Corynespora cassiicola.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Distinct bull-eye patterned lesions surrounded by yellow tissue are visible across the infected {plant} leaf.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Circular brown spots with concentric lighter zones and chlorotic halos are scattered on this leaf surface.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. The leaf displays target-like spots with alternating dark and light concentric bands surrounded by yellowing tissue.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Multiple ringed lesions with dark centers and lighter concentric zones surrounded by chlorosis indicate target spot.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. The {plant} leaf shows circular necrotic spots with ring patterns and yellow margins from Corynespora infection.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Dark centered bull-eye spots with concentric lighter rings and chlorotic borders are present on this leaf.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Characteristic ringed lesions resembling a target with brown bands and yellow halos cover portions of this leaf.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Circular lesions with alternating dark and light concentric bands and yellow surrounding areas indicate target spot.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Concentric ring lesions with dry brown interiors and chlorotic halos scattered across this {plant} leaf indicate Corynespora infection.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Pathogen: Corynespora cassiicola. Large and small ringed brown lesions with distinct concentric zonation and yellow margin tissue visible on the leaf.",
    ],
    "mosaic_virus": [
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). Irregular mosaic patterning of alternating light and dark green patches with leaf distortion and crumpling is visible.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). The leaf shows a mottled appearance with alternating yellow-green and dark green zones and slight distortion.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). Systemic viral infection has caused mosaic discoloration with irregular green islands and leaf surface distortion.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). Characteristic mosaic symptoms appear as irregular light and dark green mottling with mild leaf curling.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). The {plant} leaf displays a patchwork of pale and dark green areas with slight wrinkling from mosaic virus.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). Irregular alternating light and dark green mosaic pattern with leaf distortion indicates viral mosaic infection.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). The leaf surface shows chlorotic and dark green mosaic patterning with slight rugosity from viral infection.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). Mosaic discoloration with intermingled pale and deep green patches and mild leaf deformation is present.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). Systemic mosaic virus symptoms including mottled coloration and leaf distortion are evident on this {plant} leaf.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). The leaf displays irregular green mosaic patterning with alternating chlorotic patches from viral mosaic infection.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). Mosaic of bright and dull green patches with crinkling and leaf narrowing visible on the distorted {plant} leaf.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Pathogen: Tomato Mosaic Virus (ToMV). The blade shows a blistered mosaic surface with interveinal chlorosis and dark-green virus-infected islands.",
    ],
    "yellow_leaf_curl": [
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Severe upward curling of leaf margins with intense yellowing and stunted crinkled appearance throughout.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). The leaf is severely distorted with upward cupping edges and prominent chlorosis typical of TYLCV infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Extreme chlorosis and upward leaf cupping with crinkled texture indicate yellow leaf curl viral infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). The {plant} leaf shows severe yellowing and upward curling with reduced size from TYLCV infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Strong upward leaf rolling with intense yellowing and stunted growth characterize this TYLCV infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Leaf margins curl upward severely with chlorotic discoloration and distorted crinkled surface from viral infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Yellowing and upward cupping of the {plant} leaf with small leaf size indicate yellow leaf curl virus.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Severe chlorosis combined with upward leaf curling and reduced blade size are visible on this infected leaf.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). The leaf displays intense yellowing and strong upward rolling with distorted surface texture from TYLCV.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Upward leaf curl and bright yellow discoloration with crinkled texture indicate severe yellow leaf curl virus.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Uniformly yellowed leaf with strongly upward-curled margins and reduced leaf size from begomovirus TYLCV.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Pathogen: Tomato Yellow Leaf Curl Virus (TYLCV). Entire leaf shows interveinal yellowing and strong inward rolling of the margin from systemic TYLCV infection.",
    ],
}

SEVERITY_MAP = {
    "bacterial_spot":   "Moderate to Severe",
    "early_blight":     "Mild to Moderate",
    "late_blight":      "Moderate to Severe",
    "leaf_mold":        "Mild",
    "septoria":         "Moderate",
    "spider_mites":     "Mild to Moderate",
    "target_spot":      "Moderate",
    "mosaic_virus":     "Severe",
    "yellow_leaf_curl": "Severe",
    "yellowleaf":       "Severe",
    "healthy":          "None",
}


def parse_folder_name(folder_name):
    if "___" in folder_name:
        parts = folder_name.split("___", 1)
    elif "__" in folder_name:
        parts = folder_name.split("__", 1)
    else:
        parts = folder_name.split("_", 1)
    plant     = re.sub(r"_+", " ", parts[0]).strip()
    condition = re.sub(r"_+", " ", parts[1]).strip() if len(parts) > 1 else "Unknown"
    return plant, condition


def get_severity(key):
    for sev_key, sev_val in SEVERITY_MAP.items():
        if sev_key in key:
            return sev_val
    return "Moderate"


def get_templates(key):
    for map_key, tmpl_list in TEMPLATES.items():
        if map_key in key or key in map_key:
            return tmpl_list
    return [
        "Plant: {plant}. Condition: {cond}. Severity: {sev}. Pathogen: Unknown. The leaf shows visible disease symptoms including discoloration and tissue damage consistent with {cond} infection.",
        "Plant: {plant}. Condition: {cond}. Severity: {sev}. Pathogen: Unknown. This leaf exhibits signs of {cond} with abnormal coloration and visible lesions present on the surface.",
        "Plant: {plant}. Condition: {cond}. Severity: {sev}. Pathogen: Unknown. Disease symptoms including lesions and discoloration are present on this {plant} leaf, consistent with {cond}.",
        "Plant: {plant}. Condition: {cond}. Severity: {sev}. Pathogen: Unknown. The {plant} leaf displays characteristic symptoms of {cond} with visible tissue damage and abnormal pigmentation.",
        "Plant: {plant}. Condition: {cond}. Severity: {sev}. Pathogen: Unknown. Visible signs of {cond} including discoloration and lesions are observed on the surface of this leaf.",
    ]


def build_caption(plant, condition, img_index):
    key       = condition.lower().replace(" ", "_")
    sev       = get_severity(key)
    templates = get_templates(key)
    template  = templates[img_index % len(templates)]
    return template.format(plant=plant, cond=condition, sev=sev)


def build_instruction(img_index):
    return INSTRUCTION_VARIANTS[img_index % len(INSTRUCTION_VARIANTS)]


def main():
    dataset_path = Path(DATASET_ROOT)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {DATASET_ROOT}")
        return

    class_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
    print(f"Found {len(class_folders)} class folders\n")

    records = []
    for class_folder in class_folders:
        plant, condition = parse_folder_name(class_folder.name)
        images = [f for f in class_folder.iterdir() if f.suffix in VALID_EXTS]
        if not images:
            print(f"WARNING: No images in {class_folder.name}")
            continue

        for img_index, img_path in enumerate(sorted(images)):
            caption     = build_caption(plant, condition, img_index)
            instruction = build_instruction(img_index)
            records.append({
                "image":       str(img_path.resolve()),
                "caption":     caption,
                "instruction": instruction,
                "plant":       plant,
                "condition":   condition,
                "class_name":  class_folder.name,
            })

        key    = condition.lower().replace(" ", "_")
        n_tmpl = len(get_templates(key))
        print(f"{class_folder.name}: {len(images)} images | {n_tmpl} templates")

    random.seed(42)
    random.shuffle(records)
    split         = int(len(records) * 0.9)
    train_records = records[:split]
    val_records   = records[split:]

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"train": train_records, "val": val_records}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Total images : {len(records)}")
    print(f"Train        : {len(train_records)}")
    print(f"Val          : {len(val_records)}")
    print(f"Saved to     : {OUTPUT_FILE}")

    print(f"\nCaption variation check (Late Blight, first 5):")
    lb = [r for r in records if "late_blight" in r["class_name"].lower()][:5]
    for i, r in enumerate(lb):
        print(f"  [{i+1}] {r['caption'][:90]}...")


if __name__ == "__main__":
    main()
