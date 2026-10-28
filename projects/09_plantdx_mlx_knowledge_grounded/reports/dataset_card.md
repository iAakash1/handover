# PlantDx Dataset Audit — Dataset Card

- Generated: `2026-07-12T15:01:54+00:00`
- Tool: `plantdx.audit` (plantdx 0.1.0)
- Config hash: `f17c99b6075e34d5`
- Audit checksum: `969c6c8de8238fbad9ff229a4f042064367beee70a99fa97521f0aead39f87c3`

## Totals

- Images: **58305** (ok: 58305, corrupt: 0, unsupported: 0)
- Classes (across datasets): 46
- Exact-duplicate groups: 42 · near-duplicate groups: 0
- Split status: not_performed — Dataset splitting is a later milestone; the audit only inventories images.

## tomato — PlantVillage

- Root: `/Users/aakashjawle/Desktop/experiments/tomato/raw/PlantVillage` (exists: True)
- Images: 54305 (ok: 54305, corrupt: 0, unsupported: 0)
- Classes discovered: 38 (config expects: 10)
- **Width (px):** min=256, max=256, mean=256.0
- **Height (px):** min=256, max=256, mean=256.0
- **Aspect ratio:** min=1.0, max=1.0, mean=1.0
- Class imbalance (max/min): 36.23
- Dataset checksum: `7c644d5cef461efe3994a830c6deb9869bd7f8c9c841c769b832f3b702b0eff3`

| class | images |
|-------|--------|
| Apple___Apple_scab | 630 |
| Apple___Black_rot | 621 |
| Apple___Cedar_apple_rust | 275 |
| Apple___healthy | 1645 |
| Blueberry___healthy | 1502 |
| Cherry_(including_sour)___Powdery_mildew | 1052 |
| Cherry_(including_sour)___healthy | 854 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 513 |
| Corn_(maize)___Common_rust_ | 1192 |
| Corn_(maize)___Northern_Leaf_Blight | 985 |
| Corn_(maize)___healthy | 1162 |
| Grape___Black_rot | 1180 |
| Grape___Esca_(Black_Measles) | 1383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1076 |
| Grape___healthy | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 5507 |
| Peach___Bacterial_spot | 2297 |
| Peach___healthy | 360 |
| Pepper,_bell___Bacterial_spot | 997 |
| Pepper,_bell___healthy | 1478 |
| Potato___Early_blight | 1000 |
| Potato___Late_blight | 1000 |
| Potato___healthy | 152 |
| Raspberry___healthy | 371 |
| Soybean___healthy | 5090 |
| Squash___Powdery_mildew | 1835 |
| Strawberry___Leaf_scorch | 1109 |
| Strawberry___healthy | 456 |
| Tomato___Bacterial_spot | 2127 |
| Tomato___Early_blight | 1000 |
| Tomato___Late_blight | 1909 |
| Tomato___Leaf_Mold | 952 |
| Tomato___Septoria_leaf_spot | 1771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1676 |
| Tomato___Target_Spot | 1404 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5357 |
| Tomato___Tomato_mosaic_virus | 373 |
| Tomato___healthy | 1591 |

**Issues (2):** (full detail in the CSV reports)
- `class_count_mismatch`: discovered 38 classes; config expects 10
- `class_imbalance`: max/min class ratio = 36.23

## mango — MangoLeafBD

- Root: `/Users/aakashjawle/Desktop/experiments/mango/raw/MangoLeafBD` (exists: True)
- Images: 4000 (ok: 4000, corrupt: 0, unsupported: 0)
- Classes discovered: 8 (config expects: 8)
- **Width (px):** min=187, max=320, mean=274.9235
- **Height (px):** min=70, max=320, mean=260.2715
- **Aspect ratio:** min=0.5844, max=4.5714, mean=1.0856
- Class imbalance (max/min): 1.0
- Dataset checksum: `9161ad1ef44c2c450bf8657ba5aa72a30ffe7c0ea35da234bfac001bb0d4f94c`

| class | images |
|-------|--------|
| Anthracnose | 500 |
| Bacterial Canker | 500 |
| Cutting Weevil | 500 |
| Die Back | 500 |
| Gall Midge | 500 |
| Healthy | 500 |
| Powdery Mildew | 500 |
| Sooty Mould | 500 |
