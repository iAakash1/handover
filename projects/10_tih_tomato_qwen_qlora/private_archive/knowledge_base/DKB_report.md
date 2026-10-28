# PlantDx — Disease Knowledge Base (DKB)

**Output 1 of 2 — Human-readable report.**
Companion machine-readable file: [`dkb.json`](dkb.json).

Version: 1.0 · Compiled: 2026-07-11 · Scope: 10 tomato classes (PlantVillage) + 8 mango classes (MangoLeafBD) = 18 labels.

---

## 0. Purpose, scope, and how to read this document

This Disease Knowledge Base is the **single source of truth** for the PlantDx project. It is a *curated compilation of established plant-pathology knowledge*, grounded in authoritative sources (APS, university extension services, CABI, FAO, and peer-reviewed literature). It contains **no image analysis, no synthetic labels, and no VLM-generated content.** The dataset labels (PlantVillage, MangoLeafBD) are treated as ground truth; this document supplies the *scientific description* attached to each label.

Every downstream stage (caption generation, caption validation, instruction-tuning-dataset construction, fine-tuning, evaluation, and the paper) should draw its vocabulary and factual claims from here.

### 0.1 The single most important constraint: leaf-only observability

Both corpora are **single-leaf images**:

- **PlantVillage (tomato):** individual detached/attached leaflets or compound-leaf sections on a plain, near-uniform background, photographed under controlled lighting. Fruit, stems, flowers, whole plants, roots, and field context are **not present**.
- **MangoLeafBD (mango):** single mango leaves (≈240×320 px), photographed with mobile phones; ~1,800 distinct leaves augmented by zoom/rotation to 4,000 images across 8 classes (500 each). Again, **only the leaf is visible**.

Therefore this DKB explicitly separates two kinds of facts for every class:

1. **Leaf-observable features** — what can legitimately appear in a caption of a single leaf image (lesions, color, texture, margins, distribution on the lamina).
2. **Context features** — fruit lesions, twig/branch cankers, blossom blight, gummosis, whole-tree decline, yield loss, vascular streaking, insect adults, etc. These are documented for completeness and differential diagnosis but are listed under **"Symptoms That SHOULD NOT Be Mentioned"** because they are *not visible in a single-leaf image* and must never be asserted in a caption.

This distinction is what makes the resulting captions *grounded*. A caption that says "star-shaped gummy fruit lesions" for a mango bacterial-canker **leaf** image is factually about the disease but **false about the image**, and must be forbidden.

### 0.2 Pathogen vs. pest, and secondary organisms

Several classes are **not diseases caused by a pathogen infecting leaf tissue**, and the DKB flags this because it changes the correct vocabulary:

- **Tomato "Spider Mites"** — feeding injury by the arthropod *Tetranychus urticae* (a mite). Correct terms: *stippling, bronzing, webbing* — **not** "lesion / infection / pathogen."
- **Mango "Cutting Weevil"** — mechanical feeding/cutting damage by the beetle *Deporaus marginatus*. Correct terms: *cut, notched, windowpane* — **not** "lesion / spot."
- **Mango "Gall Midge"** — larval galls induced by the fly *Procontarinia matteiana*. Correct terms: *raised galls, wart-like bumps* — **not** "flat lesion / spot."
- **Mango "Sooty Mould"** — a **superficial saprophytic** fungal film growing on insect honeydew; the underlying leaf tissue is *not* infected. Correct terms: *superficial black coating, wipeable* — **not** "necrotic lesion / tissue death."

### 0.3 Source hierarchy and citation policy

Preferred sources, in order: American Phytopathological Society (APS) compendia and journals → University Extension (UC IPM, UF/IFAS EDIS, Cornell, Penn State, UMN, CTAHR) → CABI / FAO / national horticulture boards / national plant-health agencies → peer-reviewed journals. Blogs, commercial retail sites, and AI-generated pages were avoided. Full citations are in [§3 References](#3-references). Per-class reference keys resolve there.

### 0.4 Documented taxonomic disagreements

Where authoritative sources disagree, the DKB states the disagreement rather than silently picking one (per project policy):

- **Tomato bacterial spot** is caused by a **four-taxon species complex**, and the naming has been revised repeatedly. Current consensus (Constantin et al. 2016; Timilsina et al. 2019; APS diagnostic guide 2022) recognizes: *Xanthomonas euvesicatoria* pv. *euvesicatoria*, *X. euvesicatoria* pv. *perforans*, *X. hortorum* pv. *gardneri*, and *X. vesicatoria*. Older literature uses *X. campestris* pv. *vesicatoria*, or the 1995 Stall split (*X. vesicatoria, X. euvesicatoria, X. gardneri, X. perforans*). PlantVillage's single "Bacterial spot" label does not distinguish species; captions should therefore stay at the disease level, not name a species.
- **Tomato leaf mold** pathogen name is unsettled. Videira et al. (2017, *Studies in Mycology*) adopt **_Fulvia fulva_**; the widely used synonyms are **_Passalora fulva_** and **_Cladosporium fulvum_**. Some post-2017 papers revert to *Cladosporium fulvum*. This DKB uses *Fulvia fulva* as the accepted name with both synonyms listed.
- **Mango "Bacterial Canker"** (MangoLeafBD label) is the **same disease** most literature calls **bacterial black spot**, caused by *Xanthomonas citri* pv. *mangiferaeindicae* (syn. *X. campestris* pv. *mangiferaeindicae*). The two names are used interchangeably in the field.
- **Mango "Die Back"** is, in the full literature, a **twig/branch dieback and tree-decline syndrome** (Botryosphaeriaceae, chiefly *Lasiodiplodia theobromae*). As a MangoLeafBD **leaf** class it captures the *leaf-level* expression (marginal/tip scorch, yellowing, curling, drying) of shoots affected by that syndrome. Captions must describe only the leaf expression.
- **Mango powdery mildew / sooty mould** anamorph-based naming (*Oidium mangiferae*; *Capnodium* spp.) is standard; teleomorph assignments vary in the literature and are not needed at caption level.

### 0.5 Field glossary (used consistently below)

- **Chlorosis** — yellowing from loss of chlorophyll. **Necrosis** — tissue death (brown/black, dry). **Lesion** — a localized area of diseased tissue. **Pycnidia** — tiny dark asexual fruiting bodies (visible as pinpoint black dots), diagnostic for *Septoria*. **Stippling** — fine pale flecking from mite/insect cell-feeding. **Gall** — a raised swelling of plant tissue induced by an organism. **Shot-hole** — small holes left when necrotic lesion centers drop out. **Coriaceous** — leathery. **Acervulus/acervuli** — a fruiting structure of *Colletotrichum* (anthracnose).

---

## 1. TOMATO — PlantVillage classes (*Solanum lycopersicum*)

> Reference keys used in this section: `UCIPM-Tomato`, `APS-CompTomato`, `APS-BSGuide`, `UFIFAS-PP351`, `Videira2017`, `UMN-LeafMold`, `Constantin2016`, `Timilsina2019`, `UCIPM-Mites`, `Agrios2005`. See §3.

### 1.0 Healthy (tomato)

- **Common name:** Healthy tomato foliage
- **Causal agent / scientific name:** None (no pathogen or pest)
- **Pathogen type:** N/A · **Pathogen family:** N/A
- **Host plant:** *Solanum lycopersicum*
- **Typical environmental conditions:** N/A (well-watered, unstressed foliage)
- **Disease progression:** N/A

**Symptoms (leaf-observable)**
- **Primary symptoms:** Uniform green, intact pinnately compound leaf; serrated/lobed leaflet margins; prominent midrib and pinnate venation; slight surface pubescence (trichomes).
- **Secondary symptoms:** None.
- **Leaf color changes:** Uniform medium-to-deep green (adaxial), slightly paler abaxial; no discoloration.
- **Lesion morphology / shape / size / distribution:** None — no lesions present.
- **Leaf margin changes:** Intact, normally serrated; no scorch, notching, or cutting.
- **Leaf curling:** None (leaflets flat to gently arched).
- **Necrosis / chlorosis:** None.
- **Texture changes:** Normal turgid, slightly pubescent surface.

**Severity staging**
- **Early / Moderate / Severe:** Not applicable (healthy).
- **Severity indicators:** Absence of any lesion, discoloration, curling, webbing, coating, or cutting.

**Differential diagnosis**
- **Confused with:** Very early-stage disease, mild nutrient deficiency, or photographic/lighting artifacts.
- **Key differentiating features:** Complete absence of pathological signs; uniform coloration.
- **Diagnostic visual features:** Uniform green lamina, intact margins, clean surface.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Any disease/pest term — spots, lesions, yellowing, mottling, webbing, coating, galls, cutting.
- **Recommended controlled vocabulary:** healthy, green, uniform, intact, unblemished, smooth, normal venation.
- **Recommended synonyms:** healthy leaf, disease-free foliage, normal leaf.
- **Recommended adjectives:** green, uniform, healthy, intact, unblemished, smooth, turgid.
- **Forbidden adjectives:** diseased, infected, necrotic, chlorotic, spotted, blighted, mottled, curled, webbed, coated, galled.
- **Recommended caption vocabulary:** "a healthy tomato leaflet with uniform green coloration and intact margins."
- **Severity vocabulary:** none / not applicable. · **Color vocabulary:** green, deep green, medium green. · **Shape vocabulary:** serrated, lobed, pinnate. · **Texture vocabulary:** smooth, pubescent, turgid.

**Management** — Standard cultural practice and routine scouting; no treatment required.

**References:** `UCIPM-Tomato`, `APS-CompTomato`, `Agrios2005`.

---

### 1.1 Bacterial Spot (tomato)

- **Common name:** Bacterial spot (bacterial leaf spot)
- **Causal agent / scientific name:** *Xanthomonas* spp. — four-taxon complex: *X. euvesicatoria* pv. *euvesicatoria*, *X. euvesicatoria* pv. *perforans*, *X. hortorum* pv. *gardneri*, *X. vesicatoria* (see §0.4). PlantVillage label does not resolve species.
- **Pathogen type:** Bacterium (Gram-negative)
- **Pathogen family:** Xanthomonadaceae
- **Host plant:** Tomato and pepper
- **Typical environmental conditions:** Warm (24–30 °C), wet/humid; spread by rain splash, wind-driven rain, overhead irrigation, contaminated seed and transplants; enters via stomata and wounds.
- **Disease progression:** Seed/transplant-borne inoculum → small water-soaked leaf spots → dark angular necrotic lesions → coalescence → leaf blighting and defoliation.

**Symptoms (leaf-observable)**
- **Primary symptoms:** Numerous small dark-brown to black, greasy/water-soaked spots; **angular** (often vein-delimited); frequently ringed by a **yellow (chlorotic) halo**.
- **Secondary symptoms:** Coalescence into irregular necrotic blotches; **shot-hole**/tattered look as centers dry and drop; marginal scorch; defoliation.
- **Leaf color changes:** Yellow halos around spots; general yellowing when severe.
- **Lesion morphology:** Water-soaked → necrotic, greasy, dark; **no** concentric rings; **no** pycnidia or fungal growth.
- **Lesion shape:** Small, circular-to-angular. · **Lesion size:** ~1–3 mm (usually < 3 mm). · **Lesion distribution:** Scattered, numerous, both surfaces, often concentrated toward tips/margins.
- **Leaf margin changes:** Marginal necrosis/scorch when lesions coalesce at margins.
- **Leaf curling:** Minimal; slight marginal distortion when severe.
- **Necrosis:** Small dark necrotic centers, coalescing. · **Chlorosis:** Yellow halos; diffuse yellowing when severe.
- **Texture changes:** Slightly sunken to flat, greasy/water-soaked early lesions.

**Severity staging**
- **Early:** A few tiny water-soaked dark specks with faint halos.
- **Moderate:** Many dark angular spots with distinct yellow halos; some coalescence.
- **Severe:** Extensive coalesced necrosis, marginal scorch, shot-hole, defoliation.
- **Severity indicators:** Spot count/density, degree of coalescence, extent of marginal scorch and defoliation.

**Differential diagnosis**
- **Confused with:** Bacterial speck (*Pseudomonas syringae* pv. *tomato* — smaller, darker specks with larger halos), early blight (larger, concentric rings), Septoria leaf spot (gray centers + pycnidia), target spot (concentric rings).
- **Key differentiating features:** Many **small angular** dark greasy spots **with yellow halos**; **no** concentric rings, **no** pycnidia; shot-hole.
- **Diagnostic visual features:** Small dark angular water-soaked spots + chlorotic halos + shot-hole.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Fruit spots/scabbing, concentric/target rings, pycnidia, fungal mold/mycelium, webbing, mosaic/mottling (none apply to this disease or are not leaf-visible).
- **Recommended controlled vocabulary:** dark spots, angular lesions, water-soaked, greasy, chlorotic halo, shot-hole, coalescing, scattered.
- **Recommended synonyms:** bacterial leaf spot, bacterial spotting.
- **Recommended adjectives:** small, angular, dark-brown, black, greasy, water-soaked, halo-bordered, scattered, numerous.
- **Forbidden adjectives:** concentric, target-like, ring-patterned, powdery, velvety, moldy, webbed, mosaic, mottled.
- **Recommended caption vocabulary:** "small dark angular spots with yellow halos scattered across the leaflet."
- **Severity vocabulary:** few / scattered / numerous / coalescing / extensive. · **Color vocabulary:** dark-brown, black, yellow (halo). · **Shape vocabulary:** angular, circular, irregular (coalesced). · **Texture vocabulary:** greasy, water-soaked, sunken.

**Management**
- **Practices:** Certified pathogen-free seed and transplants; crop rotation; sanitation; avoid overhead irrigation and working plants when wet.
- **Treatment:** Copper-based bactericides (often + mancozeb) preventively; efficacy limited by copper-tolerant strains.
- **Prevention:** Resistant/tolerant varieties where available; seed treatment (hot water/acid); rogue infected transplants.

**References:** `APS-BSGuide`, `Constantin2016`, `Timilsina2019`, `UCIPM-Tomato`, `APS-CompTomato`.

---

### 1.2 Early Blight (tomato)

- **Common name:** Early blight
- **Causal agent / scientific name:** *Alternaria solani* (tomato-specialized *A. linariae* / *A. tomatophila* also reported)
- **Pathogen type:** Fungus · **Pathogen family:** Pleosporaceae
- **Host plant:** Tomato, potato, other solanaceous hosts
- **Typical environmental conditions:** Warm (24–29 °C), high humidity, dew, alternating wet/dry; favored by plant stress; survives in residue and soil.
- **Disease progression:** Oldest/lowest leaves first → circular lesions enlarge with concentric rings → leaflet chlorosis → upward defoliation (also stem "collar rot" and fruit — not leaf-visible).

**Symptoms (leaf-observable)**
- **Primary symptoms:** Brown-to-dark-brown circular lesions with characteristic **concentric rings ("target"/"bull's-eye")**, often surrounded by a **yellow halo**; begin on oldest/lowest leaves.
- **Secondary symptoms:** Extensive leaflet chlorosis; premature defoliation from the bottom up.
- **Leaf color changes:** Yellowing around and between lesions; whole leaflet yellows and dies.
- **Lesion morphology:** Necrotic with concentric rings; sometimes dark velvety sporulation in the center.
- **Lesion shape:** Circular to oval/angular. · **Lesion size:** ~3–12+ mm, enlarging. · **Lesion distribution:** Lower/older leaves first; scattered then coalescing.
- **Leaf margin changes:** Not margin-specific; entire leaflet can senesce.
- **Leaf curling:** Affected leaflets distort as they senesce.
- **Necrosis:** Prominent dark necrotic zones with rings. · **Chlorosis:** Pronounced yellow halos and general leaflet yellowing.
- **Texture changes:** Dry, papery; occasionally velvety (sporulation) in lesion centers.

**Severity staging**
- **Early:** One or a few small dark spots on lower leaves; faint rings.
- **Moderate:** Enlarging target-pattern lesions with yellow halos; some leaflet yellowing.
- **Severe:** Coalesced lesions, extensive leaflet blighting, bottom-up defoliation.
- **Severity indicators:** Number/size of ringed lesions, extent of chlorosis and defoliation, height of upward progression.

**Differential diagnosis**
- **Confused with:** Target spot (*Corynespora* — finer/subtler rings, distributed through canopy), Septoria (smaller, gray centers + pycnidia), late blight (water-soaked, no rings), bacterial spot.
- **Key differentiating features:** Coarse **concentric "bull's-eye" rings + yellow halo + lower-leaf-first** distribution.
- **Diagnostic visual features:** Target/bull's-eye concentric rings with yellow halo on older leaves.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Fruit (stem-end) lesions, stem collar rot, pycnidia, white underside sporulation (that is late blight), webbing, mosaic.
- **Recommended controlled vocabulary:** concentric rings, target-like, bull's-eye, yellow halo, brown lesion, necrotic, coalescing.
- **Recommended synonyms:** target-spot pattern (descriptive only — do not confuse with the Target Spot *disease*), *Alternaria* leaf blight.
- **Recommended adjectives:** concentric, target-like, ringed, brown, dark, halo-bordered, circular.
- **Forbidden adjectives:** water-soaked, greasy, powdery, velvety (as coating), webbed, mosaic, mottled, angular-only.
- **Recommended caption vocabulary:** "brown circular lesions with concentric rings and a yellow halo on a lower leaflet."
- **Severity vocabulary:** few / enlarging / coalesced / blighting. · **Color vocabulary:** brown, dark-brown, yellow (halo). · **Shape vocabulary:** circular, concentric, target-like, ringed. · **Texture vocabulary:** dry, papery, velvety-centered.

**Management**
- **Practices:** Crop rotation; remove/destroy lower infected leaves and residue; staking/spacing for airflow; avoid stress.
- **Treatment:** Protectant + systemic fungicides (chlorothalonil, mancozeb, azoxystrobin/other QoIs); TOM-CAST forecasting to time sprays.
- **Prevention:** Resistant/tolerant cultivars; balanced fertility; mulching to reduce soil splash.

**References:** `UCIPM-Tomato`, `APS-CompTomato`, `Agrios2005`.

---

### 1.3 Late Blight (tomato)

- **Common name:** Late blight
- **Causal agent / scientific name:** *Phytophthora infestans*
- **Pathogen type:** **Oomycete (water mold)** — not a true fungus · **Pathogen family:** Peronosporaceae
- **Host plant:** Tomato and potato
- **Typical environmental conditions:** Cool-to-moderate (optimum ~18 °C, range ~10–24 °C), very high humidity (> 90 %), leaf wetness, fog/rain; **explosively** epidemic under cool wet conditions.
- **Disease progression:** Rapid — water-soaked leaf lesions → large greasy blotches → white underside sporulation → whole-leaf and whole-plant collapse within days; petioles/stems affected.

**Symptoms (leaf-observable)**
- **Primary symptoms:** Large, **irregular, water-soaked/greasy grayish-green to brown-black blotches**, often starting at leaf tips/margins; **white downy sporulation** on the underside at lesion margins under humidity.
- **Secondary symptoms:** Rapid browning/collapse of leaflets; brown petiole lesions; leaves droop and die.
- **Leaf color changes:** Grayish-green → brown/black; narrow pale-green/yellow border.
- **Lesion morphology:** Water-soaked, oily; **no** concentric rings; white fuzzy sporulation ring on lower surface.
- **Lesion shape:** Large, irregular, diffuse blotches. · **Lesion size:** Large, rapidly expanding (cm-scale). · **Lesion distribution:** Often begins at margins/tips; spreads fast across the leaflet.
- **Leaf margin changes:** Marginal water-soaking common at onset.
- **Leaf curling:** Affected areas collapse/wilt; leaflets droop.
- **Necrosis:** Extensive brown-black necrosis. · **Chlorosis:** Narrow pale yellow-green border only.
- **Texture changes:** Greasy/water-soaked → dry/papery; **white downy growth** on underside is diagnostic.

**Severity staging**
- **Early:** Small water-soaked pale-green spots, often at margins/tips.
- **Moderate:** Enlarging greasy grayish-brown blotches; white underside sporulation.
- **Severe:** Large coalesced dark blotches, leaf collapse, defoliation.
- **Severity indicators:** Blotch size/number, presence and extent of underside sporulation, speed of collapse.

**Differential diagnosis**
- **Confused with:** Early blight (concentric rings), bacterial spot, gray mold (*Botrytis*). Late blight's **greasy water-soaked blotches + white downy sporulation + rapid spread** distinguish it.
- **Key differentiating features:** Water-soaked grayish-green blotches, **no** rings, **white downy sporulation on underside**.
- **Diagnostic visual features:** Oily blotches + white downy underside sporulation + pale-green margin.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Fruit rot (firm greasy brown fruit lesions), concentric rings, pycnidia, mosaic, webbing, powder.
- **Recommended controlled vocabulary:** water-soaked, greasy, oily, grayish-green, blotch, downy sporulation, irregular, necrotic.
- **Recommended synonyms:** *Phytophthora* leaf blight.
- **Recommended adjectives:** water-soaked, greasy, oily, grayish-green, brown-black, irregular, downy, diffuse.
- **Forbidden adjectives:** concentric, target-like, powdery, pycnidial, mosaic, mottled, webbed, angular.
- **Recommended caption vocabulary:** "large irregular water-soaked grayish-green blotches with a pale margin."
- **Severity vocabulary:** early / spreading / collapsing / extensive. · **Color vocabulary:** grayish-green, brown-black, white (sporulation). · **Shape vocabulary:** irregular, diffuse, blotchy. · **Texture vocabulary:** water-soaked, greasy, oily, downy.

**Management**
- **Practices:** Destroy volunteers/cull piles; remove infected plants; avoid overhead irrigation; forecasting (e.g., BLITECAST/Simcast).
- **Treatment:** Protectant (chlorothalonil, mancozeb) + oomycete-specific (mefenoxam, cymoxanil, phosphonates, mandipropamid).
- **Prevention:** Resistant cultivars; certified clean transplants/seed potatoes; airflow.

**References:** `UCIPM-Tomato`, `APS-CompTomato`, `Agrios2005`.

---

### 1.4 Leaf Mold (tomato)

- **Common name:** Leaf mold
- **Causal agent / scientific name:** *Fulvia fulva* (syn. *Passalora fulva*, *Cladosporium fulvum*) — see §0.4 for the nomenclature disagreement.
- **Pathogen type:** Fungus · **Pathogen family:** Mycosphaerellaceae
- **Host plant:** Tomato (especially greenhouse / high-tunnel)
- **Typical environmental conditions:** Persistent high relative humidity (> 85 %), moderate temperatures (22–24 °C), poor ventilation; predominantly a protected-culture disease.
- **Disease progression:** Lower/older leaves first → pale diffuse patches on the upper surface with **olive-to-brown velvety mold on the underside** → leaflets yellow, curl, wither, and drop.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Diffuse pale-green to yellow patches on the upper (adaxial) surface** with indistinct margins; corresponding **olive-green to grayish-brown velvety/felty fungal growth on the underside (abaxial)** — the key diagnostic sign.
- **Secondary symptoms:** Leaflets yellow, curl, wither, and drop; reduced vigor.
- **Leaf color changes:** Upper — pale-green/yellow patches; lower — olive-to-brown mold.
- **Lesion morphology:** Diffuse chlorotic patches (top) + velvety sporulation (bottom); **no** rings, **no** pycnidia.
- **Lesion shape:** Irregular, diffuse, indistinct margins. · **Lesion size:** Variable; patches enlarge and coalesce. · **Lesion distribution:** Lower/older leaves first; two-sided (chlorosis top, mold bottom).
- **Leaf margin changes:** Leaflets curl and dry from margins in advanced stages.
- **Leaf curling:** Upward/inward curling and wilting of affected leaflets.
- **Necrosis:** Browning and death of leaflets late. · **Chlorosis:** Diffuse yellow upper-surface patches (key sign).
- **Texture changes:** **Velvety/felty olive-to-brown coating on the underside** — key diagnostic texture.

**Severity staging**
- **Early:** Pale-green diffuse spots on the upper surface.
- **Moderate:** Yellow upper patches + olive velvety mold on the underside.
- **Severe:** Coalesced chlorosis, brown withering, curling, defoliation.
- **Severity indicators:** Extent of upper chlorosis, density of underside sporulation, curling/defoliation.

**Differential diagnosis**
- **Confused with:** Early blight (concentric rings), nutrient deficiency (no mold), powdery mildew (white, upper surface), gray mold.
- **Key differentiating features:** **Olive-green/brown velvety sporulation on the LOWER surface beneath diffuse yellow upper patches**; greenhouse setting.
- **Diagnostic visual features:** Upper diffuse yellow patches + lower olive/brown velvety mold.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Fruit symptoms (rare), pycnidia, concentric rings, white powder on the upper surface (that is powdery mildew), webbing, mosaic.
- **Recommended controlled vocabulary:** diffuse yellow patches, olive-green mold, velvety, felty, underside sporulation, chlorotic patch.
- **Recommended synonyms:** *Cladosporium* leaf mold, *Fulvia* leaf mold.
- **Recommended adjectives:** diffuse, pale, yellow, olive-green, brown, velvety, felty, fuzzy (underside).
- **Forbidden adjectives:** concentric, target-like, water-soaked, greasy, angular, pycnidial, powdery (white), mosaic.
- **Recommended caption vocabulary:** "diffuse yellow patches on the upper surface with olive-green velvety mold beneath."
- **Severity vocabulary:** early / patchy / coalescing / withering. · **Color vocabulary:** pale-green, yellow, olive-green, brown. · **Shape vocabulary:** diffuse, irregular, indistinct. · **Texture vocabulary:** velvety, felty, fuzzy.

**Management**
- **Practices:** Reduce humidity, improve ventilation/spacing, avoid leaf wetness, sanitation of crop debris.
- **Treatment:** Protectant fungicides (chlorothalonil, mancozeb) and labeled options for protected culture.
- **Prevention:** Resistant cultivars (*Cf* resistance genes); environmental control in greenhouses.

**References:** `Videira2017`, `UMN-LeafMold`, `APS-CompTomato`.

---

### 1.5 Septoria Leaf Spot (tomato)

- **Common name:** Septoria leaf spot
- **Causal agent / scientific name:** *Septoria lycopersici*
- **Pathogen type:** Fungus · **Pathogen family:** Mycosphaerellaceae
- **Host plant:** Tomato and some solanaceous weeds
- **Typical environmental conditions:** Warm (20–25 °C), wet/humid; splashing water; survives on residue and weed hosts.
- **Disease progression:** Lower/older leaves first → numerous small circular spots with dark margins and tan/gray centers → **black pycnidia** develop in centers → leaflet chlorosis → upward defoliation.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Numerous small (≈1.5–3 mm) circular spots with dark-brown margins and light-gray-to-tan centers**; **tiny black pycnidia visible as pinpoint dots in the centers** — key diagnostic.
- **Secondary symptoms:** General yellowing of heavily spotted leaflets; premature bottom-up defoliation.
- **Leaf color changes:** Yellowing of affected leaflets.
- **Lesion morphology:** Circular, dark-margined, gray/tan center with pycnidia; **no** concentric rings.
- **Lesion shape:** Small, circular, uniform. · **Lesion size:** Small, ≈1.5–3 mm (typically smaller than early blight). · **Lesion distribution:** Very numerous, densely scattered; lower/older leaves first.
- **Leaf margin changes:** Not margin-specific; whole leaflet yellows/dies when spots are dense.
- **Leaf curling:** Minimal.
- **Necrosis:** Small necrotic centers, coalescing. · **Chlorosis:** General yellowing with heavy infection.
- **Texture changes:** Dry; visible pinpoint black pycnidia in centers.

**Severity staging**
- **Early:** A few small circular dark-margined spots on lower leaves.
- **Moderate:** Many circular gray-centered spots with black pycnidia; some yellowing.
- **Severe:** Dense spotting, extensive chlorosis, defoliation.
- **Severity indicators:** Spot density, visibility of pycnidia, extent of chlorosis/defoliation.

**Differential diagnosis**
- **Confused with:** Early blight (larger, concentric rings, fewer), bacterial spot (angular, greasy, no pycnidia), target spot.
- **Key differentiating features:** **Many small circular spots + gray/tan centers + pinpoint black pycnidia**; no concentric rings.
- **Diagnostic visual features:** Small circular gray-centered spots with visible black pycnidia.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Fruit (rarely affected), concentric/target rings, white mold, webbing, mosaic.
- **Recommended controlled vocabulary:** small circular spots, dark margins, gray center, tan center, pycnidia, numerous, speckled.
- **Recommended synonyms:** *Septoria* leaf spot.
- **Recommended adjectives:** small, numerous, circular, gray-centered, dark-margined, speckled, dotted.
- **Forbidden adjectives:** concentric, target-like, water-soaked, greasy, angular, velvety, powdery, mosaic.
- **Recommended caption vocabulary:** "numerous small circular spots with dark margins, gray centers, and tiny black pycnidia."
- **Severity vocabulary:** few / numerous / dense / coalescing. · **Color vocabulary:** gray, tan, dark-brown, yellow (chlorosis). · **Shape vocabulary:** circular, small, uniform. · **Texture vocabulary:** dry, speckled (pycnidia).

**Management**
- **Practices:** Rotation; remove crop residue and weed hosts; staking/spacing; avoid overhead irrigation.
- **Treatment:** Protectant fungicides (chlorothalonil, mancozeb) ± strobilurins; TOM-CAST timing.
- **Prevention:** Sanitation; mulching to reduce splash; clean transplants.

**References:** `UCIPM-Tomato`, `APS-CompTomato`, `Agrios2005`.

---

### 1.6 Spider Mites — Two-Spotted Spider Mite (tomato)

> **This class is arthropod feeding injury, not a pathogen-caused disease.** Use pest/feeding-injury vocabulary, never "infection/lesion/pathogen."

- **Common name:** Two-spotted spider mite (spider mite damage)
- **Causal agent / scientific name:** *Tetranychus urticae* (the mite itself; damage is feeding injury)
- **Pathogen type:** **Arthropod pest** (Arachnida: Acari) — not a pathogen · **Pathogen family:** Tetranychidae
- **Host plant:** Very wide host range, including tomato
- **Typical environmental conditions:** Hot, dry, dusty; water-stressed plants; favored by high temperature (> 27 °C) and low humidity; outbreaks often follow broad-spectrum insecticide use.
- **Disease progression:** Fine stippling → chlorotic/bronzed patches → underside webbing → desiccation/bronzing → leaf drop.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Fine pale/yellow stippling (tiny chlorotic flecks)** on the upper surface from cell-content feeding; **fine silk webbing on the underside**; mites and eggs may be visible as tiny dots.
- **Secondary symptoms:** Bronzing/reddening, desiccation, curling, and leaf drop; general decline.
- **Leaf color changes:** Stippled yellow → bronzed/tan → gray-brown.
- **Lesion morphology:** **Not lesions** — diffuse stippling and bronzing; **no** defined margins, rings, or pycnidia.
- **Lesion shape:** Diffuse fine speckling (no discrete lesions). · **Lesion size:** Micro-flecks (< 1 mm) coalescing into patches. · **Lesion distribution:** Often begins near the midrib/base; both surfaces; webbing on the underside.
- **Leaf margin changes:** Curling and desiccation of margins when severe.
- **Leaf curling:** Downward curling and desiccation.
- **Necrosis:** Bronzing/desiccation rather than discrete necrotic lesions. · **Chlorosis:** Fine chlorotic stippling — hallmark.
- **Texture changes:** **Fine silk webbing on the underside** — key sign; dry/brittle when severe.

**Severity staging**
- **Early:** Faint fine stippling on the upper surface.
- **Moderate:** Dense stippling, bronzing patches, fine underside webbing, mites visible.
- **Severe:** Extensive bronzing/desiccation, heavy webbing, leaf drop.
- **Severity indicators:** Stippling density, extent of bronzing, presence/quantity of webbing and mites.

**Differential diagnosis**
- **Confused with:** Nutrient deficiency, ozone/air-pollution injury, thrips or leafhopper stippling, early chlorosis.
- **Key differentiating features:** **Fine stippling + silk webbing (underside) + tiny moving mites**; no lesions, rings, or pycnidia.
- **Diagnostic visual features:** Fine stippling and webbing (and mites, if visible).

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Lesions, margined spots, concentric rings, pycnidia, mold, mosaic, bacterial ooze — **none apply**. Describe as *feeding injury*, not *infection*.
- **Recommended controlled vocabulary:** stippling, fine flecking, bronzing, webbing, speckled, desiccation, mites.
- **Recommended synonyms:** mite injury, mite stippling, spider-mite damage.
- **Recommended adjectives:** fine, stippled, bronzed, speckled, webbed, dusty, desiccated, mottled-fine.
- **Forbidden adjectives:** concentric, target-like, water-soaked, pycnidial, angular-lesion, moldy, mosaic, blighted, infected.
- **Recommended caption vocabulary:** "fine yellow stippling and bronzing with fine webbing on the underside."
- **Severity vocabulary:** faint / stippled / bronzed / desiccated. · **Color vocabulary:** yellow, tan, bronze, gray-brown. · **Shape vocabulary:** fine, speckled, diffuse. · **Texture vocabulary:** webbed, dusty, desiccated, brittle.

**Management** *(pest management, not disease control)*
- **Practices:** Reduce dust and water stress; overhead washing; conserve natural enemies (avoid broad-spectrum insecticides that flare mites).
- **Treatment:** Selective miticides/acaricides; biological control with predatory mites (*Phytoseiulus persimilis*, *Neoseiulus* spp.).
- **Prevention:** Adequate irrigation; monitoring hot/dry edges of fields; avoid mite-flaring pesticides.

**References:** `UCIPM-Mites`, `APS-CompTomato`.

---

### 1.7 Target Spot (tomato)

- **Common name:** Target spot
- **Causal agent / scientific name:** *Corynespora cassiicola*
- **Pathogen type:** Fungus · **Pathogen family:** Corynesporascaceae (order Pleosporales)
- **Host plant:** Tomato and many other hosts (highly polyphagous)
- **Typical environmental conditions:** Warm (24–32 °C), high humidity, prolonged leaf wetness (≈16–44 h); important in Florida and humid tropics.
- **Disease progression:** Leaf spots → lesions with subtle concentric rings and pale centers → coalescence → blighting (also stem and sunken fruit lesions — not leaf-visible).

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Brown-to-dark-brown lesions with subtle concentric rings and light-brown/gray centers**, sometimes with a diffuse yellow (chlorotic) halo; often pinpoint dark flecks early.
- **Secondary symptoms:** Coalescence into large necrotic areas; blighting and defoliation.
- **Leaf color changes:** Diffuse yellow halos; general yellowing when severe.
- **Lesion morphology:** Concentric rings (finer/subtler than early blight), light-gray centers that may crack.
- **Lesion shape:** Circular to irregular. · **Lesion size:** Pinpoint early → enlarging to moderate; variable. · **Lesion distribution:** Throughout the canopy (not strictly lower-leaf-first); scattered.
- **Leaf margin changes:** Not margin-specific.
- **Leaf curling:** Minimal.
- **Necrosis:** Brown necrotic centers. · **Chlorosis:** Diffuse yellow halo.
- **Texture changes:** Dry; centers sometimes cracked; dark sporulation possible.

**Severity staging**
- **Early:** Small pinpoint brown/black flecks.
- **Moderate:** Enlarging lesions with subtle concentric rings + pale centers + yellow halo.
- **Severe:** Coalesced necrosis, blighting, defoliation.
- **Severity indicators:** Lesion size/number, ring visibility, extent of coalescence and defoliation.

**Differential diagnosis**
- **Confused with:** Early blight (coarser rings, lower-leaf-first, larger halos), bacterial spot (angular, greasy), Septoria (pycnidia, gray centers).
- **Key differentiating features:** **Subtle fine concentric rings + pale center + distribution throughout the canopy**; no pycnidia.
- **Diagnostic visual features:** Fine concentric-ring "target" lesions with pale centers.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Sunken fruit lesions, stem lesions, pycnidia, white underside sporulation, webbing, mosaic.
- **Recommended controlled vocabulary:** target-like, concentric rings, pale center, brown lesion, chlorotic halo, coalescing.
- **Recommended synonyms:** *Corynespora* leaf spot.
- **Recommended adjectives:** concentric, target-like, brown, ringed, pale-centered, subtle-ringed.
- **Forbidden adjectives:** water-soaked, greasy, pycnidial, powdery, velvety, webbed, mosaic, angular.
- **Recommended caption vocabulary:** "brown lesions with subtle concentric rings and pale centers scattered across the leaflet."
- **Severity vocabulary:** pinpoint / enlarging / coalescing / blighting. · **Color vocabulary:** brown, dark-brown, gray (center), yellow (halo). · **Shape vocabulary:** concentric, target-like, circular, irregular. · **Texture vocabulary:** dry, cracked-center.

**Management**
- **Practices:** Rotation; sanitation; airflow/spacing; residue management.
- **Treatment:** Fungicides (chlorothalonil, mancozeb, strobilurins — with resistance-management caution).
- **Prevention:** No resistant commercial cultivars available; cultural control and preventive spraying emphasized.

**References:** `UFIFAS-PP351`, `APS-CompTomato`.

---

### 1.8 Tomato Mosaic Virus (ToMV)

> **Systemic virus — expresses as mottling/distortion, not discrete lesions.**

- **Common name:** Tomato mosaic
- **Causal agent / scientific name:** *Tomato mosaic virus* (ToMV); closely related to *Tobacco mosaic virus* (TMV)
- **Pathogen type:** Virus (positive-sense single-stranded RNA) · **Pathogen family:** Virgaviridae, genus *Tobamovirus*
- **Host plant:** Tomato, tobacco, pepper, and many others
- **Typical environmental conditions:** Not weather-driven. Spread mechanically (handling, tools, contaminated seed, grafting); **no insect vector required**; extremely stable and persistent on surfaces/debris.
- **Disease progression:** Systemic — light/dark-green mosaic mottling, leaf distortion ("fern-leaf"), stunting; expression varies with strain, temperature, and light.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Mottled light- and dark-green mosaic pattern** on leaflets; **leaf distortion** — narrowing, puckering, "fern-leaf"/thread-leaf; surface blistering (rugosity).
- **Secondary symptoms:** Stunting, reduced leaflet size, mild curling, occasional yellow mottling.
- **Leaf color changes:** Mosaic — alternating light-green/dark-green; sometimes yellow mottling.
- **Lesion morphology:** **No discrete lesions/spots** — a systemic mottle; no rings, pycnidia, or mold.
- **Lesion shape:** N/A (mottling, not lesions). · **Lesion size:** N/A. · **Lesion distribution:** Systemic; whole-leaflet, most evident on new growth.
- **Leaf margin changes:** Distorted, narrowed margins.
- **Leaf curling:** Mild curling, puckering, malformation.
- **Necrosis:** Generally none on leaves (certain strains/conditions can cause streak/necrosis). · **Chlorosis:** Mottled (mosaic) chlorosis.
- **Texture changes:** Rugose (blistered), puckered surface; ferny distortion.

**Severity staging**
- **Early:** Mild light/dark-green mottling on young leaves.
- **Moderate:** Distinct mosaic mottling + leaf distortion/narrowing.
- **Severe:** Pronounced fern-leaf, stunting, puckering, malformation.
- **Severity indicators:** Contrast/extent of mosaic, degree of distortion, stunting.

**Differential diagnosis**
- **Confused with:** Other viruses (e.g., CMV), growth-regulator herbicide injury (similar distortion), nutrient issues; and **TYLCV** — but TYLCV is *cupping + marginal yellowing*, not mosaic.
- **Key differentiating features:** **Light/dark-green mosaic mottling + fern-leaf distortion, without spots/lesions**; systemic.
- **Diagnostic visual features:** Green mosaic/mottling + leaf distortion (fern-leaf).

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Spots, lesions, rings, pycnidia, mold, webbing, bacterial ooze, water-soaking; internal fruit browning (not leaf-visible).
- **Recommended controlled vocabulary:** mosaic, mottling, light-and-dark-green, fern-leaf, distortion, puckering, rugose, blistering.
- **Recommended synonyms:** tomato mosaic, mosaic virus symptoms, tobamovirus mottling.
- **Recommended adjectives:** mottled, mosaic, distorted, ferny, puckered, blistered, malformed, rugose.
- **Forbidden adjectives:** spotted, concentric, target-like, water-soaked, pycnidial, necrotic-lesion, webbed, powdery, angular.
- **Recommended caption vocabulary:** "a light-and-dark-green mosaic mottling with fern-leaf distortion of the leaflets."
- **Severity vocabulary:** mild / distinct / pronounced. · **Color vocabulary:** light-green, dark-green, yellow (mottle). · **Shape vocabulary:** ferny, narrowed, distorted, puckered. · **Texture vocabulary:** rugose, blistered, puckered.

**Management**
- **Practices:** Strict hygiene (hand and tool disinfection); rogue and destroy infected plants; avoid tobacco handling near plants.
- **Treatment:** No chemical cure (viral).
- **Prevention:** Virus-free/certified seed; resistant cultivars (*Tm-1*, *Tm-2*, *Tm-2²* genes); sanitation of surfaces and debris.

**References:** `APS-CompTomato`, `UCIPM-Tomato`, `Agrios2005`.

---

### 1.9 Tomato Yellow Leaf Curl Virus (TYLCV)

> **Whitefly-vectored virus — expresses as leaf cupping + yellowing + stunting, not lesions.**

- **Common name:** Tomato yellow leaf curl
- **Causal agent / scientific name:** *Tomato yellow leaf curl virus* (TYLCV)
- **Pathogen type:** Virus (single-stranded DNA) · **Pathogen family:** Geminiviridae, genus *Begomovirus*
- **Host plant:** Tomato (and other hosts)
- **Typical environmental conditions:** Transmitted by the whitefly *Bemisia tabaci* in a persistent, circulative manner; favored by warm conditions and high whitefly populations; **not** mechanically or seed transmitted.
- **Disease progression:** Systemic — new growth shows upward cupping/curling, marginal/interveinal yellowing, reduced leaflet size, and stunting; flower drop.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Upward curling/cupping of leaflets**, **interveinal and marginal chlorosis (yellowing)**, marked **reduction in leaflet size**, and overall stunting with a bushy appearance; most pronounced on new/apical growth.
- **Secondary symptoms:** Shortened internodes (bushy/stunted habit); leaflets thickened, brittle, crumpled.
- **Leaf color changes:** Yellowing — especially leaf margins and interveinal areas of younger leaves.
- **Lesion morphology:** **No lesions/spots** — deformation + chlorosis only.
- **Lesion shape:** N/A. · **Lesion size:** N/A (reduced overall leaflet size is the sign). · **Lesion distribution:** Systemic; most pronounced on new/apical growth.
- **Leaf margin changes:** Chlorotic (yellow) margins; upward-rolled margins.
- **Leaf curling:** **Pronounced upward cupping/curling** — hallmark.
- **Necrosis:** Typically none. · **Chlorosis:** Marginal and interveinal yellowing.
- **Texture changes:** Crumpled, thickened, brittle small leaflets.

**Severity staging**
- **Early:** Slight upward curling and marginal yellowing of the youngest leaves.
- **Moderate:** Distinct cupping, interveinal/marginal chlorosis, reduced leaflet size.
- **Severe:** Severe stunting, strong cupping, bushy habit, small chlorotic leaves.
- **Severity indicators:** Degree of cupping, extent of yellowing, leaflet-size reduction, stunting.

**Differential diagnosis**
- **Confused with:** Physiological leaf roll (no yellowing/stunting, older leaves), growth-regulator herbicide injury, nutrient deficiency, broad-mite injury, other begomoviruses; and **ToMV** — but ToMV is *mosaic mottling*, not cupping.
- **Key differentiating features:** **Upward cupping + marginal/interveinal yellowing + reduced leaflet size + stunting on new growth** (physiological roll lacks yellowing/stunting; ToMV is mosaic).
- **Diagnostic visual features:** Small, upward-cupped, yellowing leaflets on new growth.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Spots, lesions, rings, pycnidia, mold, webbing, water-soaking, mosaic mottling; flower/fruit/yield effects (not leaf-visible).
- **Recommended controlled vocabulary:** upward curling, cupping, marginal yellowing, interveinal chlorosis, reduced leaflet size, stunting.
- **Recommended synonyms:** yellow leaf curl, leaf-curl virus symptoms, begomovirus curling.
- **Recommended adjectives:** cupped, curled, upward-rolled, yellow-margined, chlorotic, small, stunted, crumpled.
- **Forbidden adjectives:** spotted, concentric, target-like, water-soaked, pycnidial, necrotic, webbed, mosaic-mottled, powdery.
- **Recommended caption vocabulary:** "small upward-cupped leaflets with yellowing margins on new growth."
- **Severity vocabulary:** slight / distinct / severe. · **Color vocabulary:** yellow, chlorotic, pale-green. · **Shape vocabulary:** cupped, curled, upward-rolled, small. · **Texture vocabulary:** crumpled, thickened, brittle.

**Management**
- **Practices:** Whitefly management (insecticides, reflective mulches, insect screens/exclusion); rogue infected plants; weed/host and sanitation management.
- **Treatment:** No chemical cure (viral); manage the vector.
- **Prevention:** Resistant/tolerant cultivars (*Ty* genes); virus-free transplants; area-wide whitefly control.

**References:** `UCIPM-Tomato`, `APS-CompTomato`, `Agrios2005`.

---

## 2. MANGO — MangoLeafBD classes (*Mangifera indica*)

> Reference keys used in this section: `MangoLeafBD`, `UFIFAS-HS1369`, `Sossah2024`, `CTAHR-PD46`, `CABI-Deporaus`, `PHA-GallMidge`, `NHB-Mango`, `Chomnunti2011`, `Ploetz2003`, `Arauz2000`, `CABI`. See §3.
>
> **Note on leaf class semantics.** MangoLeafBD labels reflect *how a diseased/pest-affected leaf appears*, sometimes differing from full-plant disease names. Where the field name differs from the common literature term (Bacterial Canker ≈ bacterial black spot; Die Back = leaf expression of a twig/branch syndrome), this is stated per class.

### 2.0 Healthy (mango)

- **Common name:** Healthy mango foliage
- **Causal agent / scientific name:** None
- **Pathogen type:** N/A · **Pathogen family:** N/A
- **Host plant:** *Mangifera indica*
- **Typical environmental conditions:** N/A
- **Disease progression:** N/A

**Symptoms (leaf-observable)**
- **Primary symptoms:** Uniform green, lanceolate to oblong-lanceolate simple leaf with an **entire (smooth) margin**, prominent midrib and pinnate venation, **leathery (coriaceous), glossy** texture. Young flush is normally **coppery-red to pale/light-green** — this is healthy, not disease.
- **Secondary symptoms:** None.
- **Leaf color changes:** Mature leaves uniform dark green; young flush reddish/pale (normal).
- **Lesion morphology / shape / size / distribution:** None.
- **Leaf margin changes:** Intact, entire, smooth; no cutting/notching/scorch.
- **Leaf curling:** None (young flush may be limp before hardening — normal).
- **Necrosis / chlorosis:** None.
- **Texture changes:** Normal glossy, leathery lamina; no coating, galls, or webbing.

**Severity staging** — Not applicable (healthy). **Severity indicators:** absence of spots, coatings, galls, cutting, or discoloration.

**Differential diagnosis**
- **Confused with:** Normal coppery young flush (may look "off-color" but is healthy); mild lighting/photographic artifacts.
- **Key differentiating features:** No spots, galls, coatings, cutting, or necrosis; uniform lamina.
- **Diagnostic visual features:** Uniform green, glossy, entire-margined leaf.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Any disease/pest term — spots, lesions, galls, coating, cutting, mold, mottling.
- **Recommended controlled vocabulary:** healthy, green, uniform, intact, glossy, entire margin, leathery.
- **Recommended synonyms:** healthy leaf, disease-free foliage, normal mango leaf.
- **Recommended adjectives:** green, dark-green, uniform, glossy, leathery, intact, unblemished.
- **Forbidden adjectives:** diseased, spotted, necrotic, galled, coated, cut, mottled, blighted, chlorotic.
- **Recommended caption vocabulary:** "a healthy mango leaf with uniform green coloration, a glossy surface, and an intact margin."
- **Severity vocabulary:** none / not applicable. · **Color vocabulary:** green, dark-green, coppery (young flush). · **Shape vocabulary:** lanceolate, oblong, entire-margined. · **Texture vocabulary:** glossy, leathery, smooth.

**Management** — Routine orchard care and scouting; no treatment required.

**References:** `Ploetz2003`, `MangoLeafBD`, `NHB-Mango`.

---

### 2.1 Anthracnose (mango)

- **Common name:** Mango anthracnose
- **Causal agent / scientific name:** *Colletotrichum gloeosporioides* species complex (also *C. asianum*, *C. siamense* and others regionally)
- **Pathogen type:** Fungus · **Pathogen family:** Glomerellaceae
- **Host plant:** Mango (broad host range)
- **Typical environmental conditions:** Warm, wet, humid; rain, dew, and splashing spread conidia; the single most important field/postharvest disease in humid tropics.
- **Disease progression:** Young flush most susceptible → small dark spots → coalescence into irregular necrotic blotches, shot-hole, distortion; also blossom blight and postharvest fruit rot (**not** leaf-visible).

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Small dark-brown to black, irregular necrotic spots**, frequently along **leaf margins and tips** and between veins; enlarge and coalesce into **large angular/irregular dark necrotic areas**; young leaves distort.
- **Secondary symptoms:** **Shot-hole** (necrotic centers drop out), leaf crinkling/distortion, premature leaf drop.
- **Leaf color changes:** Dark-brown-to-black lesions; occasional yellow (chlorotic) halo; surrounding tissue may yellow.
- **Lesion morphology:** Dark, necrotic, irregular; may crack; typically **no concentric rings**; no macroscopically visible pycnidia (fruiting acervuli are micro-scale).
- **Lesion shape:** Irregular to angular (when coalesced). · **Lesion size:** ~1–5 mm early → large coalesced blotches. · **Lesion distribution:** Margins, tips, and scattered across the lamina; young leaves.
- **Leaf margin changes:** **Marginal necrosis/blackening ("black patches along the leaf margin")** — key leaf sign (consistent with MangoLeafBD class description).
- **Leaf curling:** Young leaves distort/crinkle.
- **Necrosis:** Dark-brown-to-black necrosis. · **Chlorosis:** Occasional yellow halo.
- **Texture changes:** Dry, brittle, cracked; shot-hole.

**Severity staging**
- **Early:** Tiny dark specks/spots on young leaves.
- **Moderate:** Enlarging dark irregular spots; some marginal blackening.
- **Severe:** Large coalesced black necrotic blotches, shot-hole, distortion, defoliation.
- **Severity indicators:** Blotch size/coverage, marginal blackening, shot-hole, distortion.

**Differential diagnosis**
- **Confused with:** Bacterial canker/black spot (angular, water-soaked, **raised**, vein-limited, chlorotic-halo), *Alternaria*, algal leaf spot.
- **Key differentiating features:** **Irregular dark-brown/black necrotic blotches (especially margins/tips) + shot-hole**; lesions are dry/necrotic, **not** raised water-soaked angular (that is bacterial).
- **Diagnostic visual features:** Dark irregular necrotic marginal/tip blotches, shot-hole, distortion.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Fruit tear-stain / alligator-skin lesions, blossom blight, mummified fruit, bacterial ooze, galls, surface mold coating, cut margins.
- **Recommended controlled vocabulary:** dark spots, necrotic blotches, irregular lesions, marginal necrosis, black patches, shot-hole, coalescing.
- **Recommended synonyms:** *Colletotrichum* leaf spot, anthracnose leaf blight.
- **Recommended adjectives:** dark-brown, black, irregular, necrotic, coalescing, marginal, angular (coalesced), cracked.
- **Forbidden adjectives:** raised, water-soaked, powdery, velvety, galled, cut, mosaic, concentric, target-like.
- **Recommended caption vocabulary:** "irregular dark-brown to black necrotic blotches along the leaf margin, with shot-holing."
- **Severity vocabulary:** specks / enlarging / coalesced / extensive. · **Color vocabulary:** dark-brown, black, yellow (halo). · **Shape vocabulary:** irregular, angular, coalescing. · **Texture vocabulary:** dry, brittle, cracked.

**Management**
- **Practices:** Orchard sanitation and pruning for airflow; remove diseased litter.
- **Treatment:** Protectant/systemic fungicides (copper, mancozeb, azoxystrobin, prochloraz); postharvest hot-water and fungicide dips.
- **Prevention:** Less-susceptible cultivars; canopy management; timely flush/flowering sprays.

**References:** `Arauz2000`, `Ploetz2003`, `MangoLeafBD`, `CABI`.

---

### 2.2 Bacterial Canker (mango) — bacterial black spot

> **MangoLeafBD label "Bacterial Canker" = the disease commonly called bacterial black spot** (see §0.4).

- **Common name:** Bacterial black spot / bacterial canker of mango
- **Causal agent / scientific name:** *Xanthomonas citri* pv. *mangiferaeindicae* (syn. *X. campestris* pv. *mangiferaeindicae*)
- **Pathogen type:** Bacterium (Gram-negative) · **Pathogen family:** Xanthomonadaceae
- **Host plant:** Mango
- **Typical environmental conditions:** Warm, humid; rain and wind-driven rain; entry through wounds (wind/sand abrasion) and stomata; spread by rain splash and wind.
- **Disease progression:** Leaf lesions begin water-soaked → **raised, angular, black, vein-delimited lesions** (often with a chlorotic halo) → older lesions dry to light-brown/ash-gray; also twig cankers and star-shaped gummy fruit lesions (**not** leaf-visible).

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Raised, angular, black leaf lesions delimited by veins**, initially **water-soaked** and frequently surrounded by a **chlorotic (yellow) halo**.
- **Secondary symptoms:** Lesions coalesce; older lesions dry to **light-brown/ash-gray**; severe infection → defoliation; leaves may crack.
- **Leaf color changes:** Black lesions with yellow halos; older centers ash-gray/light-brown.
- **Lesion morphology:** **Raised**, angular, vein-limited, black, water-soaked → necrotic; may exude bacterial gum.
- **Lesion shape:** **Angular (vein-delimited)** — key sign. · **Lesion size:** ~1–5 mm, coalescing. · **Lesion distribution:** Scattered, vein-delimited, on both surfaces.
- **Leaf margin changes:** Marginal lesions possible; predominantly interveinal/angular.
- **Leaf curling:** Minimal.
- **Necrosis:** Black necrotic angular lesions. · **Chlorosis:** **Yellow halo** around lesions — key sign.
- **Texture changes:** **Raised**, sometimes with exudate/gum; older lesions cracked.

**Severity staging**
- **Early:** Tiny water-soaked angular spots with a faint halo.
- **Moderate:** Raised black angular lesions with distinct chlorotic halos.
- **Severe:** Coalesced black lesions, ash-gray dried centers, defoliation.
- **Severity indicators:** Lesion density, degree of raising/coalescence, halo prominence, defoliation.

**Differential diagnosis**
- **Confused with:** Anthracnose (irregular, dry, **not** raised/water-soaked, not strictly vein-angular), *Alternaria*, algal leaf spot.
- **Key differentiating features:** **Raised, angular (vein-limited), black, water-soaked lesions with a yellow halo** (vs. anthracnose's flat, dry, irregular necrosis).
- **Diagnostic visual features:** Raised angular black vein-limited lesions + chlorotic halo.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Star-shaped/gummy fruit lesions, twig cankers, "tear-stain," fungal mold, galls, cut margins, mosaic.
- **Recommended controlled vocabulary:** angular lesions, raised, black spots, water-soaked, chlorotic halo, vein-limited, ash-gray (old).
- **Recommended synonyms:** bacterial black spot, bacterial canker (mango).
- **Recommended adjectives:** angular, raised, black, water-soaked, halo-bordered, vein-limited.
- **Forbidden adjectives:** irregular-dry (anthracnose), powdery, velvety, galled, cut, concentric, target-like, mosaic.
- **Recommended caption vocabulary:** "raised angular black lesions with yellow halos, delimited by the leaf veins."
- **Severity vocabulary:** few / raised / coalescing / defoliating. · **Color vocabulary:** black, yellow (halo), ash-gray, light-brown (old). · **Shape vocabulary:** angular, vein-limited, raised. · **Texture vocabulary:** raised, water-soaked, gummy, cracked.

**Management**
- **Practices:** Windbreaks (reduce wounding); pruning and sanitation; pathogen-free planting material.
- **Treatment:** Copper-based bactericides (preventive); limited curative options.
- **Prevention:** Less-susceptible cultivars; avoid overhead wetting; clean nursery stock.

**References:** `UFIFAS-HS1369`, `Sossah2024`, `MangoLeafBD`, `CABI`.

---

### 2.3 Cutting Weevil (mango)

> **This class is insect (beetle) leaf-cutting damage, not a pathogen-caused disease.** Use cutting/feeding-damage vocabulary, never "lesion/spot/infection."

- **Common name:** Mango leaf-cutting weevil
- **Causal agent / scientific name:** *Deporaus marginatus* (Pascoe) (the insect; damage is mechanical cutting/feeding)
- **Pathogen type:** **Insect pest** (Coleoptera: Attelabidae) — not a pathogen · **Pathogen family:** Attelabidae (leaf-rolling weevils)
- **Host plant:** Mango (especially nursery seedlings and young flush)
- **Typical environmental conditions:** Active on tender new flush; nurseries and young trees; seasonal with flushing.
- **Disease progression:** Adults make scissor-like cuts on young leaves → cut/notched/trimmed margins and "windowpane" feeding scars → cut leaf tips drop; larvae feed between leaf surfaces.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Clean, scissor-like cuts and notches at leaf margins/tips**; **"windowpane" translucent feeding scars** (epidermis left intact) on young leaves; overall **trimmed/ragged** leaf appearance.
- **Secondary symptoms:** Missing leaf sections; browning/drying at cut edges; larval mining between leaf surfaces.
- **Leaf color changes:** Green lamina with brown/dry cut edges; **no** chlorosis/mosaic.
- **Lesion morphology:** **Not lesions** — mechanical cuts, notches, and windowpane scars; edges may brown/dry.
- **Lesion shape:** Linear/clean cuts, notched margins, irregular missing tissue. · **Lesion size:** From small marginal notches to large missing sections. · **Lesion distribution:** Margins and tips of young/tender leaves.
- **Leaf margin changes:** **Cut, notched, trimmed** — key sign (MangoLeafBD: "leaves look cleanly cut").
- **Leaf curling:** Minimal; young leaves may distort.
- **Necrosis:** Browning/drying only at cut edges. · **Chlorosis:** None characteristic.
- **Texture changes:** Cut edges and translucent windowpane areas.

**Severity staging**
- **Early:** Small marginal notches / windowpane scars on flush.
- **Moderate:** Multiple clean cuts and notches; ragged leaves.
- **Severe:** Extensive tissue removed; cut/dropped tips; defoliated flush.
- **Severity indicators:** Amount of tissue removed, number of cuts/notches, windowpane area.

**Differential diagnosis**
- **Confused with:** Mechanical/physical damage, caterpillar/grasshopper chewing, hail damage.
- **Key differentiating features:** **Clean scissor-like cuts + windowpane scars on tender flush** — no spots, mold, galls, or mosaic.
- **Diagnostic visual features:** Cleanly cut/notched leaf margins; windowpane feeding.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Lesions, spots, rings, mold, ooze, galls, mosaic — **none apply**. Describe as *cutting/feeding damage*, not *infection*.
- **Recommended controlled vocabulary:** cut, notched, trimmed, windowpane, chewed margin, missing tissue, ragged.
- **Recommended synonyms:** leaf-cutting weevil damage, weevil feeding damage.
- **Recommended adjectives:** cut, notched, trimmed, ragged, clean-cut, chewed.
- **Forbidden adjectives:** spotted, necrotic-lesion, water-soaked, concentric, powdery, moldy, galled, mosaic, chlorotic, infected.
- **Recommended caption vocabulary:** "cleanly cut and notched leaf margins with translucent windowpane feeding scars."
- **Severity vocabulary:** notched / cut / ragged / defoliated. · **Color vocabulary:** green, brown (cut edges). · **Shape vocabulary:** cut, notched, ragged, clean-edged. · **Texture vocabulary:** cut, translucent (windowpane).

**Management** *(pest management, not disease control)*
- **Practices:** Collect and destroy fallen cut leaves; nursery sanitation; monitor during flush.
- **Treatment:** Insecticide sprays on new flush during adult activity.
- **Prevention:** Protect young/nursery plants during flushing; remove alternate breeding sites.

**References:** `CABI-Deporaus`, `MangoLeafBD`, `CABI`.

---

### 2.4 Die Back (mango)

> **MangoLeafBD "Die Back" = the leaf-level expression** (marginal/tip scorch, yellowing, curling, drying) **of a twig/branch dieback and tree-decline syndrome** (see §0.4). Captions describe only the leaf.

- **Common name:** Mango dieback (twig blight / tree decline)
- **Causal agent / scientific name:** *Lasiodiplodia theobromae* (syn. *Botryodiplodia theobromae*; teleomorph *Botryosphaeria rhodina*) and other Botryosphaeriaceae
- **Pathogen type:** Fungus · **Pathogen family:** Botryosphaeriaceae
- **Host plant:** Mango (broad woody-host range)
- **Typical environmental conditions:** Opportunistic on stressed trees (drought, heat, wounds, pruning cuts); warm and humid; enters via wounds.
- **Disease progression:** Twig/branch dieback from the tip downward → leaves on affected shoots **yellow, brown, curl, and dry** → defoliation; vascular discoloration in stems (**not** leaf-visible).

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Yellowing followed by browning from the leaf tip/margin inward**; leaves on affected shoots **wilt, curl, dry, and become necrotic (scorched)**; leaf drop.
- **Secondary symptoms:** Whole-shoot decline (twig level); scorched, curled, desiccated leaves.
- **Leaf color changes:** Yellowing → brown/tan necrosis; scorched appearance.
- **Lesion morphology:** **Diffuse marginal/tip necrosis (scorch)** progressing inward, rather than discrete spotted lesions.
- **Lesion shape:** Marginal/tip necrotic zones, wedge/blotch. · **Lesion size:** Large diffuse areas up to whole-leaf. · **Lesion distribution:** Tips and margins inward; leaves on affected twigs.
- **Leaf margin changes:** **Browning/scorching from the margins and tip** — key leaf sign.
- **Leaf curling:** **Curling and drying** of affected leaves.
- **Necrosis:** Extensive marginal-to-interveinal necrosis (scorch). · **Chlorosis:** Yellowing preceding necrosis.
- **Texture changes:** Dry, brittle, curled, scorched.

**Severity staging**
- **Early:** Yellowing/browning of leaf tips/margins on a shoot.
- **Moderate:** Extensive marginal necrosis, curling, drying.
- **Severe:** Fully necrotic, dried, curled leaves; defoliation (shoot dieback).
- **Severity indicators:** Extent of marginal/tip necrosis, degree of curling/drying, proportion of leaf affected.

**Differential diagnosis**
- **Confused with:** Anthracnose (discrete dark spots vs. diffuse scorch), abiotic leaf scorch (drought/salt/nutrient), bacterial canker (angular spots).
- **Key differentiating features:** **Diffuse marginal/tip scorch + curling/drying** associated with shoot dieback (vs. discrete spotting); no pycnidia typically visible on the leaf.
- **Diagnostic visual features:** Marginal/tip browning, scorch, curling, and drying (whole-leaf decline).

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Twig/branch cankers, gummosis, vascular streaking, fruit rot, bark pycnidia (not leaf-visible).
- **Recommended controlled vocabulary:** marginal necrosis, tip dieback, scorch, browning, yellowing, curling, drying, wilting.
- **Recommended synonyms:** leaf scorch (dieback), shoot-dieback leaf symptom.
- **Recommended adjectives:** scorched, browned, yellowing, curled, dried, necrotic, wilted.
- **Forbidden adjectives:** discrete-spotted, concentric, water-soaked-angular, powdery, velvety, galled, cut, mosaic.
- **Recommended caption vocabulary:** "leaf browning and scorching from the tip and margins inward, with curling and drying."
- **Severity vocabulary:** yellowing / scorching / dried. · **Color vocabulary:** yellow, brown, tan. · **Shape vocabulary:** marginal, tip-inward, wedge, diffuse. · **Texture vocabulary:** dry, brittle, curled, scorched.

**Management**
- **Practices:** Prune and destroy affected shoots well below symptoms; avoid stress and wounding; balanced irrigation/nutrition; orchard hygiene.
- **Treatment:** Protect pruning cuts (copper/fungicide paints); manage predisposing stress.
- **Prevention:** Avoid mechanical wounds; maintain tree vigor; clean tools.

**References:** `Ploetz2003`, `NHB-Mango`, `MangoLeafBD`, `CABI`.

---

### 2.5 Gall Midge (mango)

> **This class is insect (fly) larval-gall induction, not a pathogen-caused disease.** Use gall/bump vocabulary, never "flat lesion/spot."

- **Common name:** Mango leaf gall midge
- **Causal agent / scientific name:** *Procontarinia matteiana* (and other *Procontarinia* spp.) (the insect; damage is larval galling)
- **Pathogen type:** **Insect pest** (Diptera: Cecidomyiidae) — not a pathogen · **Pathogen family:** Cecidomyiidae
- **Host plant:** Mango
- **Typical environmental conditions:** Active on tender new flush; seasonal with flushing; warm conditions.
- **Disease progression:** Adult lays eggs in tender leaves → larvae feed inside → **small raised wart-like galls** form within ~a week → galls turn necrotic → premature leaf drop.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Numerous small, raised, wart-like/pimple-like galls (bumps)** on the lamina; initial tiny reddish spots; galls are hard and discrete, often with a central larval puncture.
- **Secondary symptoms:** Galls turn brown/black and necrotic; **shot-hole** as gall tissue dies; leaf distortion; defoliation.
- **Leaf color changes:** Green leaf with raised galls; galls green → brown/black; sometimes a chlorotic ring around each gall.
- **Lesion morphology:** **Raised discrete galls (swellings)** — not flat lesions; central pinhole; may necrose and drop out.
- **Lesion shape:** Circular, raised, pimple/wart-like bumps. · **Lesion size:** Small, ~1–3 mm bumps, numerous. · **Lesion distribution:** Densely scattered across the lamina, often between veins.
- **Leaf margin changes:** Not margin-specific.
- **Leaf curling:** Leaf distortion when galls are dense.
- **Necrosis:** Galls necrose (brown/black); shot-hole late. · **Chlorosis:** Possible halo around galls.
- **Texture changes:** **Raised, bumpy, wart-like** — key sign (MangoLeafBD: "pimple-like/raised wart-like galls").

**Severity staging**
- **Early:** Small reddish spots / tiny bumps on new leaves.
- **Moderate:** Numerous raised green-to-brown galls (pimples) across the lamina.
- **Severe:** Dense necrotic galls, shot-hole, distortion, defoliation.
- **Severity indicators:** Gall density, degree of necrosis/shot-hole, leaf distortion.

**Differential diagnosis**
- **Confused with:** Bacterial/fungal leaf spots (**flat**, not raised), erineum/eriophyid-mite galls, physiological bumps.
- **Key differentiating features:** **Raised discrete wart/pimple galls (3-D bumps) with a central puncture** — vs. flat lesions; no mold or ooze.
- **Diagnostic visual features:** Raised pimple/wart-like galls scattered on the lamina.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Flat necrotic lesions as the primary feature, mold, bacterial ooze, mosaic, cut margins, fruit.
- **Recommended controlled vocabulary:** galls, raised bumps, wart-like, pimple-like, nodules, swellings, scattered.
- **Recommended synonyms:** leaf-gall midge damage, midge galls.
- **Recommended adjectives:** raised, bumpy, wart-like, pimple-like, nodular, swollen, scattered.
- **Forbidden adjectives:** flat-lesion, water-soaked, concentric, powdery, velvety, cut, mosaic, angular.
- **Recommended caption vocabulary:** "numerous small raised wart-like galls scattered across the leaf lamina."
- **Severity vocabulary:** few / numerous / dense. · **Color vocabulary:** green, brown, black (necrotic galls), reddish (early). · **Shape vocabulary:** raised, wart-like, pimple-like, nodular, circular. · **Texture vocabulary:** bumpy, raised, hard.

**Management** *(pest management, not disease control)*
- **Practices:** Collect and destroy fallen galled leaves; shallow ploughing to expose pupae; monitor during flush.
- **Treatment:** Insecticide sprays timed to adult activity/egg-laying on new flush.
- **Prevention:** Protect flush; sanitation of leaf litter.

**References:** `PHA-GallMidge`, `MangoLeafBD`, `CABI`.

---

### 2.6 Powdery Mildew (mango)

- **Common name:** Mango powdery mildew
- **Causal agent / scientific name:** *Oidium mangiferae* (anamorph; commonly referenced name)
- **Pathogen type:** Fungus (obligate biotroph, ectophytic) · **Pathogen family:** Erysiphaceae
- **Host plant:** Mango
- **Typical environmental conditions:** Cool nights, dry days, high humidity/dew during flowering and flush; **a dry-weather disease** (unlike anthracnose, not driven by rain).
- **Disease progression:** **White superficial powdery growth** on young leaves, inflorescences, and young fruit → leaf distortion → purplish-brown weathering → flower/fruit drop.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **White-to-grayish superficial powdery fungal coating** (mycelium + conidia) on the leaf surface (especially young leaves and undersides); patches coalesce.
- **Secondary symptoms:** Young leaves distort/curl; grey necrotic lesions on the upper surface; **purplish-brown weathering** where the powder wears away; leaf drop.
- **Leaf color changes:** White powdery patches over the surface; underlying/weathered tissue purplish-brown; associated chlorosis.
- **Lesion morphology:** **Superficial powdery mycelial coating** — initially wipeable; later necrotic patches beneath.
- **Lesion shape:** Diffuse patches, coalescing. · **Lesion size:** Variable; coalesce to cover large areas. · **Lesion distribution:** Young leaves (both surfaces, often underside); scattered → coalescing.
- **Leaf margin changes:** Distortion/curling of young leaves.
- **Leaf curling:** Young leaves curl (often downward) and distort.
- **Necrosis:** Grey necrotic lesions (upper surface); purplish-brown weathered areas. · **Chlorosis:** Associated yellowing.
- **Texture changes:** **White powdery/mealy coating** — key diagnostic texture; superficial.

**Severity staging**
- **Early:** Small white powdery patches on young leaves.
- **Moderate:** Coalescing white powdery coating; leaf distortion.
- **Severe:** Extensive powdery coating, purplish-brown weathering, necrosis, defoliation.
- **Severity indicators:** Coating coverage, degree of distortion, extent of weathering/necrosis.

**Differential diagnosis**
- **Confused with:** **Sooty mould (BLACK, not white; on older leaves; on honeydew)**, dust/spray residue, salt deposit.
- **Key differentiating features:** **WHITE superficial powdery coating on young tissue, wipeable** (vs. black sooty mould).
- **Diagnostic visual features:** White powdery/mealy coating on the leaf surface.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Black surface coating (that is sooty mould), inflorescence/fruit symptoms, bacterial ooze, galls, cutting, concentric rings.
- **Recommended controlled vocabulary:** white powdery coating, mealy, powdery mildew, superficial growth, distortion, purplish-brown weathering.
- **Recommended synonyms:** *Oidium* powdery mildew.
- **Recommended adjectives:** white, powdery, mealy, superficial, dusty, coating, whitish-gray.
- **Forbidden adjectives:** black-sooty, water-soaked, concentric, target-like, angular, galled, cut, mosaic, greasy.
- **Recommended caption vocabulary:** "a white powdery coating on the leaf surface with young-leaf distortion."
- **Severity vocabulary:** patchy / coalescing / extensive. · **Color vocabulary:** white, grayish-white, purplish-brown (weathered). · **Shape vocabulary:** diffuse, patchy, coalescing. · **Texture vocabulary:** powdery, mealy, superficial, dusty.

**Management**
- **Practices:** Canopy airflow; monitor at flowering/flush; sanitation.
- **Treatment:** Sulfur/wettable sulfur; triazoles; strobilurins — timed to flowering and new flush.
- **Prevention:** Less-susceptible cultivars; preventive sprays at panicle emergence.

**References:** `CTAHR-PD46`, `NHB-Mango`, `Ploetz2003`, `CABI`.

---

### 2.7 Sooty Mould (mango)

> **This class is a SUPERFICIAL saprophytic fungal film on insect honeydew — the leaf tissue underneath is NOT infected.** Use surface-coating vocabulary; never assert tissue necrosis.

- **Common name:** Sooty mould (sooty mold)
- **Causal agent / scientific name:** Complex of superficial saprophytic fungi — *Capnodium* spp. (e.g., *C. mangiferae*, *C. ramosum*), *Meliola*, *Tripospermum*, *Cladosporium* — growing on **honeydew** excreted by sucking insects (hoppers, scales, mealybugs, aphids)
- **Pathogen type:** Fungus — **superficial saprophyte/epiphyte, not a tissue pathogen** · **Pathogen family:** Capnodiaceae (and others)
- **Host plant:** Mango (surface only; indirect — follows sucking-insect infestation)
- **Typical environmental conditions:** Presence of honeydew-secreting insects; humid, shaded canopies; not directly weather-driven.
- **Disease progression:** Honeydew deposits on the leaf → black superficial fungal colonization → **black sooty coating over the (usually upper) leaf surface** → shading reduces photosynthesis; the film flakes/peels when dry, revealing green tissue beneath.

**Symptoms (leaf-observable)**
- **Primary symptoms:** **Black-to-dark-brown superficial sooty/velvety coating on the leaf surface** (typically adaxial); **superficial (sits on the surface) and wipeable/peelable**, forming a film or crust.
- **Secondary symptoms:** Leaf looks blackened and dull; reduced photosynthesis; when dry, the film flakes to reveal **green, undamaged tissue beneath**.
- **Leaf color changes:** Black/dark-brown coating over green; underlying tissue remains green.
- **Lesion morphology:** **Superficial black film/crust — not tissue lesions**; **no** necrosis of underlying tissue.
- **Lesion shape:** Diffuse film/patches following honeydew distribution. · **Lesion size:** Patches to whole-leaf coverage. · **Lesion distribution:** Upper surface, along the midrib and where honeydew falls; patchy → continuous.
- **Leaf margin changes:** Not margin-specific.
- **Leaf curling:** None characteristic.
- **Necrosis:** **None** (superficial; tissue beneath is healthy). · **Chlorosis:** None directly (shading effect only).
- **Texture changes:** **Sooty, velvety, powdery-black, crusty film** — key sign; wipeable/flaky.

**Severity staging**
- **Early:** Thin black smudges/patches on the upper surface.
- **Moderate:** Continuous black sooty coating over part of the leaf.
- **Severe:** Entire leaf surface black-coated and dull.
- **Severity indicators:** Coating coverage and thickness (not tissue damage, which is absent).

**Differential diagnosis**
- **Confused with:** **Powdery mildew (WHITE, not black)**, anthracnose/necrotic spots (embedded, tissue death), dust deposit, black mildew (*Meliola* — more adherent).
- **Key differentiating features:** **BLACK superficial wipeable coating with GREEN, healthy tissue underneath**, associated with honeydew/sucking insects (vs. white powdery mildew; vs. anthracnose's embedded necrosis).
- **Diagnostic visual features:** Black sooty superficial coating (wipeable) over green tissue.

**Caption-grounding controls**
- **Symptoms that SHOULD NOT be mentioned:** Tissue lesions/necrosis (tissue is healthy beneath), rings, ooze, galls, cutting, mosaic; the underlying insects or fruit.
- **Recommended controlled vocabulary:** black sooty coating, superficial film, velvety black, soot, wipeable coating, dull black surface.
- **Recommended synonyms:** sooty mold, black surface coating.
- **Recommended adjectives:** black, sooty, superficial, velvety, dark, dusty, filmy, crusty.
- **Forbidden adjectives:** white-powdery, water-soaked, concentric, target-like, angular-lesion, galled, cut, necrotic-lesion, mosaic.
- **Recommended caption vocabulary:** "a superficial black sooty coating over the leaf surface, with the tissue green beneath."
- **Severity vocabulary:** thin / continuous / complete-coating. · **Color vocabulary:** black, dark-brown. · **Shape vocabulary:** diffuse, filmy, patchy. · **Texture vocabulary:** sooty, velvety, crusty, wipeable.

**Management** *(target the cause, not the mould)*
- **Practices:** Control honeydew-producing insects (scales, mealybugs, hoppers, aphids) — the root cause; prune for light/airflow; wash foliage.
- **Treatment:** Insecticides/horticultural oils for the insect vectors; the mould clears once honeydew stops.
- **Prevention:** Ant management (ants tend honeydew insects); canopy hygiene.

**References:** `Chomnunti2011`, `NHB-Mango`, `CTAHR-PD46`, `MangoLeafBD`, `CABI`.

---

## 3. References

Citation keys used above resolve here. Access dates: 2026-07 (extension/agency web resources). Where a claim rests on general plant-pathology consensus, the textbook/compendium sources (`Agrios2005`, `APS-CompTomato`, `Ploetz2003`) are the anchor.

### 3.1 Extension service and agency references

- **`UCIPM-Tomato`** — UC Statewide IPM Program (UC ANR), *Pest Management Guidelines: Tomato* (Bacterial spot, Early blight, Late blight, Septoria leaf spot, Powdery mildew, virus, and mite sections). https://ipm.ucanr.edu/agriculture/tomato/ (e.g., Early blight: https://ipm.ucanr.edu/agriculture/tomato/early-blight/ ; Late blight: https://ipm.ucanr.edu/agriculture/tomato/late-blight/ )
- **`UCIPM-Mites`** — UC IPM, *Twospotted spider mite* (Pest Notes / crop guidelines). https://ipm.ucanr.edu/
- **`UFIFAS-PP351`** — Pernezny K., Raid R.N., Momol M.T., et al. *Target Spot of Tomato in Florida.* UF/IFAS EDIS PP351. https://edis.ifas.ufl.edu/pp351
- **`UFIFAS-HS1369`** — *Bacterial Black Spot (BBS) of Mango in Florida.* UF/IFAS EDIS HS1369. https://ask.ifas.ufl.edu/publication/HS1369
- **`UMN-LeafMold`** — University of Minnesota Extension, *Tomato leaf mold.* https://extension.umn.edu/disease-management/tomato-leaf-mold
- **`CTAHR-PD46`** — University of Hawai'i CTAHR, *Mango Powdery Mildew* (Plant Disease PD-46). https://www.ctahr.hawaii.edu/oc/freepubs/pdf/pd-46.pdf
- **`NHB-Mango`** — National Horticulture Board (India), *Mango — diseases and their management* bulletins (e.g., powdery mildew, dieback). https://nhb.gov.in/
- **`PHA-GallMidge`** — Plant Health Australia, *Mango leaf gall midge (Procontarinia matteiana)* fact sheet. https://www.planthealthaustralia.com.au/
- **`CABI`** — CABI Compendium / CABI Digital Library datasheets for the relevant pathogens and pests (mango anthracnose, bacterial black spot, dieback, sooty mould). https://www.cabidigitallibrary.org/
- **`CABI-Deporaus`** — CABI Compendium, *Deporaus marginatus* (mango leaf-cutting weevil) datasheet. https://www.cabidigitallibrary.org/doi/abs/10.1079/cabicompendium.18427

### 3.2 APS and peer-reviewed references

- **`APS-BSGuide`** — Jones J.B., Potnis N., et al. *Diagnostic Guide for Bacterial Spot of Tomato and Pepper.* Plant Health Progress, PHP-11-21-0140-DG (APS). https://apsjournals.apsnet.org/doi/pdf/10.1094/PHP-11-21-0140-DG
- **`Constantin2016`** — Constantin E.C., Cleenwerck I., Maes M., et al. (2016). *Genetic characterization of strains named as Xanthomonas axonopodis pv. dieffenbachiae leads to a taxonomic revision of the X. axonopodis species complex* and related bacterial-spot taxonomy work; see also Timilsina et al. (2019). *Plant Pathology* / *Systematic and Applied Microbiology.*
- **`Timilsina2019`** — Timilsina S., Pereira-Martin J.A., Minsavage G.V., et al. (2019). *Multiple recombination events drive the current genetic structure of Xanthomonas perforans in Florida.* / Taxonomy of tomato–pepper bacterial-spot xanthomonads (*X. euvesicatoria*, *X. hortorum* pv. *gardneri*, *X. vesicatoria*). *Frontiers in Microbiology* / *Phytopathology.*
- **`Videira2017`** — Videira S.I.R., Groenewald J.Z., Nakashima C., et al. (2017). *Mycosphaerellaceae — Chaos or clarity?* *Studies in Mycology* 87:257–421 — adopts *Fulvia fulva* for tomato leaf mold (syn. *Passalora fulva*, *Cladosporium fulvum*).
- **`Sossah2024`** — Sossah F.L., et al. (2024). *A critical review on bacterial black spot of mango caused by Xanthomonas citri pv. mangiferaeindicae: current status and direction for future research.* *Forest Pathology* 54:e12860. https://onlinelibrary.wiley.com/doi/10.1111/efp.12860
- **`Arauz2000`** — Arauz L.F. (2000). *Mango anthracnose: economic impact and current options for integrated management.* *Plant Disease* 84(6):600–611.
- **`Chomnunti2011`** — Chomnunti P., Bhat D.J., Jones E.B.G., et al. (2011). *Trichomeriaceae/Capnodiaceae* and the biology of sooty moulds. *Fungal Diversity* — reference for sooty-mould biology (superficial, honeydew-associated).
- **`MangoLeafBD`** — Ahmed S.I., Ibrahim M., Nadim M., et al. (2023). *MangoLeafBD: A comprehensive image dataset to classify diseased and healthy mango leaves.* *Data in Brief* 47:108941. Dataset: https://data.mendeley.com/datasets/hxsnvwty3r/1 ; PubMed: https://pubmed.ncbi.nlm.nih.gov/36819904/

### 3.3 Textbook / compendium references

- **`Agrios2005`** — Agrios G.N. (2005). *Plant Pathology*, 5th ed. Elsevier Academic Press. (General pathogen biology, disease cycles, symptom terminology.)
- **`APS-CompTomato`** — Jones J.B., Zitter T.A., Momol M.T., Miller S.A. (eds.) (2014). *Compendium of Tomato Diseases and Pests*, 2nd ed. APS Press. (Authoritative symptom descriptions for all tomato classes.)
- **`Ploetz2003`** — Ploetz R.C. (ed.) (2003). *Diseases of Tropical Fruit Crops.* CABI Publishing. (Mango chapter: anthracnose, bacterial black spot, dieback, powdery mildew, sooty mould.)

---

## 4. Provenance and usage notes

- **Provenance of claims.** Pathogen identities, symptomatology, and differential-diagnosis features were compiled from the sources in §3 and cross-checked against APS/extension descriptions during compilation (2026-07). Taxonomic statements reflect the disagreements documented in §0.4. No claim in this DKB was generated by, or derived from, a VLM or image analysis.
- **How downstream stages should use this file.** Caption generation and validation must (a) draw only from each class's *Recommended controlled vocabulary / adjectives / caption vocabulary*, (b) never use the class's *Forbidden adjectives*, and (c) never assert any item under *Symptoms that SHOULD NOT be mentioned* — these encode the leaf-only observability constraint (§0.1). For the four non-pathogen classes (spider mites, cutting weevil, gall midge, sooty mould), use pest/feeding/surface vocabulary, not infection/lesion vocabulary (§0.2).
- **What this DKB does not do.** It does not assign labels to images, does not rank per-image severity, and does not describe any specific photograph. Per-image severity is a downstream annotation task; the *Severity staging* here defines the vocabulary and criteria for that task, not a per-image judgment.

