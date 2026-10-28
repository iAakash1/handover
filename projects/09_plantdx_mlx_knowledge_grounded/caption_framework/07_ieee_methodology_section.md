# 07 — IEEE-Ready Methodology Section (Task 9)

**Deliverable 8 of 8.** A publication-quality methodology section describing the knowledge-grounded caption generation methodology, written as a drop-in for an IEEE-format paper. Notation and claims are consistent with docs 00–06. Figure/table captions are provided for the implementer to render. Prose is written in the impersonal present tense expected of a methods section.

> Suggested placement: *Section III — Methodology*, following a *Section II — Related Work* that situates the approach against VLM-distilled and template-only captioning. Cross-reference the DKB (Stage 1) as *Section III-A: Knowledge Base Construction* (already written); the text below is *Section III-B onward*.

---

## III-B. Overview of Knowledge-Grounded Caption Generation

We generate the instruction-tuning caption corpus by a deterministic, knowledge-grounded procedure that is, by construction, independent of any vision–language or large language model. Let $\mathcal{D}=\{d_1,\dots,d_{18}\}$ be the set of disease/condition classes (ten tomato classes from PlantVillage, eight mango classes from MangoLeafBD). For an image $x$ with ground-truth class $y(x)\in\mathcal{D}$ obtained directly from its dataset folder, the caption generator $G$ produces a set of captions
$$
G\!\left(x\right) = g\!\left(y(x),\,\mathcal{K},\,\theta,\,s_x\right),
$$
where $\mathcal{K}$ is the Disease Knowledge Base, $\theta$ the (fixed) configuration, and $s_x$ a deterministic seed derived from the image identifier. Crucially, $G$ is a function of the *label* $y(x)$ and the knowledge base $\mathcal{K}$, and **not** of the image pixels $x$ nor of any model prediction. This property is the methodological cornerstone: because no learned model ever contributes to a caption, the corpus cannot inherit the errors, biases, or hallucinations of a captioning model, and the well-known circularity of model-distilled supervision—using a model that is weak at a task to synthesize the labels that will teach that task—is structurally impossible.

The generator comprises a build-time stage that compiles $\mathcal{K}$ into a caption ontology and controlled vocabularies, and a run-time stage that, per image, selects a set of licensed *concepts*, realizes them through syntactic templates, diversifies their surface form under strict lexical constraints, and validates the result through a multi-stage checker before emission. Figure 1 summarizes the pipeline.

> **Fig. 1.** *Knowledge-grounded caption generation pipeline.* The Disease Knowledge Base is compiled once into a per-disease caption ontology, controlled vocabulary, and a symptom lexicon. At run time, a ground-truth label indexes the ontology; a concept selector draws a budgeted, coverage-guided subset of licensed concepts; a template realizer and a constrained vocabulary expander produce a candidate caption; a twelve-stage validator accepts it or triggers reseeded regeneration; a de-duplicator and diversity controller admit it to the corpus. No image pixels and no neural model enter the caption path.

## III-C. Knowledge Base as the Single Source of Truth

The DKB encodes, for every class, its causal agent and taxonomy, its leaf-observable symptomatology (primary and secondary signs, lesion morphology, color, shape, size, distribution, chlorosis, necrosis, texture, and deformation), a controlled vocabulary partitioned into color, shape, texture, severity, and location axes, and—critically—explicit *negative* knowledge: the terms and symptoms that must never be used for that class (`forbidden_terms`, `forbidden_adjectives`) and the features that, although real for the disease, are not observable in a single-leaf image (`forbidden_symptoms_not_leaf_observable`, e.g., fruit and twig lesions, gummosis, whole-tree decline). The caption ontology is obtained as a deterministic projection of the DKB; no disease fact is re-authored in the caption stage. Consequently, correcting or extending the knowledge base and rebuilding the ontology is the *only* way disease content changes, which preserves a single, auditable source of truth across the entire project.

## III-D. Ontology-Driven Concept Selection under an Information Budget

For each class $d$ the ontology defines disjoint concept sets—required $R_d$, optional $O_d$, and forbidden $F_d$—together with observability flags, per-concept salience, and co-selection constraints. A caption's content is a concept set $C\subseteq R_d\cup O_d$ with $R_d\subseteq C$ and $C\cap F_d=\varnothing$. An *information level* $L$ maps to a target cardinality $c(L)$, and the selector forms
$$
C = R_d \cup \operatorname{Sample}_{s}\!\left(O_d,\; k,\; w\right),\qquad k=\operatorname{clip}\!\left(c(L)-|R_d|,\,0,\,|O_d|\right),
$$
where $w$ is a salience- and coverage-aware weighting (Section III-G) and $\operatorname{Sample}_s$ is seeded sampling without replacement subject to co-selection constraints. The information level thus governs caption length and richness while never unlocking a concept outside $R_d\cup O_d$. Classes with few licensed concepts (e.g., *healthy*, insect leaf-cutting damage) are automatically restricted to short registers.

## III-E. Controlled Vocabulary and Drift-Free Lexical Expansion

Surface realization draws exclusively from a closed vocabulary. Each concept has a finite set of realization phrases traceable to a specific DKB field, and lexical variety is produced by an expansion operator that walks a directed acyclic graph whose edges are typed, meaning-preserving operations: same-class synonym substitution and the attachment of a size, shape, color, texture, extent, or location modifier drawn *only* from that class's DKB-defined axis. Two invariants make the expansion safe. First, every modifier value is licensed by the knowledge base, so no descriptor is invented. Second, and most importantly, expansion may only realize *already-selected* concepts more richly; it can never introduce a concept that the selector did not choose. Formally, if $\phi\mapsto\phi'$ is an expansion step and $\kappa(\cdot)$ denotes the asserted concept set of a phrase, then $\kappa(\phi')=\kappa(\phi)$. This closes the semantic-drift loophole that afflicts naive paraphrase, in which successively "improving" a phrase silently substitutes or drops attributes.

## III-F. Template-Based Realization and Severity Honesty

Captions are instantiated from a library of 52 templates spanning eight stylistic registers—minimal, single-sentence, two-sentence, clinical, descriptive, educational, dense, and long—each declaring the slots it requires. Templates carry *syntax only*: they contain no disease name, symptom, or descriptor, all of which enter through slots bound to selected concepts. A template is eligible for a caption only if its required slots are a subset of the selected concepts and if it is compatible with the sign type (lesion, coating, gall, stippling, cut, deformation, mottle, or healthy) of the class, guaranteeing grammatical, on-register output. Instruction turns are drawn from a parallel bank of task-typed prompts (description, identification, sign enumeration, targeted attribute questions, host identification, and differential reasoning), each paired with a response constraint so that, for example, a color question is answered strictly from the color concept.

A distinctive element of the methodology is its explicit treatment of severity. The datasets label disease identity but not severity; asserting a severity stage for an arbitrary image would be an unsupported per-image claim. We therefore separate *extent descriptors*, which report visible density and are admissible, from *stage descriptors* (mild/moderate/severe), which denote clinical progression and are prohibited by default; the latter are unlocked only in a severity-conditioned mode that requires an explicit per-image severity annotation. Because the benchmark images are curated class exemplars, captions describe the *characteristic* leaf-observable presentation of the labeled class, prioritizing diagnostic and primary signs, while rarer secondary signs are expressed with epistemic hedges ("may develop," "often shows") rather than asserted as present. The framework thus distinguishes claims the label licenses (disease identity and its characteristic signs) from claims it cannot (exact severity or exact counts in a specific frame), and structurally suppresses the latter.

## III-G. Diversity Control

Linguistic diversity is engineered at five seeded loci—template choice, concept subset, lexical substitution, syntactic alternation, and register/length—subject to explicit anti-domination caps (no template, sentence skeleton, or caption-initial trigram may exceed a configured share of a class's captions) and a coverage-guided sampler that biases concept selection toward under-covered concepts and concept pairs. Duplicate suppression operates at two levels: exact normalized-string equality and near-duplicate detection via MinHash over token shingles. The resulting corpus is measured against explicit acceptance gates—distinct-$n$, self-BLEU, template entropy, and concept/concept-pair coverage—so that diversity is a verified property of the released corpus rather than an aspiration.

## III-H. Automatic Validation

Every candidate caption passes a twelve-stage validator battery before admission: ontology conformance; forbidden-symptom and observability checking against a symptom lexicon; forbidden-vocabulary detection; closed-vocabulary (whitelist) enforcement; required-content presence; expansion (no-drift) integrity; pest-versus-pathogen and visual-register consistency; cross-disease hallmark-term leakage; a severity-stage guard; internal-contradiction and mutual-exclusion checks; grammar and fluency; and duplication. A blocking failure triggers reseeded regeneration up to a bounded number of attempts, after which a guaranteed-valid minimal fallback is emitted; a residual failure of the fallback raises a hard error, correctly attributing the fault to the knowledge base or its derivation rather than silently degrading the corpus. Because validators are image-blind, they enforce consistency with the *label's licensed description*; they cannot, by design, adjudicate the pixel content of a specific image.

## III-I. Human-in-the-Loop Quality Assurance

A stratified human review complements the automated battery. Whereas generation and validation are deliberately image-blind—the property that guarantees the absence of model leakage—human reviewers are image-aware, and their unique contribution is to detect the one error class the image-blind pipeline cannot: a caption that is knowledge-base-legal for the class yet over-claims a characteristic feature that is not actually visible in the specific frame. A pilot phase reviews the entire bootstrap batch to calibrate the ontology, templates, and lexicons and to establish inter-annotator agreement (Cohen's $\kappa\ge 0.80$); an acceptance audit then reviews at least one hundred captions per class, oversampling the hardest confusable pairs and the differential and hedged registers. Defects are triaged as critical (foreign or non-observable symptoms, forbidden vocabulary, register or severity violations), which must not occur and, if found, are treated as validator faults that block release; major (visual over-claim), bounded to at most one percent per class; and minor (fluency, redundancy), bounded to at most five percent. A corpus version is released only when every class satisfies these bounds and all diversity gates pass.

## III-J. Reproducibility

The entire procedure is deterministic given a single global seed. Per-image and per-caption seeds are derived by hashing, and each emitted record stores complete provenance—the template, the ordered selected concepts, every expansion edge and vocabulary choice, the validator trace, and the hashes pinning the knowledge base, ontology build, template set, vocabulary, and configuration. Any caption can therefore be regenerated bit-for-bit and independently audited, and any change to an upstream artifact yields a new, immutable corpus version. A single canonical, tool-agnostic record schema is maintained and converted to each trainer's format (Qwen2.5-VL, Qwen3-VL, InternVL3, Gemma-3, and the MLX toolchain) by pure adapters at training time, guaranteeing that all models are fine-tuned on identical content and identical image-level splits—a precondition for the fair zero-shot-versus-fine-tuned comparison reported in Section V.

## III-K. Why Not Generate Captions with an LLM or VLM?

The design choice to *exclude* generative models from the caption path is deliberate and, we argue, methodologically necessary for a knowledge-grounded agricultural dataset. Table I contrasts the two paradigms.

> **Table I.** *Knowledge-grounded generation versus LLM/VLM-distilled captioning.*

| Property | Knowledge-grounded (ours) | LLM/VLM-distilled |
|----------|---------------------------|-------------------|
| Factual grounding | Every claim traces to an authoritative KB field | Claims are model priors; unverifiable |
| Hallucination | Structurally prevented (closed vocabulary + validators) | Endemic; must be filtered post hoc |
| Circularity | Impossible—no model output enters supervision | A weak model's outputs become the labels that teach the task |
| Label leakage | None—generator is image-blind and uses only the ground-truth label | A VLM may "read" spurious cues or the class from context |
| Observability control | Enforced (single-leaf constraint in the KB) | Model may assert fruit/tree/field features absent from a leaf image |
| Severity honesty | Explicit gating of unsupported per-image claims | Model freely asserts severity it cannot know |
| Reproducibility | Bit-for-bit, fully seeded and provenanced | Sampling- and version-dependent; hard to reproduce |
| Controllability & auditability | Total—every term is licensed and traceable | Opaque; requires external verification |
| Cost / on-device feasibility | Cheap; deterministic; no inference on the M-series device | Repeated large-model inference; energy and time cost |
| Domain correctness for rare crops | High—expert-curated KB | Degrades where the base model's domain coverage is thin |

Our own zero-shot benchmark motivates the choice empirically: general-purpose open-weight VLMs are unreliable at crop-disease description, so using their outputs as training targets would encode their errors into the fine-tuned models. Prior work on image-caption datasets has repeatedly documented that model-distilled captions propagate factual errors and object hallucinations, and that agricultural and other specialized domains are precisely where such models are least reliable. By replacing generation-by-model with generation-by-knowledge, we obtain supervision whose correctness is a property of a curated, cited knowledge base rather than of a model's latent priors, and whose every token is auditable against that base. The trade-off—reduced open-ended lexical spontaneity—is addressed by the controlled diversity mechanisms of Section III-G, which we show meet explicit diversity targets while preserving the factual guarantees that a model-distilled corpus cannot provide.

---

### Notes for the paper authors (not for the printed section)
- Cite the DKB sources (APS, UC IPM, UF/IFAS, CABI, Videira 2017, etc.) from `knowledge_base/dkb.json:reference_registry` where symptomatology is asserted.
- Replace the qualitative claims in Table I's last two rows with the measured diversity metrics (doc 00 §7.7) and the QA defect rates (doc 05 §6) once the corpus is generated.
- If a reviewer asks for a baseline, the natural ablation is: (i) template-only captions without the ontology/validator (shows more hallucination/less control), and (ii) a VLM-distilled caption set on the same images (shows factual error rate), both compared to ours on the diagnostic split.
- The equations use standard notation; renumber to the paper's scheme. Figure 1 can be rendered from the ASCII pipeline in `00_methodology_overview.md §2`.
