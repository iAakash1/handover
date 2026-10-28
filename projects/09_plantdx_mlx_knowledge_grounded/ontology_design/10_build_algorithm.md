# 10. Build Algorithm (DKB → Ontology)

**Pseudocode only.** No Python, no implementation. The algorithm is a pure, deterministic function `build(DKB, Policies, Schema) → (Graph, Stats, Report)`.

## 10.1 Inputs

- `DKB` — parsed `knowledge_base/dkb.json` (18 conditions + `reference_registry` + `documented_taxonomic_disagreements`).
- `Schema` — the T-Box (`ontology_schema.json`): concept types, relation types, constraints, closed vocabularies, and a `non_structural_fields` allow-list.
- `Policies` — global policy files (owned here, not in the DKB):
  - `sign_type_map` — keyword → SignType (e.g., "coating/mold/powdery" → coating; "gall/wart" → gall).
  - `region_map` — keyword → LeafRegion / PlantPart (e.g., "margin/tip" → margin/tip; "fruit/twig" → non-leaf part).
  - `canonicalization` — surface string → canonical value id (per quality axis).
  - `severity_split` — which severity_vocabulary terms are *extent* (image-licensed) vs *stage* (not).
  - `confidence_policy` — DKB field → default confidence (R1).
  - `evidence_tier_map` — reference_registry key → EvidenceTier.

All policies are **static, human-authored, and hashed** into `policy_hash`. None is learned.

## 10.2 Determinism preconditions

- Iterate every collection in **sorted canonical order** (conditions by `id`; fields in a fixed order; list items by index then lexicographic).
- Node/edge ids are **pure functions** of stable keys (below). No counters that depend on insertion timing across conditions; discriminators are the item's sorted index within its owner.
- No wall-clock, no RNG, no set iteration without sorting.

## 10.3 Deterministic id scheme

```
crop_id(c)             = "crop:" + slug(c)
agent_id(name)         = "agent:" + slug(scientific_name)          # shared if identical
family_id(name)        = "family:" + slug(name)
condition_id(d)        = "condition:" + d.id                       # DKB id is already canonical
symptom_id(d, field, i)= "symptom:" + d.id + ":" + field + ":" + i # i = sorted index in field
value_id(axis, v)      = axis + ":" + canonicalization[axis][v]    # e.g. "color:dark_brown"  (SHARED)
region_id(r)           = "region:" + r                             # closed
part_id(p)             = "part:" + p                               # closed
evidence_id(key)       = "evidence:" + key                         # = reference_registry key
edge_id(owner, rel, k) = "e:" + owner + ":" + rel + ":" + k
```

Shared ids (agents, families, value nodes, evidence, closed vocab) are what canonicalize the graph: computing the same id twice yields the same node (idempotent upsert).

## 10.4 Algorithm

```
function build(DKB, Policies, Schema):
    G = empty graph                        # nodes/edges keyed by id (idempotent upsert)

    ## Phase 0 — instantiate closed vocabularies (crop-independent, once)
    for vocab in Schema.closed_vocabularies:
        for individual in vocab.individuals:            # Severity, SignType, LeafRegion, ...
            upsert_node(G, individual.id, vocab.for_type, individual.properties)
    for (leaf_region) in LeafRegion.individuals:
        add_edge(G, part_of, leaf_region, part:leaf, {confidence: asserted})
    for excl_pair in Policies.sign_exclusions:
        add_edge(G, mutually_exclusive_with, excl_pair.a, excl_pair.b, {confidence: asserted})  # symmetric

    ## Phase 1 — evidence nodes from the registry (shared)
    for key, ref in sorted(DKB.reference_registry):
        tier = Policies.evidence_tier_map[key]
        upsert_node(G, evidence_id(key), evidence_type_for(tier), {citation: ref.citation, url: ref.url, tier})

    ## Phase 2 — per condition (sorted by DKB id)
    for d in sorted(DKB.conditions, by=id):
        cond_type   = condition_subtype(d.is_pathogen_disease, d.agent_category)   # Disease|PestDamage|SurfaceColonization|HealthyState
        cond_id     = condition_id(d)
        upsert_node(G, cond_id, cond_type, {canonical_label: d.class_label, source: d.id})

        # 2a. crop + affects
        upsert_node(G, crop_id(d.crop), Crop, {canonical_label: d.crop})
        add_edge(G, affects, cond_id, crop_id(d.crop), {confidence: asserted, evidence: primary_refs(d)})

        # 2b. causal agent(s) — handle disputed taxonomy (R2)
        agents = resolve_agents(d)                      # 1, or >1 if taxonomy disputed
        for a in agents:
            upsert_node(G, agent_id(a.name), agent_type(d.agent_category), {scientific_name: a.name, ...})
            add_edge(G, agent_in_category, agent_id(a.name), agentcat_id(d.agent_category), {confidence: asserted})
            if a.family: 
                upsert_node(G, family_id(a.family), PathogenFamily, {})
                add_edge(G, member_of_family, agent_id(a.name), family_id(a.family), {confidence: asserted, evidence: refs(d)})
            add_edge(G, caused_by, cond_id, agent_id(a.name),
                     {confidence: asserted, disputed: (len(agents) > 1), evidence: refs_for_agent(d, a)})

        # 2c. symptoms — with confidence precedence (R1) and de-duplication by canonical label
        symptom_index = {}                              # canonical_label -> node id (merge duplicates, keep max confidence)
        for field in ORDERED_SYMPTOM_FIELDS:            # diagnostic_visual_features, key_diff…, primary…, secondary…, forbidden…
            conf = Policies.confidence_policy[field]     # asserted | typical | hedged
            for i, phrase in enumerate(sorted_stable(d[field])):
                label = canonical_symptom_label(phrase, Policies)
                sid   = symptom_index.get(label) or symptom_id(d, field, i)
                observable_hint = (field != "forbidden_symptoms_not_leaf_observable")
                upsert_symptom(G, sid, label, source_field=field, source_text=phrase)
                # sign type (exactly one)
                st = classify_sign_type(phrase, cond_type, Policies.sign_type_map)     # deterministic keyword match
                add_edge_once(G, has_sign_type, sid, sign_id(st), {confidence: asserted})
                # qualities (color/shape/size/texture/distribution/morphology)
                for axis in QUALITY_AXES:
                    for v in match_vocab(phrase, d[axis+"_vocabulary"], Policies.canonicalization[axis]):
                        upsert_node(G, value_id(axis, v), axis_type(axis), {canonical_label: v})   # SHARED
                        add_edge_once(G, has_quality_rel(axis), sid, value_id(axis, v),
                                      {confidence: conf, evidence: refs(d)})
                # anatomy: appears_on (drives observability)
                regions_or_parts = resolve_regions(phrase, field, Policies.region_map)  # leaf regions and/or non-leaf parts
                for target in regions_or_parts:
                    add_edge_once(G, appears_on, sid, anatomy_id(target), {confidence: conf, evidence: refs(d)})
                # observability = false if any target is a non-leaf part (R4 conservative)
                obs = all(is_leaf_region(t) for t in regions_or_parts)
                set_property(G, sid, observable=obs)
                add_edge_once(G, has_observability, sid, obs_id(obs), {confidence: asserted})
                # has_symptom edge (merge → keep highest confidence, union evidence) (R1)
                upsert_edge(G, has_symptom, cond_id, sid,
                            confidence=max_conf(existing, conf), primary=(field in PRIMARY_FIELDS),
                            evidence=union(existing.evidence, refs(d)))
                # enforce F7 at build time: asserted ∧ ¬observable → downgrade + record (or error per policy)
                if edge.confidence == asserted and not obs: reclassify_or_error(edge)   # see R4/F7

        # 2d. extent (image-licensed) vs severity stage (NOT image-licensed)  (severity_split)
        for term in d.severity_vocabulary:
            if Policies.severity_split.is_extent(term):
                upsert_node(G, extent_id(term), Extent, {canonical_label: term})
                add_edge(G, has_extent, cond_id, extent_id(term), {confidence: typical, image_licensed: TRUE, evidence: refs(d)})
        for stage in ["mild","moderate","severe"]:
            if d.severity[stage] nonempty:
                add_edge(G, typical_at_severity, cond_id, severity_id(stage),
                         {confidence: typical, image_licensed: FALSE, evidence: refs(d)})   # F8 always false

        # 2e. differentials (confused_with → differentiated_from, with via_symptom qualifier)
        for other in d.confused_with:
            other_id = resolve_condition_ref(other)          # may be same-crop or cross-crop
            via = best_distinguishing_symptom(d.key_differentiating_features, other)  # deterministic pick
            add_edge(G, differentiated_from, cond_id, other_id,
                     {confidence: asserted, via_symptom: via, evidence: refs(d)})

        # 2f. environment (optional)
        for env in d.environmental_conditions:
            upsert_node(G, env_id(env), EnvironmentalCondition, {canonical_label: canon(env)})
            add_edge(G, favored_by, cond_id, env_id(env), {confidence: typical, evidence: refs(d)})

    ## Phase 3 — finalize
    canonical_sort(G.nodes); canonical_sort(G.edges)          # by id, then by (type, source, target)
    Stats  = compute_statistics(G)                            # 06_statistics.md
    run_validators(G, Schema)                                 # 09_validation.md, V-ONT-1..12; abort on error
    content_hash = sha256(serialize_canonical(G.nodes ⧺ G.edges))
    Graph = attach_provenance(G, dkb_sha256, policy_hash, schema_hash, builder_version, content_hash)
    Report = build_report(field_coverage, decisions_log, warnings)
    return (Graph, Stats, Report)
```

## 10.5 Key algorithmic decisions (and why)

- **Idempotent upsert keyed by deterministic id** makes value-node sharing and evidence sharing automatic and order-independent. No dedup pass is needed; the id *is* the dedup key.
- **Confidence precedence + symptom merge (R1)** means a feature stated as both "diagnostic" and "primary" becomes one node at the higher confidence, with unioned evidence — no double counting, deterministic.
- **Observability computed once, stored, and re-checked** (Phase 2c + V-ONT-8) keeps the central scientific rule single-sourced and verifiable.
- **Severity split at build time** (2d) is where severity honesty is physically enforced: extent → image-licensed, stage → never. This is not a downstream convention; it is baked into the graph.
- **Disputed taxonomy → parallel edges (R2)** rather than a silent choice preserves the DKB's documented disagreements.
- **Fail-closed validation** (Phase 3) means a graph is emitted only if valid; there is no such thing as a persisted invalid ontology.
- **Canonical serialization + content hash** gives the reproducibility guarantee ([08_versioning.md](08_versioning.md)).

## 10.6 Complexity

Linear in DKB size: one pass over conditions × fields × items, with O(1) idempotent upserts. For `D` conditions and bounded per-condition fields, build is **O(D)** time and **O(nodes+edges)** memory — trivial at the projected scale ([06_statistics.md](06_statistics.md) §6.4).

## 10.7 Explicitly out of scope for the builder

No NLP model, no embedding, no LLM/VLM call, no image access. The only "parsing" is **deterministic keyword/vocabulary matching** against the DKB's own controlled vocabularies and the static policy maps — the same mechanism the frozen Caption Framework already relies on for its lexicon. Any phrase that cannot be classified deterministically is surfaced in the build report for a human to extend a policy map — never guessed.
