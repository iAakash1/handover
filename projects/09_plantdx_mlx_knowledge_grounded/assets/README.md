# assets/

**Authored, version-controlled inputs** consumed by the pipeline. Unlike
`artifacts/` (generated, gitignored), everything here is hand-authored or
reviewed and is change-controlled: edits bump `library_version` and require
re-validation/QA.

| Path | Contents | Populated in | Spec |
|------|----------|--------------|------|
| `templates/templates.json` | The caption templates (syntax only; 30 across the 8 doc-02 families, structured-segment schema). | M3 ✅ | doc 02 §3 |
| `templates/instructions.json` | Instruction bank (≥6 paraphrases × task type). | M4 | doc 04 §4 |
| `templates/scaffold_lexicon.json` | Fixed template glue words allowed by validator V4. | M3 | doc 03 V4 |
| `vocabulary/synonyms.json` | Synonym equivalence classes. | M2 | doc 01 §7.1 |
| `vocabulary/modifiers.json` | Color/shape/size/texture/extent/location axes. | M2 | doc 01 §7.2 |
| `vocabulary/severity_axis.json` | Extent vs stage classification. | M2 | doc 00 §5 |
| `vocabulary/location_axis.json` | Location-phrase normalization. | M2 | doc 01 §3.2 |
| `vocabulary/sign_type_map.json` | Keyword → sign_type map. | M2 | doc 01 §3.3 |
| `vocabulary/hedges.json` | Hedged connectives. | M3 | doc 02 §4 |
| `vocabulary/function_words.txt` | Stop words excluded from the closed-vocab check. | M2 | doc 03 §1 |
| `metadata/label_map.json` | Folder → `disease_id` map (the one filesystem coupling). | M4 | doc 04 §2 |
| `ontology_overrides/<disease_id>.yaml` | Optional thin ontology overrides (empty by default). | M2 | doc 01 §6 |

> These files are **empty/absent** during Milestone 1 (scaffolding). They are
> authored in the milestone noted above. No disease facts are authored here that
> are not derivable from the DKB — overrides are validated against the DKB.
