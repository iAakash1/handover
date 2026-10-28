# mango — normalized dataset card

- **Source dataset:** MangoLeafBD
- **License:** CC BY 4.0 (MangoLeafBD)
- **Original citation:** Ahmed, S.I. et al. (2023). MangoLeafBD: A comprehensive image dataset to classify diseased and healthy mango leaves. Data in Brief 47:108941
- **Download URL:** https://data.mendeley.com/datasets/hxsnvwty3r/1
- **Normalization timestamp:** 2026-07-12T15:02:06+00:00
- **Mode:** copy
- **Image count:** 0
- **Class count:** 0
- **Dataset checksum (SHA-256):** `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- **Source layout detected:** flat (class)
- **Processed directory:** `datasets/mango/processed`

## Class distribution

| class | images |
|-------|--------|

## Known limitations

- Extracted from the full MangoLeafBD dataset; only the mapped classes are included (other crops are ignored, never deleted).
- Raw datasets are immutable; this is a copy/link into `processed/`.
- Any `train`/`val` folders are merged into single class directories; the split of each image is preserved in `manifest.json`.
- Filenames are preserved except on a genuine collision, where the split is prefixed (0 file(s) affected).
- Duplicate files skipped: 4000; checksum verification failures: 0.
- Ignored (non-target) folders: 0.
