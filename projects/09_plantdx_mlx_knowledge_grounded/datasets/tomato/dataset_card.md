# tomato — normalized dataset card

- **Source dataset:** PlantVillage
- **License:** PlantVillage dataset — rights retained by original authors
- **Original citation:** Hughes, D.P. & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060
- **Download URL:** https://github.com/spMohanty/PlantVillage-Dataset
- **Normalization timestamp:** 2026-07-13T13:06:14+00:00
- **Mode:** copy
- **Image count:** 18160
- **Class count:** 10
- **Dataset checksum (SHA-256):** `b9556420f784bbe3dafeaee096356ac863448f900dc39fff42ccfc9487ae747c`
- **Source layout detected:** nested (split/class)
- **Processed directory:** `datasets/tomato/processed`

## Class distribution

| class | images |
|-------|--------|
| bacterial_spot | 2127 |
| early_blight | 1000 |
| healthy | 1591 |
| late_blight | 1909 |
| leaf_mold | 952 |
| mosaic_virus | 373 |
| septoria_leaf_spot | 1771 |
| spider_mites | 1676 |
| target_spot | 1404 |
| yellow_leaf_curl_virus | 5357 |

## Known limitations

- Extracted from the full PlantVillage dataset; only the mapped classes are included (other crops are ignored, never deleted).
- Raw datasets are immutable; this is a copy/link into `processed/`.
- Any `train`/`val` folders are merged into single class directories; the split of each image is preserved in `manifest.json`.
- Filenames are preserved except on a genuine collision, where the split is prefixed (0 file(s) affected).
- Duplicate files skipped: 0; checksum verification failures: 0.
- Ignored (non-target) folders: 56.
