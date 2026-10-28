# TIH evaluation reports (11–17 Sep 2026)

**What is here:** Per run: validation after each epoch, `test_finetuned/metrics.json` (+ predictions, confusion matrix), training summary. `tih_tomato_moreaccurate_external` is the cross-domain benchmark.

**Results:** moreaccurate: test 92.76%, cross-domain 69.32%. Late Blight dropped to 14.29% F1 on outside images.

**Finding:** The model is sensitive to new photo conditions; Late Blight vs Target Leaf Spot is the main confusion.
