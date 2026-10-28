#!/usr/bin/env bash
# One-time setup for the evaluation ("analyze" stage) environment: installs the
# pinned metrics stack (nltk, pycocoevalcap, bert-score+torch+transformers,
# rouge-score, matplotlib, scikit-learn, scipy) and downloads + caches every
# resource they need (WordNet for METEOR, the BERTScore backbone model) so that
# `plantdx evaluate --stage analyze` never touches the network. Run this once per
# machine/environment; re-run after changing the pinned versions in
# pyproject.toml's [eval] extra.
#
# Deliberately separate from `make install-train` (Apple Silicon / mlx-vlm): the
# eval stack (torch, transformers, scikit-learn, matplotlib) is never installed
# into the training environment, and mlx/mlx-vlm are never installed here. See
# docs/EVALUATION.md for why the two stages run in different environments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Installing the pinned evaluation extra (pyproject.toml [eval])"
pip install -e ".[eval]"

# NLTK's default per-user data directory (~/nltk_data). A project-local cache
# dir was tried and rejected: NLTK's downloader refuses to write outside a small
# set of standard locations (~/nltk_data, the interpreter's nltk_data, etc.) —
# this is the one it will actually accept, and it's what `nltk.data.find`
# searches first, so no NLTK_DATA env var is needed afterward.
echo "==> Downloading WordNet + OMW into ~/nltk_data (one-time, needed for METEOR)"
python -c "
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
"

echo "==> Verifying official METEOR (nltk.translate.meteor_score)"
python -c "
from nltk.translate.meteor_score import single_meteor_score
# NOTE: METEOR's fragmentation penalty applies even to an exact match (it is a
# function of chunk count vs. match count), so identical short strings do NOT
# score exactly 1.0 -- that is correct METEOR behavior, not a bug. A longer
# identical sentence drives the penalty toward ~0, which is what we check here.
tokens = 'this tomato leaf shows bacterial spot with small dark lesions'.split()
s = single_meteor_score(tokens, tokens)
assert s > 0.97, f'expected close to 1.0 for a longer identical match, got {s}'
print('METEOR smoke test OK:', s)
"

echo "==> Verifying official BLEU-1..4 + CIDEr (pycocoevalcap)"
python -c "
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
# CIDEr's TF-IDF weighting needs more than one document to produce a non-zero
# score (df=N=1 makes every IDF collapse to zero) -- a >=2-document corpus is
# the correct smoke test, not a single-pair one.
gts = {'0': ['a leaf with bacterial spot'], '1': ['a leaf with early blight']}
res = {'0': ['a leaf with bacterial spot'], '1': ['a leaf with early blight']}
bleu, _ = Bleu(4).compute_score(gts, res)
assert bleu[0] > 0.9, f'BLEU-1 sanity check failed: {bleu}'
cider, _ = Cider().compute_score(gts, res)
assert cider > 0, f'CIDEr sanity check failed: {cider}'
print('BLEU/CIDEr smoke test OK:', bleu, cider)
" 2>&1 | grep -v "^ratio:\|^{'testlen'"

echo "==> Verifying official ROUGE-L (rouge-score)"
python -c "
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
s = scorer.score('a leaf', 'a leaf')['rougeL'].fmeasure
assert s == 1.0, f'expected 1.0 for identical strings, got {s}'
print('ROUGE-L smoke test OK')
"

echo "==> Caching the BERTScore backbone model (roberta-large) into the HF cache"
if ! python -c "
import bert_score
# A tiny dummy scoring call forces bert_score to download + cache its default
# model (roberta-large) via huggingface_hub; this is the ONLY network access in
# this whole setup script's BERTScore step, and it happens once, here.
_, _, F1 = bert_score.score(['a leaf.'], ['a leaf.'], lang='en', verbose=False)
assert float(F1[0]) > 0.99, f'expected ~1.0 for identical strings, got {float(F1[0])}'
print('BERTScore smoke test OK')
"; then
    cat <<'EOF'

BERTScore failed to load its backbone model. If the traceback above ends with
    ImportError: numpy.core.multiarray failed to import
your environment has a PRE-EXISTING, unrelated package conflict: an old `numba`
build (compiled against NumPy 1.x) that transformers' AutoModel loading path
touches indirectly via optional audio support (librosa -> numba). This is not
caused by PlantDx's pinned eval dependencies. Two ways to fix it:
  1. Upgrade the conflicting packages (safe, reversible):
       pip install -U "numba>=0.59" "llvmlite>=0.42"
  2. Or install the [eval] extra into a *fresh* virtualenv that never had
     numba/librosa installed:
       python3 -m venv .venv-eval && source .venv-eval/bin/activate
       ./scripts/setup_eval_env.sh
See docs/EVALUATION.md#troubleshooting for details.
EOF
    exit 1
fi

cat <<EOF

==> Evaluation environment ready.
    WordNet + OMW cached in ~/nltk_data.
    BERTScore model cached in the default Hugging Face cache (~/.cache/huggingface).
    Re-run this script after upgrading any pinned eval dependency.
EOF
