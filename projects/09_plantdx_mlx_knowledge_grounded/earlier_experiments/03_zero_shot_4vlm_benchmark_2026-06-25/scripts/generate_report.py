"""
scripts/generate_report.py

Builds the final research report from everything produced by the earlier
pipeline stages:

  reports/final_report.md
  reports/final_report.pdf

Both formats are built from the same computed data (model_summary.json,
evaluation_master.json, ground_truth.json) so they never disagree. All
narrative text in the "Discussion" section is generated directly from
the computed metrics (argmax/argmin over real numbers) rather than
written as a fixed claim — if you re-run the benchmark with different
images or models, the discussion text updates accordingly.

If PDF generation fails for any reason (e.g. a missing system font),
the Markdown report is still written, and a pandoc fallback command is
printed.
"""

import sys
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import utils


# ── Shared data gathering ───────────────────────────────────────────────
def gather_data():
    summary = utils.load_json(config.JSON_RESULTS_DIR / "model_summary.json")
    master  = utils.load_json(config.JSON_RESULTS_DIR / "evaluation_master.json")
    ground_truth = utils.load_json(config.JSON_RESULTS_DIR / "ground_truth.json")
    return summary, master, ground_truth


def qualitative_examples(master, n=3):
    scored = [r for r in master if r["judge"]["total_score"] is not None]
    best = sorted(scored, key=lambda r: r["judge"]["total_score"], reverse=True)[:n]
    worst = sorted(scored, key=lambda r: r["judge"]["total_score"])[:n]
    halluc_scored = [r for r in master if r["judge"]["hallucination_score"] is not None]
    worst_halluc = sorted(halluc_scored, key=lambda r: r["judge"]["hallucination_score"])[:n]
    return best, worst, worst_halluc


def fmt(v, nd=3, suffix=""):
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}{suffix}"
    return str(v)


def model_rankings(models, summary):
    return sorted(models, key=lambda m: (summary[m]["avg_total_judge_score"] if summary[m]["avg_total_judge_score"] is not None else -1),
                  reverse=True)


def discussion_points(models, summary):
    ranking = model_rankings(models, summary)
    if not ranking:
        return []
    top = ranking[0]
    by_acc    = max(models, key=lambda m: (summary[m]["accuracy"] if summary[m]["accuracy"] is not None else -1))
    by_speed  = min(models, key=lambda m: (summary[m]["avg_latency_sec"] if summary[m]["avg_latency_sec"] is not None else 1e9))
    by_mem    = min(models, key=lambda m: (summary[m]["avg_peak_memory_gb"] if summary[m]["avg_peak_memory_gb"] is not None else 1e9))
    by_halluc = max(models, key=lambda m: (summary[m]["avg_hallucination_score"] if summary[m]["avg_hallucination_score"] is not None else -1))
    return [
        f"Highest overall judge score: {top} ({fmt(summary[top]['avg_total_judge_score'], 1)}/100).",
        f"Highest disease-label accuracy: {by_acc} "
        f"({fmt((summary[by_acc]['accuracy'] or 0) * 100, 1)}%).",
        f"Fastest average latency: {by_speed} "
        f"({fmt(summary[by_speed]['avg_latency_sec'], 2)}s/image).",
        f"Lowest peak memory: {by_mem} ({fmt(summary[by_mem]['avg_peak_memory_gb'], 2)} GB).",
        f"Most grounded (least hallucination): {by_halluc} "
        f"(groundedness {fmt(summary[by_halluc]['avg_hallucination_score'], 1)}/20).",
    ]


FUTURE_WORK = [
    "Expand the image set and disease-class coverage beyond what filename codes can resolve "
    "(see Dataset section — some PlantVillage site codes don't encode the disease at all).",
    "Add additional open-source VLMs as MLX-community ports become available.",
    "Cross-validate LLM-judge scores against a second judge provider "
    "(VLM_JUDGE_PROVIDER=gemini vs openai) to check inter-judge agreement.",
    "Collect multi-annotator human ground truth for a subset of images to validate "
    "the filename-based ground truth extraction.",
]


# ── Markdown report ──────────────────────────────────────────────────────
def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_markdown(summary, master, ground_truth):
    models = list(summary.keys())
    n_images = len({r["image"] for r in master})
    disease_classes = sorted({v["ground_truth_disease"] for v in ground_truth.values()
                               if v["ground_truth_disease"] != "Unknown"})
    n_unknown = sum(1 for v in ground_truth.values() if v["ground_truth_disease"] == "Unknown")
    ranking = model_rankings(models, summary)
    best, worst, worst_halluc = qualitative_examples(master)

    md = []
    md.append("# Crop Disease Vision-Language Model Benchmark Report")
    md.append(f"\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    md.append("## 1. Experimental Setup\n")
    md.append(md_table(["Component", "Value"], [
        ["Hardware", "MacBook Pro, Apple M4 Pro"],
        ["Memory", "24 GB unified memory"],
        ["Inference engine", "MLX-VLM (Apple MLX)"],
        ["Python", "3.12"],
        ["Conda environment", "`vlm`"],
    ]))
    md.append("")

    md.append("## 2. Models Evaluated\n")
    md.append(md_table(["Key", "Model ID"], [[k, config.MODELS[k]] for k in models]))
    md.append("")

    md.append("## 3. Dataset\n")
    md.append(f"- Number of images: **{n_images}**")
    md.append(f"- Disease classes resolved from filenames: **{len(disease_classes)}**")
    if disease_classes:
        md.append(f"  - {', '.join(disease_classes)}")
    pct_unknown = (n_unknown / n_images * 100) if n_images else 0
    md.append(f"- Images with unresolved ground truth (\"Unknown\"): **{n_unknown}** "
               f"({pct_unknown:.1f}% — see `scripts/extract_ground_truth.py` docstring "
               f"for filename-code coverage notes)")
    md.append("")

    md.append("## 4. Benchmark Methodology\n")
    md.append("**Prompt used for every model (identical across all runs):**\n")
    md.append("```\n" + config.PROMPT + "\n```\n")
    md.append("**LLM-as-a-judge scoring rubric (100 points total):**\n")
    md.append(md_table(
        ["Component", "Points"],
        [[config.RUBRIC_LABELS[k], f"0-{v}"] for k, v in config.RUBRIC_WEIGHTS.items()]
    ))
    md.append("\n**Additional objective metrics computed directly from outputs "
               "(independent of the LLM judge):** disease-label match accuracy "
               "(including a softer \"correct in alternatives\" variant), Expected "
               "Calibration Error (ECE) over stated confidence, latency, tokens/sec, "
               "and peak memory.\n")

    md.append("## 5. Results\n")
    md.append("### 5.1 Summary table\n")
    headers = ["Model", "Acc %", "Judge /100", "Symptom /20", "Disease /40",
               "Halluc /20", "Calib /10", "Latency (s)", "Tok/s", "Peak Mem (GB)"]
    rows = []
    for m in models:
        s = summary[m]
        rows.append([
            m,
            fmt((s["accuracy"] or 0) * 100, 1) if s["accuracy"] is not None else "N/A",
            fmt(s["avg_total_judge_score"], 1),
            fmt(s["avg_symptom_quality_score"], 1),
            fmt(s["avg_disease_quality_score"], 1),
            fmt(s["avg_hallucination_score"], 1),
            fmt(s["avg_confidence_calibration_score"], 1),
            fmt(s["avg_latency_sec"], 2),
            fmt(s["avg_tokens_per_sec"], 1),
            fmt(s["avg_peak_memory_gb"], 2),
        ])
    md.append(md_table(headers, rows))
    md.append("")

    md.append("### 5.2 Plots\n")
    plot_files = [
        ("avg_score_by_model.png", "Average judge score by model"),
        ("accuracy_by_model.png", "Disease prediction accuracy by model"),
        ("hallucination_rate_by_model.png", "Hallucination rate by model"),
        ("latency_by_model.png", "Average latency by model"),
        ("peak_memory_by_model.png", "Average peak memory by model"),
        ("radar_comparison.png", "Normalized multi-metric comparison"),
    ]
    for fname, caption in plot_files:
        if (config.PLOTS_DIR / fname).exists():
            md.append(f"**{caption}**\n")
            md.append(f"![{caption}](../plots/{fname})\n")
    for m in models:
        cm_file = f"confusion_matrix_{m}.png"
        if (config.PLOTS_DIR / cm_file).exists():
            md.append(f"**Confusion matrix — {m}**\n")
            md.append(f"![Confusion matrix {m}](../plots/{cm_file})\n")

    md.append("## 6. Qualitative Examples\n")

    md.append("### 6.1 Best outputs (highest judge score)\n")
    for r in best:
        md.append(f"**{r['model']} — {r['image']}** "
                   f"(score: {fmt(r['judge']['total_score'], 1)}/100, "
                   f"ground truth: {r['ground_truth_disease']})\n")
        md.append(f"> {r['raw_output'][:600]}\n")

    md.append("### 6.2 Worst outputs (lowest judge score)\n")
    for r in worst:
        md.append(f"**{r['model']} — {r['image']}** "
                   f"(score: {fmt(r['judge']['total_score'], 1)}/100, "
                   f"ground truth: {r['ground_truth_disease']})\n")
        md.append(f"> {r['raw_output'][:600]}\n")

    md.append("### 6.3 Hallucination examples (lowest groundedness score)\n")
    for r in worst_halluc:
        md.append(f"**{r['model']} — {r['image']}** "
                   f"(hallucination score: {fmt(r['judge']['hallucination_score'], 1)}/20, "
                   f"judge note: {r['judge']['justification'] or 'N/A'})\n")
        md.append(f"> {r['raw_output'][:600]}\n")

    md.append("## 7. Discussion\n")
    md.append("*The notes below are generated directly from the computed metrics above "
               "— review Section 6 for qualitative grounding.*\n")
    for line in discussion_points(models, summary):
        md.append(f"- {line}")
    md.append("")

    md.append("## 8. Conclusion\n")
    md.append("### Model ranking (by average judge score)\n")
    md.append(md_table(
        ["Rank", "Model", "Avg Judge Score /100"],
        [[i + 1, m, fmt(summary[m]["avg_total_judge_score"], 1)] for i, m in enumerate(ranking)]
    ))
    md.append("")
    md.append("### Future work\n")
    for line in FUTURE_WORK:
        md.append(f"- {line}")
    md.append("")

    return "\n".join(md)


# ── PDF report ───────────────────────────────────────────────────────────
def build_pdf(summary, master, ground_truth):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak,
    )

    esc = xml_escape  # escape any user/model-generated text before it hits Paragraph's mini-XML parser

    styles = getSampleStyleSheet()
    code_style = styles["Code"] if "Code" in styles else styles["BodyText"]
    title_style = styles["Title"]
    h2, h3, body = styles["Heading2"], styles["Heading3"], styles["BodyText"]

    def header_table(data, col_widths):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ]))
        return t

    doc = SimpleDocTemplate(
        str(config.REPORTS_DIR / "final_report.pdf"),
        pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    story = []
    models = list(summary.keys())

    story.append(Paragraph("Crop Disease Vision-Language Model Benchmark Report", title_style))
    story.append(Paragraph(datetime.now().strftime("Generated %Y-%m-%d %H:%M"), body))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. Experimental Setup", h2))
    story.append(header_table(
        [["Component", "Value"],
         ["Hardware", "MacBook Pro, Apple M4 Pro"],
         ["Memory", "24 GB unified memory"],
         ["Inference engine", "MLX-VLM (Apple MLX)"],
         ["Python", "3.12"],
         ["Conda environment", "vlm"]],
        [180, 300],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Models Evaluated", h2))
    story.append(header_table(
        [["Key", "Model ID"]] + [[k, config.MODELS[k]] for k in models], [100, 380]
    ))
    story.append(Spacer(1, 12))

    n_images = len({r["image"] for r in master})
    disease_classes = sorted({v["ground_truth_disease"] for v in ground_truth.values()
                               if v["ground_truth_disease"] != "Unknown"})
    n_unknown = sum(1 for v in ground_truth.values() if v["ground_truth_disease"] == "Unknown")
    pct_unknown = (n_unknown / n_images * 100) if n_images else 0

    story.append(Paragraph("3. Dataset", h2))
    story.append(Paragraph(
        f"Number of images: <b>{n_images}</b><br/>"
        f"Disease classes resolved from filenames: <b>{len(disease_classes)}</b> "
        f"({esc(', '.join(disease_classes)) if disease_classes else 'none resolved'})<br/>"
        f"Images with unresolved ground truth: <b>{n_unknown}</b> ({pct_unknown:.1f}%)", body
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Benchmark Methodology", h2))
    story.append(Paragraph("Prompt used for every model:", body))
    story.append(Paragraph(esc(config.PROMPT).replace("\n", "<br/>"), code_style))
    story.append(Spacer(1, 8))
    story.append(header_table(
        [["Component", "Points"]] +
        [[config.RUBRIC_LABELS[k], f"0-{v}"] for k, v in config.RUBRIC_WEIGHTS.items()],
        [320, 160],
    ))
    story.append(PageBreak())

    story.append(Paragraph("5. Results", h2))
    story.append(Paragraph("5.1 Summary table", h3))
    headers = ["Model", "Acc %", "Judge /100", "Symptom /20", "Disease /40",
               "Halluc /20", "Calib /10", "Latency s", "Tok/s", "Peak GB"]
    data = [headers]
    for m in models:
        s = summary[m]
        data.append([
            m,
            fmt((s["accuracy"] or 0) * 100, 1) if s["accuracy"] is not None else "N/A",
            fmt(s["avg_total_judge_score"], 1),
            fmt(s["avg_symptom_quality_score"], 1),
            fmt(s["avg_disease_quality_score"], 1),
            fmt(s["avg_hallucination_score"], 1),
            fmt(s["avg_confidence_calibration_score"], 1),
            fmt(s["avg_latency_sec"], 2),
            fmt(s["avg_tokens_per_sec"], 1),
            fmt(s["avg_peak_memory_gb"], 2),
        ])
    story.append(header_table(data, None))
    story.append(Spacer(1, 14))

    story.append(Paragraph("5.2 Plots", h3))
    plot_files = [
        "avg_score_by_model.png", "accuracy_by_model.png", "hallucination_rate_by_model.png",
        "latency_by_model.png", "peak_memory_by_model.png", "radar_comparison.png",
    ] + [f"confusion_matrix_{m}.png" for m in models]
    for fname in plot_files:
        fpath = config.PLOTS_DIR / fname
        if fpath.exists():
            try:
                story.append(RLImage(str(fpath), width=440, height=330, kind="proportional"))
                story.append(Spacer(1, 10))
            except Exception:
                pass
    story.append(PageBreak())

    story.append(Paragraph("6. Qualitative Examples", h2))
    best, worst, worst_halluc = qualitative_examples(master)

    def example_block(title, records, score_key, score_label):
        story.append(Paragraph(title, h3))
        if not records:
            story.append(Paragraph("(none available — judging may not have run)", body))
            return
        for r in records:
            score_val = r["judge"].get(score_key)
            story.append(Paragraph(
                f"<b>{esc(r['model'])} — {esc(r['image'])}</b> "
                f"({score_label}: {fmt(score_val, 1)}, "
                f"ground truth: {esc(r['ground_truth_disease'])})", body))
            snippet = r["raw_output"][:500] + ("..." if len(r["raw_output"]) > 500 else "")
            story.append(Paragraph(esc(snippet).replace("\n", "<br/>"), code_style))
            story.append(Spacer(1, 8))

    example_block("6.1 Best outputs", best, "total_score", "score/100")
    example_block("6.2 Worst outputs", worst, "total_score", "score/100")
    example_block("6.3 Hallucination examples", worst_halluc, "hallucination_score", "groundedness/20")
    story.append(PageBreak())

    story.append(Paragraph("7. Discussion", h2))
    story.append(Paragraph(
        "The notes below are generated directly from the computed metrics above "
        "(not subjective claims) — review Section 6 for qualitative grounding.", body))
    for line in discussion_points(models, summary):
        story.append(Paragraph("\u2022 " + esc(line), body))
    story.append(Spacer(1, 14))

    story.append(Paragraph("8. Conclusion", h2))
    story.append(Paragraph("Model ranking (by average judge score):", h3))
    ranking = model_rankings(models, summary)
    rank_data = [["Rank", "Model", "Avg Judge Score /100"]] + [
        [i + 1, m, fmt(summary[m]["avg_total_judge_score"], 1)] for i, m in enumerate(ranking)
    ]
    story.append(header_table(rank_data, [60, 200, 200]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Future work:", h3))
    for line in FUTURE_WORK:
        story.append(Paragraph("\u2022 " + esc(line), body))

    doc.build(story)


def main():
    config.ensure_directories()
    required = ["model_summary.json", "evaluation_master.json", "ground_truth.json"]
    for fname in required:
        if not (config.JSON_RESULTS_DIR / fname).exists():
            utils.log(f"Missing {fname}. Run the earlier pipeline steps first "
                       f"(benchmark.py -> extract_ground_truth.py -> judge_outputs.py -> evaluate.py).")
            return

    summary, master, ground_truth = gather_data()
    if not summary:
        utils.log("model_summary.json is empty — nothing to report.")
        return

    md_text = build_markdown(summary, master, ground_truth)
    md_path = config.REPORTS_DIR / "final_report.md"
    md_path.write_text(md_text, encoding="utf-8")
    utils.log(f"Wrote {md_path}")

    try:
        build_pdf(summary, master, ground_truth)
        utils.log(f"Wrote {config.REPORTS_DIR / 'final_report.pdf'}")
    except Exception as e:
        utils.log(f"PDF generation failed ({type(e).__name__}: {e}).")
        utils.log(f"The Markdown report at {md_path} is still available. "
                   f"Fallback conversion with pandoc:")
        utils.log(f"  pandoc {md_path} -o {config.REPORTS_DIR / 'final_report.pdf'}")


if __name__ == "__main__":
    main()
