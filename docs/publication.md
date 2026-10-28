# Papers, reports and conferences

## Accepted paper

| | |
|---|---|
| Title | **Efficient QLoRA Fine-Tuning of Qwen2.5-VL for Multi-Class Tomato Disease Classification and Cross-Domain Evaluation** |
| Conference | ICAIN 2026 – Proceedings of ICAIN 2026, Springer LNNS series (indexed in zbMATH, Scopus, DBLP) |
| Submission ID | 953 |
| Decision | Accepted on 1 October 2026 (among the top 20% of reviewed papers) |
| Registration | Registration and fees completed on 21 October 2026 |
| Authors | Aakash Trimbak Jawle, Santwana Gudadhe, Kamaaksha Rajendra Kharul (corresponding), Harish Lokawad, Rutuja Chavan |
| Work it describes | `projects/10_tih_tomato_qwen_qlora` (92.76% internal test, 69.32% cross-domain) |

**Camera-ready paper (12 pages):** [`docs/publication/ICAIN2026_paper953_camera_ready.pdf`](publication/ICAIN2026_paper953_camera_ready.pdf)

In short: 7,095 TIH tomato images → 6,545 after SHA-256 de-duplication (5,247 train / 649 val / 649 test), Qwen2.5-VL-7B
with 4-bit NF4 QLoRA (rank 64, frozen vision tower, answer-only loss), 192.7 min and 10.57 GB on one RTX 4500 Ada.
Internal test 92.76% accuracy / 93.78% macro-F1; external 176-image set 69.32% / 71.95%, where Late Blight recall
drops to 10% (84.62% internally).

## Project reports (in `docs/reports/`)

| File | What it is |
|---|---|
| `06-PoCRA-PCCOE-Tomato-Mango-Report-Version1.docx` | PoCRA detailed report (10 July 2026): every VLM experiment, zero-shot benchmark, references |
| `06-PoCRA-PCCOE-Tomato-Mango-Results-Report-21-06-26-Version-01.docx` | PoCRA results report: knowledge-grounded fine-tuning, tomato and mango |
| `Kamaaksha_Chanakya_Report_TIH_v2.0.docx` | Chanakya fellowship report, December 2025 – September 2026 |
| `PlantDx_Engineering_and_Research_History.pdf`, `final_plantdx_story.pdf` | Engineering history of the knowledge-grounded PlantDx pipeline |
| `VLM_Crop_Disease_Diagnosis.pptx`, `Plant-Disease-Captioning-with-BLIP-2-LoRA.pdf` | Presentations (April 2026) |

## Papers we studied (March – April 2026)

1. K. Nguyen Quoc et al., *A Vision-Language Foundation Model for Leaf Disease Identification*, arXiv:2505.07019
2. M. Gupta, P. Desai, R. Greer, *Agro-Consensus: Semantic Self-Consistency in Vision-Language Models for Crop Disease Management*, arXiv:2510.21757
3. M. Z. Hossain et al., *A Lightweight and Explainable Vision-Language Framework for Crop Disease Visual Question Answering*, arXiv:2601.05143
4. Z. Zhao et al., *Image Fusion via Vision-Language Model (FILM)*, ICML 2024, arXiv:2402.02235
5. H. Liu et al., *Visual Instruction Tuning (LLaVA)*, NeurIPS 2023, arXiv:2304.08485
6. J. Li et al., *BLIP-2*, ICML 2023, arXiv:2301.12597
7. S. Bai et al., *Qwen2.5-VL Technical Report*, arXiv:2502.13923
8. P. Panchal et al., *Deep Learning-Driven Image Captioning*, PLOS ONE 21(3), 2026, doi:10.1371/journal.pone.0345012
