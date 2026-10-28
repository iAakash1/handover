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

## Paper drafts (in `docs/publication/drafts/`)

| File | Title (as written in the draft) | Work it describes |
|---|---|---|
| `PlantDx_Knowledge_Grounded_Instruction_Tuning_draft_2026-08-21.pdf` (15 pages, 21 Aug 2026) | *PlantDx: Knowledge-Grounded Instruction Tuning for Vision-Language Crop Disease Diagnosis* | `projects/09_plantdx_mlx_knowledge_grounded` (MLX, tomato and mango) |
| `VLM_Structured_Diagnosis_OOD_draft_2026-08-31_v1.docx`, `_v2.docx` (+ `_v2.md` to read on GitHub) | *A Vision-Language Framework for Structured Plant Disease Diagnosis with Out-of-Distribution Detection* | `projects/06_qwen25vl_plantvillage` (Qwen2.5-VL LoRA, 98.50%, CLIP OOD gate) |
| `ICAIN2026_paper953_draft_2026-09-20.pdf` (12 pages) | *Efficient QLoRA Fine-Tuning of Qwen2.5-VL for Multi-Class Tomato Disease Classification and Cross-Domain Evaluation*, submitted version | `projects/10_tih_tomato_qwen_qlora`; the final version is the camera-ready paper above |

The two `VLM_Structured_Diagnosis_OOD` versions differ only in where two paragraphs sit. A second copy of the ICAIN
draft (built 7 Oct 2026) has identical text and is not repeated.

## Project reports (in `docs/reports/`)

| File | What it is |
|---|---|
| `06-PoCRA-PCCOE-Tomato-Mango-Report-Version1.docx` | PoCRA detailed report (10 July 2026): every VLM experiment, zero-shot benchmark, references |
| `06-PoCRA-PCCOE-Tomato-Mango-Results-Report-21-06-26-Version-01.docx` | PoCRA results report: knowledge-grounded fine-tuning, tomato and mango |
| `Kamaaksha_Chanakya_Report_TIH_v2.0.docx` | Chanakya fellowship report, December 2025 – September 2026 |
| `PlantDx_Engineering_and_Research_History.pdf`, `final_plantdx_story.pdf` | Engineering history of the knowledge-grounded PlantDx pipeline |
| `VLM_Crop_Disease_Diagnosis.pptx`, `Plant-Disease-Captioning-with-BLIP-2-LoRA.pdf` | Presentations (April 2026) |
| `Chanakya_Proposal_Presentation_2025-10.pptx` (+ `.md`) | Fellowship proposal deck (October 2025): image-based disease detection for cotton and maize |
| `TIH_Progress_Report_2026-06.pdf` | CHANAKYA project progress report, June 2026 (submitted by Dr Santwana Gudadhe; a phone number is blacked out) |
| `Chanakya_Progress_Review_Presentation.pdf` | CHANAKYA project progress review presentation |
| `Aakash_Chanakya_Report_TIH.pdf`, `Kamaaksha_Chanakya_Report_TIH.pdf` | Each fellow's Chanakya crop research documentation report (version 1, PDF) |
| `Kamaaksha_Chanakya_Report_TIH_v2.0_template_draft.docx` | An earlier, partly filled copy of the v2.0 report template (21 Oct 2026) |
| `Final_Project_Completion_Report.docx` (+ `.md`, `_SOURCES.md`) | TIH project completion report, *Vision-Language Model Based Plant Disease Diagnosis for Tomato and Mango* (October 2026), with the notes on where each number came from; phone numbers removed |
| `Final_Project_Review_Presentation.pptx` (+ `.md`) | Project completion review deck (16 slides) |
| `Final_Review_Presentation_Script_Two_Speakers.pdf`, `..._spoken.pdf` | Two-speaker script for the completion review (written for a 19-slide version of the deck that is not on the Mac), and an edited version for spoken delivery |
| `VLM_Teaching_Session.pptx` (+ `.md`), `VLM_Session_Run_Of_Show.md` | VLM teaching session for students (August 2026) and its run of show |

## Papers we studied (March – April 2026)

1. K. Nguyen Quoc et al., *A Vision-Language Foundation Model for Leaf Disease Identification*, arXiv:2505.07019
2. M. Gupta, P. Desai, R. Greer, *Agro-Consensus: Semantic Self-Consistency in Vision-Language Models for Crop Disease Management*, arXiv:2510.21757
3. M. Z. Hossain et al., *A Lightweight and Explainable Vision-Language Framework for Crop Disease Visual Question Answering*, arXiv:2601.05143
4. Z. Zhao et al., *Image Fusion via Vision-Language Model (FILM)*, ICML 2024, arXiv:2402.02235
5. H. Liu et al., *Visual Instruction Tuning (LLaVA)*, NeurIPS 2023, arXiv:2304.08485
6. J. Li et al., *BLIP-2*, ICML 2023, arXiv:2301.12597
7. S. Bai et al., *Qwen2.5-VL Technical Report*, arXiv:2502.13923
8. P. Panchal et al., *Deep Learning-Driven Image Captioning*, PLOS ONE 21(3), 2026, doi:10.1371/journal.pone.0345012
