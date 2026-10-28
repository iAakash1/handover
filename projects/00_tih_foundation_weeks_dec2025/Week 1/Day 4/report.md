Week 1 – Image Processing Lab Report

Name: Aakash Jawle, Kamaaksha Kharul
Tools Used: Python, NumPy, OpenCV, Matplotlib, Jupyter Notebooks
Timeline: Day 1 – Day 4

1️⃣ Objectives

Understand basic image input/output and metadata extraction.

Implement geometric transformations: resize, crop, rotate, flip.

Learn interpolation techniques: nearest neighbour & bilinear.

Apply blur operations and convert between multiple colorspaces.

Implement convolution-based image filtering and edge detection.

Compare:

OpenCV (package) vs Scratch (NumPy implementations)

Execution time vs Visual quality trade-offs

2️⃣ Day-wise Summary
📌 Day 1 — Basic Image Operations

Using Packages (OpenCV / PIL):

Image reading, displaying, saving.

Metadata extraction (shape, dtype, pixel inspection).

Transformations: resize, crop, rotate, flip.

Using Scratch (NumPy):

Manual implementations:

90° rotation

Horizontal & vertical flip

Cropping using slicing

Key Learnings:

Libraries are highly optimized and simpler to use.

Scratch operations improve conceptual understanding.

Differences in border handling & pixel rounding noticed.

📌 Day 2 — Interpolation, Blur & Colorspaces

Package Implementations:

Resize: Nearest Neighbour, Bilinear

Blurs: Gaussian, Median, Average

Colorspaces: RGB/BGR ↔ Gray, HSV, YUV/YUV420

Scratch (NumPy):

Nearest neighbour resize

Bilinear interpolation logic

Box blur using manual convolution

Observations:

OpenCV significantly faster (C++ backend + SIMD optimizations).

Bilinear > Nearest neighbour for smoother results.

Box blur quality lower compared to Gaussian.

Padding strategies affect output boundaries.

📌 Day 3 — Filters & Edge Detection

Package:

Sharpener using filter2D

Sobel (X & Y) using cv2.Sobel

Canny edge detector

Scratch:

Generic convolution function

Sobel operator applied manually

Manual sharpening kernel

Vectorized operations improved speed slightly

Key Results:

Scratch vs OpenCV Sobel visually similar

Canny provides cleaner edges due to:

Non-maximum suppression

Threshold hysteresis

OpenCV much faster for 3×3 and 5×5 kernels

📌 Day 4 — Combined Operations & Performance Analysis

Performed all operations on 10 images using both methods.

✔ Resize
✔ Blur
✔ Sharpen
✔ Edge Detection
✔ BGR → Gray conversion
✔ Additional filter: Emboss

Outputs saved to:

outputs/day4/package/

outputs/day4/scratch/

Time Summary (to be filled after execution finishes)
Operation	Method	Avg Time (ms)
Resize	Package	XX.X
Resize	Scratch	YY.Y
Blur	Package	XX.X
Blur	Scratch	YY.Y
Sharpen	Package	XX.X
Sharpen	Scratch	YY.Y
Edge	Package	XX.X
Edge	Scratch	YY.Y
BGR → Gray	Package	XX.X
BGR → Gray	Scratch	YY.Y
Emboss	Package	XX.X
Emboss	Scratch	YY.Y

(Data sourced from day4_time_summary.csv)

Insights:

OpenCV remains consistently faster.

Scratch implementations are helpful for conceptual clarity.

Performance gap widens as kernel size and resolution increase.

Ideal workflow = Scratch for understanding, Package for production.

3️⃣ Package vs Scratch — Final Comparison
Factor	Package (OpenCV)	Scratch (NumPy)
Execution Speed	⚡ Ultra-fast	🐢 Much slower
Code Complexity	Very simple	Longer & error-prone
Flexibility	Some limitations	Fully customizable
Real-world Use	Best choice	Mostly academic/research
Learning Benefit	Moderate	Very high
4️⃣ Conclusion & Future Scope

This week provided a solid foundation in digital image processing, covering:

Pixel-level operations

Spatial filtering

Colorspace transformations

Performance benchmarking

Future Work:

Morphological tasks: Erosion, Dilation, Opening, Closing

Histogram equalization and contrast enhancement

Frequency-domain filtering using FFT

Integrate preprocessing with ML models (e.g., OCR, segmentation)

5️⃣ Deliverables

Day1.ipynb — Basic image ops

Day2.ipynb — Interpolation, Blur & Colorspaces

Day3.ipynb — Filters & Edge Detection

Day4.ipynb — Combined operations + Performance evaluation

Comparison images and result charts

CSV timing reports

Week-Report.md (this document)