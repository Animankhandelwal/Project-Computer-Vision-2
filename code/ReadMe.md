# Color Image Reconstruction from Glass Plate Negatives

This project reconstructs a color image from a single grayscale scan that contains three vertically stacked exposures (Blue, Green, and Red). The main objective is to accurately align the Green and Red channels to the Blue channel and combine them into a single RGB image. I built this pipeline with a focus on robustness, correctness on high-resolution images, and practical usability for batch processing.

---

## 🔎 What I Built

I implemented a complete end-to-end pipeline that:
- Splits stacked glass-plate scans into B, G, and R channels
- Estimates the optimal 2D translation to align G→B and R→B
- Reconstructs a clean RGB image after alignment
- Supports both single-image runs and batch processing over folders
- Generates an HTML gallery for visual inspection of results

The implementation is designed to work well on large, high-resolution historical scans where naive alignment methods tend to fail.

---

## 🧠 Concepts & Techniques Used

### Channel Alignment as Translation Search
Each input image contains three stacked grayscale exposures. I treat alignment as a 2D translation problem, estimating `(dy, dx)` offsets that best align the Green and Red channels to the Blue channel. After computing these offsets, I apply the shifts and stack the channels to form the final RGB image.

### Robust Similarity Metrics
I primarily use **Normalized Cross-Correlation (NCC)** for alignment because it is more robust to brightness and contrast differences between channels. I also include **L2 / SSD** as a baseline option for comparison and debugging.

### Edge-Based Matching (Gradient Magnitude)
Aligning raw pixel intensities often failed on challenging images due to exposure differences and low-texture regions. To make alignment more robust, I perform matching on **gradient magnitude images**. Edges are far more stable across channels and provide a stronger structural signal for alignment.

### Overlap-Only Scoring (No Wrap-Around)
Instead of using circular shifts for scoring (which can introduce false correlations due to wrap-around), I compute similarity only on the **valid overlapping region** between shifted images. This prevents border artifacts from dominating the alignment score.

### Border Mitigation via Center Cropping
Scanned glass plates often contain strong borders, frames, and artifacts. I mitigate their influence by scoring alignment on a **center crop** of the image, focusing the metric on meaningful content rather than scan artifacts.

### Coarse-to-Fine Image Pyramid
To handle large images efficiently and robustly, I use a **coarse-to-fine pyramid**:
1. Downsample the image until it is small enough  
2. Find a coarse alignment  
3. Propagate and refine the offset at higher resolutions  

This significantly reduces runtime and helps capture large displacements reliably.

---

## ⚠️ Challenges I Faced

- **Borders dominating alignment:**  
  Strong frames and scan artifacts consistently fooled similarity metrics. I addressed this using center cropping and edge-based matching.

- **Wrap-around artifacts from naive shifting:**  
  Circular shifts created misleading correlations. Switching to overlap-only scoring removed this failure mode.

- **Exposure differences across channels:**  
  Raw intensity matching often failed when one channel was darker or brighter. Gradient-based alignment proved far more stable.

- **Large displacements in high-resolution images:**  
  Exhaustive search at full resolution was slow and brittle. The pyramid approach solved both performance and robustness issues.

These issues forced me to move beyond a naive alignment pipeline and design something more principled and reliable.

---

## 🚀 How to Run the Pipeline

### 1) Install Dependencies
```bash
pip install numpy imageio tifffile
```

### 2) Run on a Single Image
```bash
python main.py -i data/emir.tif -o outputs
```

### 3) Run on a Folder of Images
```bash
python main.py -i data/ -o outputs --make-html
```

### 4) Recommended Flags for Best Results
```bash
python main.py -i data/ -o outputs --make-html \
  --metric ncc --min-size 128 --refine-radius 6 --crop-frac 0.7
```

### CLI Arguments
- `-i / --input`: Input image path or folder  
- `-o / --outdir`: Output directory  
- `--metric`: `ncc` or `l2`  
- `--min-size`: Pyramid base resolution threshold  
- `--refine-radius`: Local refinement window per pyramid level  
- `--crop-frac`: Central crop fraction used for scoring  
- `--make-html`: Generates `outputs/index.html` to visually inspect results  
- `--no-pyramid`: Disables pyramid  

Outputs are saved as `*_aligned.jpg` in the output directory.


## 📚 References 

### Books
- Richard Szeliski, *Computer Vision: Algorithms and Applications*, Springer.  
  (Chapters on image alignment, matching, and multi-scale methods)
- Rafael C. Gonzalez and Richard E. Woods, *Digital Image Processing*, Pearson.  
  (Fundamentals of image gradients, edge detection, and image transformations)

### Research Papers
- Brown, M., & Lowe, D. G. (2007). *Automatic Panoramic Image Stitching using Invariant Features*.  
  International Journal of Computer Vision.  
- Lucas, B. D., & Kanade, T. (1981). *An Iterative Image Registration Technique with an Application to Stereo Vision*.  
  Proceedings of IJCAI.  
- Burt, P. J., & Adelson, E. H. (1983). *The Laplacian Pyramid as a Compact Image Code*.  
  IEEE Transactions on Communications.  

### Blogs & Tutorials

- Template Matching and Normalized Cross-Correlation (OpenCV Docs):  
  https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html  
 
- Edge Detection and Gradients (OpenCV Tutorial):  
  https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html  
