```markdown
# AFIS – Automated Fingerprint Identification System (Educational)

A complete, educational implementation of an Automated Fingerprint Identification System (AFIS) written in Python. This project demonstrates the full fingerprint recognition pipeline from image acquisition to matching and secure storage.

## Features

- **Image Acquisition** – loading and validating fingerprint images with quality checks (contrast, dimensions)
- **Full Preprocessing Pipeline** – noise reduction, contrast normalization (CLAHE), Gabor filter enhancement, orientation & frequency estimation, binarization, morphological cleaning, and skeletonization
- **Feature Extraction** – minutiae detection (ridge endings and bifurcations) using crossing number, plus global features (core/delta points, pattern classification: arch/loop/whorl)
- **Template Creation & Storage** – compact fingerprint templates with SHA‑256 hashing, stored in a SQLite database with indexing and audit logging
- **Matching** – 1:1 verification and 1:N identification using geometric alignment of minutiae with configurable spatial/angular tolerances
- **Interactive Demo** – command-line interface for enrollment, verification, identification, and system statistics

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- SciPy (optional, used for some smoothing)
- Matplotlib (for visualizations)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

```
afis_project/
├── afis_core.py               # All core classes (acquisition, preprocessing, extraction, matching, database, system)
├── main.py                    # Interactive demo CLI
├── requirements.txt
├── fingerprint_templates.db   # Created automatically
└── README.md
```

## Usage

### Interactive Demo

Run the demo with:

```bash
python main.py
```

Available commands:

- `enroll` – add a new fingerprint (provide person ID and image path)
- `verify` – check if a fingerprint matches a claimed identity
- `identify` – find the best matching person(s) from the database
- `stats` – show system statistics (templates, preprocessing times, minutiae counts)
- `delete` – remove all templates of a person
- `quit` – exit

### Example Session

```
> enroll
Enter person ID (e.g., 'john_doe'): alice
Select finger position (optional, press Enter to skip):
  1 - Right thumb
  ...
Position (1-10, or blank): 1
Enter image path: sample_fingerprints/alice_thumb.png
Processing image: sample_fingerprints/alice_thumb.png ...
✅ Enrollment successful: 56 minutiae, pattern: loop
```

```
> identify
Enter image path: unknown.png
Number of top candidates to show (default 5): 3
Identifying fingerprint...
Top candidates:
  1. alice – score: 0.678 (matches: 38)
  2. bob – score: 0.123 (matches: 7)
  3. charlie – score: 0.095 (matches: 4)
```

### Using the Code Directly

```python
from afis_core import AFISSystem

afis = AFISSystem()

# Enroll a fingerprint
template = afis.enroll("fingerprint.tif", "user123", finger_position=1)

# Verify
is_match, score, details = afis.verify("query.tif", "user123")

# Identify
results = afis.identify("unknown.tif", top_k=3)

# Statistics
stats = afis.get_statistics()
```

## How It Works (Briefly)

1. **Acquisition** – loads the image, validates contrast and dimensions, resizes to 500×500.
2. **Preprocessing**:
   - Non‑local means denoising + median filter
   - CLAHE contrast normalization
   - Orientation field estimation (gradient‑based)
   - Ridge frequency estimation
   - Gabor filter enhancement
   - Adaptive thresholding + morphological cleaning
   - Zhang‑Suen skeletonization
3. **Feature Extraction**:
   - Minutiae detected using crossing number on the skeleton
   - Spurious minutiae removed (border and close pairs)
   - Global singular points (core/delta) via Poincaré index
   - Pattern classification based on core/delta counts
4. **Template**:
   - Stores minutiae list and global features
   - Generates a hash for fast pre‑matching
   - Serializable to bytes for database storage
5. **Matching**:
   - Finds optimal alignment by trying multiple minutiae pairs
   - Counts matching minutiae within spatial and angular tolerances
   - Optionally refines alignment using all matched points
   - Returns a score between 0 and 1
6. **Database**:
   - SQLite with indexes on pattern type and minutiae count
   - Audit log records all operations (save, load, delete)

## Performance Notes

- Designed for educational purposes – not production‑ready.
- Matching is linear after indexing; for large databases, further optimization (e.g., MCC descriptors, deep learning) would be needed.
- Default parameters work well for standard FVC2002 images; adjust `config` dictionaries in each class for fine‑tuning.

## Ethical Considerations

- This project is intended **only for educational and research purposes**.
- Do **not** use for real‑world security or forensic applications without thorough validation.
- The database includes an audit log to track all operations.
- When handling real fingerprint images, ensure compliance with data protection regulations (e.g., GDPR).

## License

MIT License – free for academic and educational use.

## Acknowledgments

- Inspired by the *Handbook of Fingerprint Recognition* by Maltoni, Maio, Jain, and Prabhakar.
- FVC2002/FVC2004 datasets for benchmarking.
- OpenCV, scikit‑image, and the Python scientific ecosystem.

---

*For questions or contributions, please open an issue on the repository.*
```
