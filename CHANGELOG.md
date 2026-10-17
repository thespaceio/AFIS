# Changelog

## Unreleased

### Added
- **Browser demo** (`web/`, published to GitHub Pages). It runs the real AFIS engine in the browser with Pyodide on the bundled synthetic prints, offering Identify, Verify, Analyse and an adjustable threshold. There's no upload: everything stays on the device, and the runtime is self-hosted rather than loaded from a CDN.
- `scripts/build_web.py` builds the static site with a trimmed copy of Pyodide. `scripts/test_web.py` runs an end-to-end test in headless Chromium, checking results, page errors, external requests and phone layout.
- The `Web demo` workflow builds and tests the site on every change and deploys it from `main`.

## 2.1.0: first public release

### Accuracy (measured, see README)
- On FVC2002 DB1–4 "B" (real prints, held out from tuning), a single enrolled impression gives EER 15.3%. At the default threshold, FAR is 0.17% and FRR 40.2%. With 3 enrolled impressions, EER is 9.4% and FRR 21.5%.
- The 0.84% EER that PR #2 reported came from mostly synthetic and augmented data and does not reflect real-world performance.
- Matcher tuning on FVC2000+FVC2004 made no measurable difference, so the original settings and the 0.18 threshold are kept.

### Added
- **Installable desktop app.** Self-contained bundles for Linux (AppImage and tar.gz), Windows (zip) and macOS (zip, Apple silicon). None of them needs Python. GitHub Actions builds and smoke-tests every bundle.
- `install.sh` / `uninstall.sh` for Linux, including Kali: per-user install with a menu entry and launchers, plus a check for Qt system libraries. Uninstalling keeps your data unless you pass `--purge`.
- **Home page** with a quick start and a **Load demo data** button. 16 curated synthetic demo prints (4 fingers × 4 impressions) ship with the app.
- **Settings page.** It covers the match threshold (saved between sessions), whether to keep encrypted enrolment images, the data and key locations, an integrity check, and delete-all with typed confirmation.
- Database page: filter, delete a single template, delete a person (with confirmation), and an empty-state hint.
- Audit page: **Verify log integrity** and **Export CSV**.
- First-run responsible-use notice, Help menu, About dialog, Ctrl+1…8 page shortcuts and an app icon.
- New CLI commands: `analyse`, `list`, `audit [--export]`, `check`, `demo`, `info`, `delete-all`, plus `--version` and `--threshold`. Exit codes: 0 = success, 1 = no match, 2 = error.
- `afis.synthetic`: an SFinGe-style synthetic fingerprint generator for tests, demos and evaluation.
- `scripts/evaluate.py`: several folders, `--synthetic`, `--gallery K` (multi-impression enrolment), failure-to-enrol rate, FAR resolution limits, error rates at several thresholds, and raw score export.
- `scripts/tune.py`, plus a tuning workflow on FVC2000/FVC2004 that keeps FVC2002 as the held-out test set.
- An FVC2002 accuracy workflow that downloads the public databases at run time. The images are never stored in the repository.
- The Enroll page suggests enrolling 2–3 impressions per finger.
- `SECURITY.md`, `CHANGELOG.md` and `packaging/INSTALL.txt`.

### Security
- The encryption key file is created atomically with 0600 permissions, and the data folder with 0700.
- The key is checked when the database opens. A missing key file is reported instead of being silently replaced.
- The template hash now covers the person and template IDs, and templates must match their row, so swapped or relabelled templates are detected.
- The audit log is HMAC hash-chained with an encrypted chain head, so edits, deletions, insertions and truncation are detected.
- `PRAGMA secure_delete`, and `VACUUM` after a full wipe.
- Person-ID validation, image size and pixel limits, and decoding by content.
- A thread lock around the shared SQLite connection, used by the GUI worker and the main thread.

### Changed
- The GUI and CLI share one per-user data folder: XDG on Linux, Application Support on macOS, `%APPDATA%` on Windows. An existing `~/.afis` folder is still used.
- Image quality now also requires a plausible ridge spacing. Random noise, smooth blobs and checkerboards no longer pass the enrolment gate.
- Switched to `opencv-python-headless`, because the regular wheel's bundled Qt plugins conflict with PySide6. `matplotlib` is now optional and only needed for evaluation plots.
- Verify and Identify wording now says "consistent with" or "candidate" rather than asserting identity. Every result shows a note that scores are probabilistic.
- Non-matching pairs are drawn with amber lines.
- The README screenshots and pipeline figure use synthetic prints only.

### Fixed
- Pure noise images could be enrolled.
- Person IDs with Yoruba and other combining diacritics (e.g. "Adébáyọ̀") were rejected.
- On smaller screens the Verify and Identify buttons were pushed out of view.
- Unreadable or non-image files dropped on the app failed silently. They now show a clear message.
- Synthetic impressions were padded with mid-grey, which left visible rectangles.

## 2.0.0 (pull request #2, unreleased)
- Rebuilt the engine (segmentation, orientation, frequency, Gabor, skeleton, minutiae, singular points and a rotation-invariant matcher), added encrypted storage and the PySide6 GUI, and replaced pickle with JSON.
