## AFIS 2.1.0: first public release

A desktop app for learning fingerprint recognition: image enhancement, minutiae extraction, 1:1 verification and 1:N identification against an encrypted local database. **No Python needed** for the downloads below.

### Download

| Platform | File | How to start |
|---|---|---|
| **Linux** (Kali, Ubuntu, Debian, Fedora, x86-64) | `AFIS-2.1.0-x86_64.AppImage` | `chmod +x AFIS-*.AppImage && ./AFIS-*.AppImage` |
| Linux, with menu entry | `AFIS-2.1.0-linux-x86_64.tar.gz` | `tar xzf AFIS-*.tar.gz && cd AFIS && ./install.sh`, then run `afis-gui` |
| **Windows** 10/11 (x64) | `AFIS-2.1.0-windows-x64.zip` | Unzip, then double-click `AFIS\AFIS.exe` |
| **macOS** 13+ (Apple silicon) | `AFIS-2.1.0-macos-arm64.zip` | Unzip, then right-click `AFIS.app` and choose **Open** the first time |

Check downloads against `SHA256SUMS.txt`. The builds are not code-signed, so Windows SmartScreen and macOS Gatekeeper will warn the first time you open them.

On a minimal Linux install, the window may not open until the Qt libraries are installed: `sudo apt install libegl1 libxkbcommon-x11-0 libxcb-cursor0 libfontconfig1`.

### Try it in a minute
Open AFIS → **Home** → **Load demo data** → **Identify** → *Browse…* → `demo-03_2.png`. The demo prints are synthetic and belong to no one.

### What's in this release
- Desktop app with these pages: Home (quick start and demo data), Analyse (every pipeline stage), Enroll, Verify (1:1), Identify (1:N), Database, Audit log and Settings
- Encrypted template store with these protections:
  - Fernet encryption
  - a key check
  - templates bound to their record
  - an HMAC-chained audit log
  - secure delete
  - typed confirmation before deleting everything
- A CLI covering every feature, with scriptable exit codes (`afis --help`)
- Accuracy evaluation tooling (FAR, FRR and EER) and a synthetic fingerprint generator

### Accuracy: please read
Measured on the public FVC2002 databases (real fingerprints from 4 sensors, held out from tuning):
- **One impression enrolled:** EER 15.3%. At the default threshold, about 0.2% of impostors are accepted and 40% of genuine attempts are rejected.
- **Three impressions enrolled per finger:** EER 9.4%, with 21.5% of genuine attempts rejected.

AFIS is a learning tool, far from commercial accuracy. See the README for full results, methodology and limitations.

### Responsible use
Match scores are probabilistic. AFIS is educational software and has **not** been validated for forensic, legal or law-enforcement decisions. Only enrol people who have given informed consent.
