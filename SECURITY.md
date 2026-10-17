# Security & privacy

AFIS stores biometric data. This document describes how that data is protected, what the protections do **not** cover, and how to report a problem.

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/thespaceio/AFIS/security/advisories/new) instead of a public issue, and include steps to reproduce. You should get a reply within a week.

## What is stored, and where

| Item | Location | Protection |
|---|---|---|
| Templates (minutiae + pattern) | `afis_templates.db` in the per-user data folder | Encrypted with Fernet (AES-128-CBC + HMAC-SHA256) |
| Enrolment images (optional) | Same database | Encrypted with Fernet; turn off under **Settings → Privacy** to store templates only |
| Person ID, finger, pattern, minutiae count, quality, time | Same database, plaintext columns | Needed to list records without decrypting; treat the person ID as personal data |
| Audit log | Same database, plaintext | HMAC hash chain makes edits detectable |
| Encryption key | `afis_templates.key` next to the database, or the `AFIS_KEY` environment variable | File created with owner-only permissions (0600 on Linux/macOS) |
| Settings | `settings.json` next to the database | Threshold, image-storage choice, notice acknowledgement |

Data folder: Linux `~/.local/share/afis`, macOS `~/Library/Application Support/AFIS`, Windows `%APPDATA%\AFIS` (created with 0700 permissions on Linux/macOS). Override with `AFIS_HOME` or `AFIS_DB`. Nothing is sent over the network.

## Protections

- **Encryption at rest.** Templates and images are encrypted before they are written.
- **Key check.** A known value encrypted with the key is stored in the database. A wrong key is rejected when the database is opened. If the key file is missing for a database that holds data, AFIS refuses to create a new key, which would make the data unreadable and could be mistaken for a fresh start.
- **Integrity and binding.** Each template carries a SHA-256 hash over its person ID, template ID, finger and minutiae, and is checked against the database row it was read from. A template copied into another person's record is rejected (`TemplateIntegrityError`).
- **No unsafe deserialisation.** Templates are JSON. AFIS 1.x used `pickle`, which can run code from a tampered file.
- **Parameterised SQL** everywhere.
- **Tamper-evident audit log.** Every enrolment, verification, identification and deletion is logged. Each entry includes an HMAC (keyed from the encryption key) over its contents and the previous entry's HMAC. An encrypted chain head reveals entries removed from the end. Check it with **Audit log → Verify log integrity** or `afis check`.
- **Deliberate deletion.** Deleting a person or template asks for confirmation, with Cancel as the default. Deleting everything requires typing `DELETE`. Deleted rows are overwritten on disk (`PRAGMA secure_delete`), and the database is vacuumed after a full wipe. The uninstaller keeps your data unless you pass `--purge` and type `DELETE`.
- **Input hardening.** Image files are size-limited (40 MB, 40 megapixels) and decoded by content, not extension. Person IDs are restricted to letters, digits, combining accents, spaces and `. _ - @ '`.
- **Repository hygiene.** `.gitignore` excludes `*.db`, `*.key`, `settings.json`, `samples/*`, `data/` and `results/`. The bundled demo prints are synthetic.

## Limits: what this does not protect against

- **Anyone with both the database and the key can read everything.** The key file sits next to the database for convenience. For stronger separation, keep the key elsewhere and provide it through `AFIS_KEY`, or use full-disk encryption.
- **The audit chain is tamper-evident, not tamper-proof.** Someone holding the key can rewrite the whole log consistently, and an attacker could roll back to an older copy of the database file. For real accountability, export the log (`afis audit --export log.csv`) to separate, append-only storage.
- **Malware running as your user** can read the key and the decrypted data in memory.
- **No user accounts or access control.** Anyone who can run AFIS as your OS user can enrol, search and delete.
- **Templates are not cancellable biometrics.** A leaked template can't be "changed" like a password.
- The builds are not code-signed. Download them only from the official GitHub Releases page and check `SHA256SUMS.txt`.

## Responsible use

Fingerprints are sensitive personal data. Laws such as the GDPR (EU/UK), BIPA (Illinois) and the NDPR/NDPA (Nigeria) may require consent, a stated purpose, retention limits and security measures. Only enrol people who have given informed consent, and delete records you no longer need.

Match scores are **probabilistic**. AFIS has not been independently validated and must not be used as the sole basis for forensic, legal, employment, immigration or law-enforcement decisions. See the README for the measured error rates and their limits.
