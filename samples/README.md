# Sample fingerprints

Real fingerprint images are **not** included in this repository. They are biometric data, and dataset licences usually forbid redistribution. Everything in this folder except this README is ignored by git.

- **To try AFIS:** use the bundled *synthetic* demo prints in [`afis/demo_prints/`](../afis/demo_prints) (**Home → Load demo data** in the app, or `afis demo`).
- **To test on real prints:** download the free **FVC2002** databases from the official site, http://bias.csr.unibo.it/fvc2002/databases.asp. Each `DBn_B.zip` holds 80 images (10 fingers × 8 impressions). Extract the `.tif` files here (e.g. `samples/101_1.tif … samples/110_8.tif`) for the extra tests in `tests/`, or anywhere for evaluation:

```bash
python scripts/evaluate.py path/to/DB1_B path/to/DB2_B --all-impostors
```

File names follow `<finger>_<impression>.<ext>`, which is what `scripts/evaluate.py` expects. Check the dataset's terms before you use it, and don't commit or redistribute the images.
