# Sample fingerprints

Fingerprint images are not included in this repository (biometric data and dataset licences).

To try AFIS or run the full test suite, use the free **FVC2002** databases:

1. Download `DB1_B.zip` (80 images: 10 fingers × 8 impressions) from the FVC2002 site:
   http://bias.csr.unibo.it/fvc2002/databases.asp
2. Extract the `.tif` files into this folder, e.g. `samples/101_1.tif … samples/110_8.tif`.

File names follow `<finger>_<impression>.tif`, which is what `scripts/evaluate.py` expects.
