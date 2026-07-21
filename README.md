# Astrobridge-Meddling

- **`sandbox.ipynb`** — General messing around. Learnt to use HATS, pull images from jwst_ceers using hats and understanding how to use the flux, ivar and mask values.

- **`vis_encoder_work.ipynb`** — the vision encoder side specifically: feeding the
  rendered images through DINOv2 and pulling out attention maps.

- **`dino_visualizer.py`** — Helper script for ease of extraction of attention maps from the last layer of the backbone (had made this a while back, thankfully came in handy)

- **`LSDB/`** — stuff related to pulling data via lsdb / the HATS catalog format.

- **`jwst_masked_flux/`** — saved PNGs of the flux images after cleaning with the
  mask - fed into dinov2 and gemini

- **`jwst_snr/`** — saved PNGs of the signal-to-noise maps for the same objects. Used to sanity-check whether a source is a real detection or just noise, not used as model input.

- **`attention_maps/`** — DINOv2's attention map outputs for the JWST images.

- **`gemini-3.6-flash-notes.txt`** — the captions/descriptions Gemini generated for
  the same set of images (plus prompt at the top)

- **`random_results/`** — exactly what it sounds like. Odds and ends, not
  necessarily curated (initial hats pull + figuring that bit out)


Nothing here is validated or benchmarked yet — this is a first qualitative pass,
not a result.