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

---

Using DINOv2 as a choice of vision encoder
Noticed quite a few explorations have been done using DINO as a choice of vision encoder (something I did notice and figure since I’ve noticed it does perform pretty damn well off the shelf on multiple other - albeit less noisy - tasks)

Couple of relevant findings: 
- https://arxiv.org/abs/2310.03024 uses a “carefully modified” version of the dinov2 imagenet checkpoint, fine-tuned on curated galaxy images.
- A solid eval paper from what I can skim from Matt! (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6911300)
- https://cs231n.stanford.edu/2025/papers/text_file_840589796-AstroDINO_final.pdf which uses DINO off the shelf (“end goal is to have a DINO based model that can do semantic segmentation on astronomical imaging.” to quote the paper)

so what i did for my jwst images, was pull a small sample (7) of rows from jwst ceers using mmu hats, then used the flux bands of a single channel and plotted them into a greyscale image along with the mask to ensure the image was an image (lol) and also didnt contain noisy or useless pixels (which the mask helped filter), also used the ivar x flux to have a baseline on how clear the distinction of the object of interest was from the rest of the noisy image.
using this as my image (flux x mask) - passed through dinov2, extracted the output of the final layer of the backbone and rendered attention maps (which worked surprisingly really well based off an eyeball survey)
Then I passed the same noisy flux x mask images into gemini-3.6-flash (thanks Ibrahim!) and got descriptions of the same, highlighting points of interest/regions of interest + astronomical interpretation (which I unfortunatelty cannot verify due to my current domain knowledge) - I eyeballed the captions on locality, position etc + confidence with what i observed in the attention maps! (directly input to the web UI for the gemini model)
What I can probably say is the model of choice here does a decently good job at visually identifying areas of interest in an image from what I see.
Whew.

Another interesting benchmarking paper (I haven’t read much, just title + bit of the abstract as of now, but looks like it spans multiple modalities and how they work with VLMs) - https://arxiv.org/abs/2604.24589

Bits of already known information - there has already been work done in Spectra-Image alignment https://arxiv.org/abs/2310.03024

Attaching link to all my exploratory work for perusal (README file contains all the specifics)
https://github.com/Infonioknight/Astrobridge-Meddling


