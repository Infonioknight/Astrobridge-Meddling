# Architecture — Design Choices and Literature

## Pipeline

Frozen domain encoders → per-modality projector + learned modality embedding → shared Q-Former (32 queries)
→ adapter → prefix embeddings into Qwen3.8-27B → caption. Stage 1 trains the fusion stack against a frozen
LLM; stage 2 adds LoRA to the LLM while the fusion stack continues training at a lower learning rate.

---

## Frozen encoders, trained bridge, frozen LLM

| Paper | arXiv |
|---|---|
| Frozen — Tsimpoukelli et al. 2021 | 2106.13884 |
| BLIP-2 — Li et al. 2023 | 2301.12597 |
| LLaVA — Liu et al. 2023 | 2304.08485 |

Frozen establishes that a trained visual prefix into a frozen language model yields a working captioner.
BLIP-2 pairs a frozen image encoder and frozen LLM with a small trained Q-Former between them, in two
stages. LLaVA uses a projector-only bridge with an align-then-instruction-tune split.

BLIP-2's first stage uses contrastive and image-text matching objectives against the frozen encoder before
the language-model stage. We use captioning loss in both stages; the contrastive objectives assume pair
counts several orders of magnitude above what is available here.

---

## Fixed-length bottleneck over variable-length input

| Paper | arXiv |
|---|---|
| BLIP-2 — Li et al. 2023 | 2301.12597 |
| Flamingo — Alayrac et al. 2022 | 2204.14198 |

BLIP-2 supplies the learned-query cross-attention mechanism and the choice of 32 queries. Flamingo's
Perceiver Resampler establishes that a fixed number of latents can absorb variable-length, variable-count
visual input, which is what allows the sequence length seen by the LLM to stay constant regardless of which
modalities are present.

In BLIP-2 the Q-Former attends over a single modality. Here one shared query set cross-attends jointly over
the concatenated tokens of all present modalities, producing a single fused representation rather than a
per-modality one. The module is scaled down accordingly: 3 layers, d=384, ~7M parameters, against BLIP-2's
188M.

---

## Prefix injection

| Paper | arXiv |
|---|---|
| Prefix-Tuning — Li & Liang 2021 | 2101.00190 |
| Prompt Tuning — Lester et al. 2021 | 2104.08691 |
| Frozen — Tsimpoukelli et al. 2021 | 2106.13884 |

Prefix-Tuning and Prompt Tuning establish that continuous vectors prepended in embedding space steer a
frozen language model. Frozen supplies the input-conditioned form, where the prefix is computed from the
input rather than learned once and fixed.

Terminology: "soft prompt" in the prompt-tuning literature denotes a task vector learned once. The prefix
here is computed per object, so BLIP-2 and LLaVA usage — visual prefix, prefix embeddings — is more precise.

---

## Modality embeddings and absent modalities

| Paper | arXiv |
|---|---|
| ImageBind — Girdhar et al. 2023 | 2305.05665 |
| AION-1 — Parker, Lanusse, Shen et al. 2025 | 2510.17960 |

ImageBind uses per-modality projection into a shared space with modality-specific parameters alongside
shared ones. AION-1 applies modality-specific tokenisation followed by transformer modelling over
cross-modal token sequences, trained with masked modelling across modality subsets, so arbitrary subsets
being present or absent is native to the formulation.

Absent modalities contribute zero rows to the token sequence and are excluded through the attention mask
rather than represented by zero vectors. A zero-vector placeholder is a constant the model can learn to
read as a presence indicator, which permits a modality-appropriate caption to be produced without attending
to that modality's content, and this is not visible in the training loss. The constraint is enforced by unit
test.

---

## Two-stage training and LoRA

| Paper | arXiv |
|---|---|
| AstroLLaVA — Zaman, Smith et al. 2025 | 2504.08583 |
| LoRA — Hu et al. 2021 | 2106.09685 |
| QLoRA — Dettmers et al. 2023 | 2305.14314 |

AstroLLaVA applies two-stage fine-tuning to a LLaVA-family model in the astronomy domain and is the direct
in-domain precedent for the staging. LoRA supplies the low-rank adaptation used at stage 2; QLoRA supplies
the 4-bit fallback for single-GPU configurations.

Full fine-tuning is not used: at the available joint-object count, updating all weights of a 28B decoder
overfits and degrades the general reasoning capacity that motivated selecting a large decoder.

---

## Encoders

| Paper | arXiv |
|---|---|
| AION-1 — Parker, Lanusse, Shen et al. 2025 | 2510.17960 |
| The Multimodal Universe — MMU Collaboration 2024 | 2412.02527 |
| AstroCLIP — Lanusse et al. 2023 | 2310.03024 |
| AstroPT — Smith et al. 2024 | 2405.14930 |

AION-1 spans 300M–3.1B parameters over Legacy Survey, HSC, SDSS, DESI and Gaia, covering roughly 200M
observations, and reports strong downstream results from a single frozen encoder — the usage pattern here.
The Multimodal Universe is the underlying dataset and the substrate for the crossmatch. AstroCLIP is the
closest prior work on the image–spectrum pair specifically. AstroPT is the image-encoder baseline.

Encoder selection also rests on internal benchmarking: on a 12-seed linear-probe comparison AION-1
outperformed AstroPT on stellar-mass regression with roughly a third the seed variance, and band-label
awareness was verified empirically, in that relabelling a channel changes the embedding while permuting
channels and labels together does not.

Encoder outputs are used token-level rather than mean-pooled, since the Q-Former cross-attends over
structure that pooling removes.

---

## Grounding evaluation

| Paper | arXiv |
|---|---|
| Object Hallucination in Image Captioning — Rohrbach et al. 2018 | 1809.02156 |
| POPE — Li et al. 2023 | 2305.10355 |

CHAIR measures hallucination at the level of individual claims rather than by text similarity; POPE probes
hallucination with targeted queries rather than generation metrics. Both support claim-level evaluation
over BLEU-style scoring.

Three evaluation components are specific to this setup:

Caption content is restricted to what the presented modality subset supports, with a separate caption per
subset. Training on a single caption regardless of what is shown associates modality-specific quantities
with objects rather than with the evidence for them.

The shuffle test substitutes another object's modality content while holding presence flags fixed. The
caption, and specifically the claims attributed to that modality, must change. This detects a model reading
presence flags rather than content, which does not appear in the loss.

Joint-claim fraction reports the proportion of claims requiring two or more modalities.

---

## Areas without direct precedent

A shared Q-Former fusing heterogeneous scientific modalities into a single query set for captioning:
BLIP-2 is single-modality, AION-1 fuses without captioning, AstroLLaVA captions from images only.

Provenance-restricted caption tiers, where the caption varies with the modality subset presented.

Whether pairwise supervision combined with unimodal context induces joint reasoning when
triple-modality data is scarce.

---

## Citation status

AION-1, the Multimodal Universe and AstroLLaVA identifiers were verified against arXiv. AstroCLIP
(2310.03024) and AstroPT (2405.14930) are unverified and should be confirmed before entering the
bibliography.
