"""Frozen spectrum-encoder probes: how much redshift does an embedding carry?

Self-contained version of the evaluation we ran on DESI spectra. It streams a
sample of DESI spectra, pushes them once through each frozen encoder, and fits
two cheap read-out probes (ridge regression and cosine kNN) to predict
spectroscopic redshift from the embeddings alone. No encoder is trained or
fine-tuned; the PCA baseline is the only thing fitted, and only on the train
split. Every encoder sees identical object-level splits, identical CV folds and
identical hyper-parameter grids, so the comparison is fair by construction.

Encoders compared
    aion        AION-1 encoder representation, DESI spectrum modality (768-d)
    specformer  AstroCLIP SpecFormer, from its pretrained checkpoint (768-d)
    pca         PCA on continuum-normalised log-flux, train-fitted (128-d)
    (a train-median-redshift baseline is always reported for reference)

Install
    pip install "numpy<3" scipy pandas scikit-learn datasets huggingface_hub
    pip install torch                 # for either neural encoder
    pip install polymathic-aion==0.0.2  # for --encoders aion; exact tested codec/package
    pip install lightning             # for --encoders specformer (checkpoint metadata)

    Keep the supplied `specformer_model.py` next to this script when using
    --encoders specformer. It is the frozen-inference architecture for the
    pinned checkpoint, which has no config.json and cannot be loaded through
    transformers alone.

    The AION Hugging Face revision below pins the main model weights.
    `polymathic-aion==0.0.2` also pins CodecManager and the spectrum codec that
    converts raw DESI arrays into the model's input tokens.

Run
    python spectrum_encoder_probes.py --sample-size 10000 --device cuda --out results/
    python spectrum_encoder_probes.py --sample-size 256 --encoders pca --device cpu

Cost note: 10,000 spectra through both neural encoders took ~2 h on one GPU.
Start small (--sample-size 256 --encoders pca) to check the plumbing first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration. Defaults reproduce the run reported to the team.
# --------------------------------------------------------------------------- #

DESI_REPO = "MultimodalUniverse/desi"
DESI_REVISION = "933b9056b93b0f3e7790ee28c43412b49c39e232"
AION_REPO = "polymathic-ai/aion-base"
AION_REVISION = "40541618104bab0fa85c8af68daeb867a720bb8c"
SPECFORMER_REPO = "polymathic-ai/specformer"
SPECFORMER_REVISION = "160d67f0c07daf33d192568ca60ff38d76c39d66"


@dataclass(frozen=True)
class ProbeConfig:
    sample_size: int = 10_000
    sample_seed: int = 20260717
    shuffle_buffer_size: Optional[int] = None
    split_seeds: Tuple[int, ...] = (20260717, 20260718, 20260719)
    train_ratio: float = 0.8
    cv_folds: int = 5
    ridge_alpha_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    knn_k: int = 10
    outlier_threshold: float = 0.15
    device: str = "cuda"
    batch_size: int = 64
    pca_components: int = 128
    pca_resample_pixels: int = 1024
    pca_continuum_window: int = 51  # odd, in pixels
    encoders: Tuple[str, ...] = ("aion", "specformer", "pca")


# --------------------------------------------------------------------------- #
# Data: stream a DESI sample, shape it, split it by object.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpectrumBatch:
    """Spectra on one shared wavelength grid, ready to embed."""

    object_id: np.ndarray  # (n,) str
    flux: np.ndarray  # (n, n_pix) float32
    wave: np.ndarray  # (n_pix,) float32, shared across rows
    ivar: Optional[np.ndarray] = None  # (n, n_pix) float32
    mask: Optional[np.ndarray] = None  # (n, n_pix) bool, True == bad pixel

    def __len__(self) -> int:
        return int(self.object_id.shape[0])

    def take(self, index: np.ndarray) -> "SpectrumBatch":
        return SpectrumBatch(
            object_id=self.object_id[index],
            flux=self.flux[index],
            wave=self.wave,
            ivar=None if self.ivar is None else self.ivar[index],
            mask=None if self.mask is None else self.mask[index],
        )


def stream_desi_sample(
    sample_size: int,
    seed: int,
    shuffle_buffer_size: Optional[int] = None,
) -> pd.DataFrame:
    """Stream `sample_size` good DESI spectra. No bulk download of the dataset.

    MultimodalUniverse stores ZWARN as a boolean "no problem" flag (True iff
    the raw DESI ZWARN == 0), so the good-spectrum filter is just `ZWARN is
    True`. The stream is normally shuffled by seed before taking. A buffer
    size of zero disables shuffling for tiny connectivity/RAM smoke tests.
    """
    from datasets import load_dataset

    dataset = load_dataset(DESI_REPO, revision=DESI_REVISION, split="train", streaming=True)
    dataset = dataset.filter(lambda row: bool(row["ZWARN"]) is True)
    if shuffle_buffer_size != 0:
        buffer_size = (
            max(sample_size * 10, 10_000)
            if shuffle_buffer_size is None
            else int(shuffle_buffer_size)
        )
        if buffer_size < sample_size:
            raise ValueError("shuffle_buffer_size must be zero or at least sample_size")
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    rows: List[Dict[str, Any]] = []
    for row in dataset:
        rows.append(row)
        if len(rows) >= sample_size:
            break
    if len(rows) < sample_size:
        raise RuntimeError(f"stream ended with {len(rows)} rows, wanted {sample_size}")
    return pd.DataFrame(rows)


def to_spectrum_batch(frame: pd.DataFrame) -> SpectrumBatch:
    """Unpack the nested `spectrum` struct into flat arrays.

    Each row's `spectrum` is a dict of fixed-length arrays (flux, ivar, lambda,
    mask, lsf_sigma) on DESI's instrument-fixed coadd grid, so one shared
    wavelength vector covers the whole sample.
    """

    def field_of(name: str, dtype: Any) -> np.ndarray:
        return np.stack([np.asarray(row[name], dtype=dtype) for row in frame["spectrum"]])

    wave_rows = field_of("lambda", np.float32)
    if not np.allclose(wave_rows, wave_rows[0]):
        raise ValueError("expected one shared wavelength grid across the sample")

    return SpectrumBatch(
        object_id=frame["object_id"].astype(str).to_numpy(),
        flux=field_of("flux", np.float32),
        wave=wave_rows[0],
        ivar=field_of("ivar", np.float32),
        mask=field_of("mask", bool),
    )


def object_level_split(object_ids: Sequence[str], seed: int, train_ratio: float) -> np.ndarray:
    """Deterministic, leakage-free train/test mask, hashed from the object ID.

    Hashing (rather than shuffling) means an object always lands on the same
    side for a given seed, whatever order or subset it arrives in.
    """

    def fraction(object_id: str) -> float:
        digest = hashlib.sha256(f"{object_id}|{seed}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    return np.array([fraction(str(value)) < train_ratio for value in object_ids], dtype=bool)


# --------------------------------------------------------------------------- #
# Encoders. Each one takes a SpectrumBatch and returns (n, dim) float32.
# --------------------------------------------------------------------------- #


def resolve_device(declared: str) -> str:
    """Declared device, falling back to CPU when CUDA is not actually there."""
    if declared not in {"cuda", "cpu"}:
        raise ValueError(f"device must be 'cuda' or 'cpu', got {declared!r}")
    if declared == "cpu":
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def chunks(n: int, size: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, n, size):
        yield start, min(start + size, n)


class SpectrumEncoder:
    """Shared interface. `fit` is a no-op for the pretrained encoders."""

    name: str
    output_dim: int

    def fit(self, train_batch: SpectrumBatch) -> "SpectrumEncoder":
        return self

    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        raise NotImplementedError


class PCAEncoder(SpectrumEncoder):
    """Classical baseline: PCA of continuum-normalised log-flux, train-fitted.

    Anchors what "no pretrained encoder at all" buys. CPU-only by design.
    """

    def __init__(self, config: ProbeConfig) -> None:
        self.name = "pca"
        self.output_dim = config.pca_components
        self._resample_pixels = config.pca_resample_pixels
        self._window = config.pca_continuum_window
        self._pca = PCA(n_components=config.pca_components, random_state=0)
        self._grid: Optional[np.ndarray] = None
        self._fitted = False
        if self._window % 2 == 0:
            raise ValueError("pca_continuum_window must be odd")

    def _features(self, batch: SpectrumBatch) -> np.ndarray:
        if self._grid is None:
            self._grid = np.linspace(
                float(batch.wave.min()), float(batch.wave.max()), self._resample_pixels
            )
        flux = np.asarray(batch.flux, dtype=np.float64)
        wave = np.asarray(batch.wave, dtype=np.float64)
        resampled = np.stack([np.interp(self._grid, wave, row) for row in flux])

        # Sliding-median continuum, then log-flux: removes the overall shape so
        # the components describe line structure rather than brightness.
        continuum = median_filter(resampled, size=(1, self._window), mode="nearest")
        continuum = np.where(continuum > 0, continuum, np.nan)
        normalised = np.nan_to_num(resampled / continuum, nan=1.0, posinf=1.0, neginf=1.0)
        return np.log(np.clip(normalised, 1e-3, None)).astype(np.float32)

    def fit(self, train_batch: SpectrumBatch) -> "PCAEncoder":
        self._pca.fit(self._features(train_batch))
        self._fitted = True
        return self

    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PCAEncoder.embed called before fit() on the train split")
        return self._pca.transform(self._features(batch)).astype(np.float32)


class AionEncoder(SpectrumEncoder):
    """AION-1 encoder representation for the DESI spectrum modality.

    The codec turns raw flux/ivar/mask/wavelength into tokens, the model's
    encoder consumes them, and we mean-pool over the token axis — the same
    pooling AION's own property-prediction example uses.
    """

    def __init__(self, config: ProbeConfig, num_encoder_tokens: int = 600) -> None:
        self.name = "aion"
        self.output_dim = 768
        self._device = resolve_device(config.device)
        self._batch_size = config.batch_size
        self._num_encoder_tokens = num_encoder_tokens
        self._model: Any = None
        self._codecs: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from aion import AION
        from aion.codecs import CodecManager

        model = AION.from_pretrained(AION_REPO, revision=AION_REVISION)
        self._model = model.to(self._device).eval().requires_grad_(False)
        self._codecs = CodecManager(device=self._device)

    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        import torch
        from aion.modalities import DESISpectrum

        self._load()
        pooled_chunks = []
        with torch.no_grad():
            for start, end in chunks(len(batch), self._batch_size):
                flux = torch.as_tensor(batch.flux[start:end], dtype=torch.float32, device=self._device)
                wavelength = torch.as_tensor(
                    batch.wave, dtype=torch.float32, device=self._device
                ).unsqueeze(0).expand(flux.shape[0], -1)
                ivar = (
                    torch.as_tensor(batch.ivar[start:end], dtype=torch.float32, device=self._device)
                    if batch.ivar is not None
                    else torch.ones_like(flux)
                )
                mask = (
                    torch.as_tensor(batch.mask[start:end], dtype=torch.bool, device=self._device)
                    if batch.mask is not None
                    else torch.zeros_like(flux, dtype=torch.bool)
                )
                # The codec pads and re-grids internally, so raw DESI arrays go
                # straight through with no manual resampling.
                spectrum = DESISpectrum(flux=flux, ivar=ivar, mask=mask, wavelength=wavelength)
                tokens = self._codecs.encode(spectrum)
                encoded = self._model.encode(tokens, num_encoder_tokens=self._num_encoder_tokens)
                pooled_chunks.append(encoded.mean(dim=1).to(torch.float32).cpu().numpy())
        return np.concatenate(pooled_chunks, axis=0).astype(np.float32)


class SpecFormerEncoder(SpectrumEncoder):
    """AstroCLIP SpecFormer, loaded from its published Lightning checkpoint.

    SpecFormer takes flux only — no wavelength or ivar input — and normalises
    internally, so the input is just (batch, length, 1). The output is
    (batch, sections, 768), mean-pooled over sections as in AstroCLIP's own
    embedding script.
    """

    def __init__(self, config: ProbeConfig) -> None:
        self.name = "specformer"
        self.output_dim = 768
        self._device = resolve_device(config.device)
        self._batch_size = config.batch_size
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download

        try:
            from specformer_model import SpecFormer
        except ImportError as error:  # pragma: no cover - environment guidance
            raise ImportError(
                "SpecFormer needs the vendored architecture file. Copy "
                "src/spec_probes/specformer_model.py from AION-Search next to this "
                "script. The published checkpoint has no config.json, so it cannot "
                "be loaded through transformers."
            ) from error

        path = hf_hub_download(
            repo_id=SPECFORMER_REPO, filename="specformer.ckpt", revision=SPECFORMER_REVISION
        )
        # weights_only=False: the checkpoint's hyper_parameters entry is a
        # Lightning AttributeDict, not tensors. This is the pinned, MIT-licensed
        # single-file checkpoint, not an arbitrary download.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Build from the checkpoint's own hyper-parameters so the architecture
        # can never drift from the weights.
        model = SpecFormer(**dict(checkpoint["hyper_parameters"]))
        model.load_state_dict(checkpoint["state_dict"])
        self._model = model.to(self._device).eval().requires_grad_(False)

    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        import torch

        self._load()
        pooled_chunks = []
        with torch.no_grad():
            for start, end in chunks(len(batch), self._batch_size):
                flux = torch.as_tensor(
                    batch.flux[start:end], dtype=torch.float32, device=self._device
                ).unsqueeze(-1)
                pooled = self._model(flux)["embedding"].mean(dim=1)
                pooled_chunks.append(pooled.to(torch.float32).cpu().numpy())
        return np.concatenate(pooled_chunks, axis=0).astype(np.float32)


def build_encoder(name: str, config: ProbeConfig) -> SpectrumEncoder:
    builders = {"pca": PCAEncoder, "aion": AionEncoder, "specformer": SpecFormerEncoder}
    if name not in builders:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(builders)}")
    return builders[name](config)


# --------------------------------------------------------------------------- #
# Probes. Pure functions: embeddings and labels in, predictions out.
# --------------------------------------------------------------------------- #


def make_cv_folds(n_samples: int, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Fold indices computed once per split seed and reused across encoders.

    Passing the identical index arrays to every encoder makes "same folds" a
    structural guarantee, not a probabilistic consequence of a shared seed.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kfold.split(np.arange(n_samples)))


def select_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: Sequence[float],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """Lowest mean cross-validated MAE wins; ties go to the smaller alpha.

    The scaler is refitted inside each fold on that fold's training rows only,
    so the validation rows never contribute scaling statistics.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    scaled_folds = []
    for train_index, val_index in folds:
        scaler = StandardScaler().fit(x_train[train_index])
        scaled_folds.append(
            (train_index, val_index, scaler.transform(x_train[train_index]), scaler.transform(x_train[val_index]))
        )

    best_alpha, best_error = None, np.inf
    for alpha in sorted(float(value) for value in alpha_grid):
        errors = []
        for train_index, val_index, fold_train, fold_val in scaled_folds:
            model = Ridge(alpha=alpha, solver="svd").fit(fold_train, y_train[train_index])
            errors.append(np.mean(np.abs(model.predict(fold_val) - y_train[val_index])))
        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_alpha, best_error = alpha, mean_error
    return float(best_alpha)


def probe_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    config: ProbeConfig,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Fit both probes on one standardised train split and predict the test set.

    Standardising on the train split only puts encoders of very different
    representation scale (128-d vs 768-d) on the same footing, for the ridge
    penalty and the cosine distance alike.
    """
    alpha = select_ridge_alpha(x_train, y_train, config.ridge_alpha_grid, folds)
    scaler = StandardScaler().fit(np.asarray(x_train, dtype=np.float64))
    train_scaled = scaler.transform(np.asarray(x_train, dtype=np.float64))
    test_scaled = scaler.transform(np.asarray(x_test, dtype=np.float64))

    # solver="svd" is pinned rather than "auto": it is the one Ridge solver with
    # no version-dependent SciPy keyword surface.
    ridge = Ridge(alpha=alpha, solver="svd").fit(train_scaled, y_train)
    knn = KNeighborsRegressor(
        n_neighbors=config.knn_k, metric="cosine", algorithm="brute"
    ).fit(train_scaled, y_train)
    return {
        "linear": ridge.predict(test_scaled),
        "knn": knn.predict(test_scaled),
        "_alpha": np.array([alpha]),
    }


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #


def redshift_metrics(z_pred: np.ndarray, z_true: np.ndarray, threshold: float) -> Dict[str, float]:
    """The standard spec-z read-out: NMAD, catastrophic outliers, MAE, R2.

    The residual is (z_pred - z_true) / (1 + z_true) throughout. NMAD describes
    the bulk of the distribution and ignores the tail, so always read it next to
    the outlier fraction — a sharp encoder can still fail badly on a few percent
    of objects.
    """
    z_pred = np.asarray(z_pred, dtype=np.float64)
    z_true = np.asarray(z_true, dtype=np.float64)
    residual = (z_pred - z_true) / (1.0 + z_true)
    total_variance = float(np.sum((z_true - np.mean(z_true)) ** 2))
    return {
        "nmad": float(1.4826 * np.median(np.abs(residual - np.median(residual)))),
        "catastrophic_outlier_fraction": float(np.mean(np.abs(residual) > threshold)),
        "mae": float(np.mean(np.abs(z_pred - z_true))),
        "r2": float(1.0 - np.sum((z_true - z_pred) ** 2) / total_variance),
        "n": int(z_true.size),
    }


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def run_one_seed(
    batch: SpectrumBatch,
    redshift: np.ndarray,
    frozen_embeddings: Mapping[str, np.ndarray],
    split_seed: int,
    config: ProbeConfig,
) -> pd.DataFrame:
    """Every encoder, both probes, one train/test split. Returns row-level rows.

    Row-level predictions are the primary evidence; the metric tables are
    recomputed from them, never accumulated alongside them.
    """
    is_train = object_level_split(batch.object_id, split_seed, config.train_ratio)
    train_index = np.flatnonzero(is_train)
    test_index = np.flatnonzero(~is_train)
    train_batch, test_batch = batch.take(train_index), batch.take(test_index)
    z_train, z_test = redshift[train_index], redshift[test_index]
    folds = make_cv_folds(len(train_index), config.cv_folds, split_seed)

    print(f"  seed {split_seed}: {len(train_index)} train / {len(test_index)} test")
    rows: List[Dict[str, Any]] = []

    def record(encoder_name: str, probe: str, predictions: np.ndarray) -> None:
        rows.extend(
            {
                "object_id": object_id,
                "encoder": encoder_name,
                "probe": probe,
                "split_seed": split_seed,
                "z_true": float(true_value),
                "z_pred": float(predicted_value),
            }
            for object_id, true_value, predicted_value in zip(test_batch.object_id, z_test, predictions)
        )

    # Trivial reference point: predict the train-split median for everything.
    record("median_baseline", "baseline", np.full(len(test_index), float(np.median(z_train))))

    for encoder_name in config.encoders:
        if encoder_name == "pca":
            # PCA is the only train-fitted encoder, so it must be refitted for
            # each outer split. Frozen neural embeddings are reused below.
            encoder = PCAEncoder(config).fit(train_batch)
            embeddings_train = encoder.embed(train_batch)
            embeddings_test = encoder.embed(test_batch)
        else:
            embeddings = frozen_embeddings[encoder_name]
            embeddings_train = embeddings[train_index]
            embeddings_test = embeddings[test_index]

        predictions = probe_predictions(embeddings_train, z_train, embeddings_test, config, folds)
        print(f"    {encoder_name}: dim={embeddings_train.shape[1]} alpha={predictions['_alpha'][0]:g}")
        for probe in ("linear", "knn"):
            record(encoder_name, probe, predictions[probe])

    return pd.DataFrame(rows)


def summarise(predictions: pd.DataFrame, config: ProbeConfig) -> pd.DataFrame:
    """Per-seed metrics from the rows, then mean and spread across split seeds."""
    per_seed = [
        {"encoder": encoder, "probe": probe, "split_seed": seed,
         **redshift_metrics(group["z_pred"].to_numpy(), group["z_true"].to_numpy(), config.outlier_threshold)}
        for (encoder, probe, seed), group in predictions.groupby(["encoder", "probe", "split_seed"])
    ]
    def population_std(values: pd.Series) -> float:
        # ddof=0: the spread of the seeds we actually ran, not an estimate of a
        # wider population.
        return float(np.std(values.to_numpy(dtype=np.float64), ddof=0))

    frame = pd.DataFrame(per_seed)
    metric_columns = ["nmad", "catastrophic_outlier_fraction", "mae", "r2"]
    summary = frame.groupby(["encoder", "probe"])[metric_columns].agg(["mean", population_std])
    summary.columns = [
        f"{metric}_{'std' if statistic == 'population_std' else statistic}"
        for metric, statistic in summary.columns
    ]
    summary["n_seeds"] = frame.groupby(["encoder", "probe"]).size()
    return summary.reset_index().sort_values(["probe", "nmad_mean"])


def embed_frozen_encoders(
    batch: SpectrumBatch,
    config: ProbeConfig,
) -> Dict[str, np.ndarray]:
    """Embed the full sample once per frozen neural encoder.

    The returned arrays keep the same row order as ``batch``. Each split can
    therefore select train and test rows by index without another GPU pass.
    PCA is intentionally excluded because it must be fitted separately on
    each split's training rows.
    """
    frozen: Dict[str, np.ndarray] = {}
    for encoder_name in config.encoders:
        if encoder_name == "pca":
            continue

        print(f"embedding full sample once with {encoder_name}")
        encoder = build_encoder(encoder_name, config)
        embeddings = encoder.embed(batch)
        expected_shape = (len(batch), encoder.output_dim)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"{encoder_name} produced shape {embeddings.shape}, expected {expected_shape}"
            )
        frozen[encoder_name] = embeddings

        # Do not keep both large neural models resident on the GPU. The NumPy
        # embeddings above remain available after the encoder is released.
        del encoder
        if config.device == "cuda":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return frozen


def main() -> None:
    defaults = ProbeConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-size", type=int, default=defaults.sample_size)
    parser.add_argument("--sample-seed", type=int, default=defaults.sample_seed)
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=defaults.shuffle_buffer_size,
        help="streaming shuffle rows; use 0 only for a tiny unshuffled smoke test",
    )
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(defaults.split_seeds))
    parser.add_argument("--encoders", nargs="+", default=list(defaults.encoders),
                        choices=["aion", "specformer", "pca"])
    parser.add_argument("--device", default=defaults.device, choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--out", type=Path, default=Path("probe_results"))
    parser.add_argument("--cache", type=Path, default=None,
                        help="npz path for the streamed sample; reused if it exists")
    args = parser.parse_args()

    config = ProbeConfig(
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        split_seeds=tuple(args.split_seeds),
        encoders=tuple(args.encoders),
        device=args.device,
        batch_size=args.batch_size,
    )

    if args.cache is not None and args.cache.exists():
        print(f"loading cached sample from {args.cache}")
        with np.load(args.cache, allow_pickle=False) as payload:
            batch = SpectrumBatch(
                object_id=payload["object_id"], flux=payload["flux"], wave=payload["wave"],
                ivar=payload["ivar"], mask=payload["mask"],
            )
            redshift = payload["z"]
    else:
        print(f"streaming {config.sample_size} DESI spectra (ZWARN good only)")
        frame = stream_desi_sample(
            config.sample_size,
            config.sample_seed,
            config.shuffle_buffer_size,
        )
        batch = to_spectrum_batch(frame)
        redshift = frame["Z"].to_numpy(dtype=np.float64)
        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.cache, object_id=batch.object_id, flux=batch.flux,
                                wave=batch.wave, ivar=batch.ivar, mask=batch.mask, z=redshift)
            print(f"cached sample to {args.cache}")

    print(f"{len(batch)} spectra, {batch.flux.shape[1]} pixels, z in "
          f"[{redshift.min():.3f}, {redshift.max():.3f}]")

    frozen_embeddings = embed_frozen_encoders(batch, config)
    predictions = pd.concat(
        [
            run_one_seed(batch, redshift, frozen_embeddings, seed, config)
            for seed in config.split_seeds
        ],
        ignore_index=True,
    )
    summary = summarise(predictions, config)

    args.out.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.out / "predictions.parquet", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "config.json").write_text(json.dumps(asdict(config), indent=2, default=list))

    print(f"\nspec-z recovery, mean over {len(config.split_seeds)} split seeds "
          f"(lower NMAD / outliers is better)\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nwrote {args.out / 'predictions.parquet'} and {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
