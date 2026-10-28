# Bayesian Test-Time Adaptation via Dirichlet Feature Projection and GMM-Driven Inference (BTTA-DG)

This repository contains the official implementation of the paper:  
**"Bayesian Test-Time Adaptation via Dirichlet Feature Projection and GMM-Driven Inference for Motor Imagery EEG Decoding"** (ICLR 2026)

## Abstract
Electroencephalography (EEG)-based motor imagery (MI) brain-computer interfaces (BCIs) face significant generalization challenges due to cross-subject and cross-session variability. We propose **BTTA-DG**, a Bayesian test-time adaptation framework that combines **Dirichlet feature projection** and **Gaussian Mixture Model (GMM)-driven inference** to address these challenges. Our method:
- Utilizes a lightweight `SincAdaptNet` with learnable Sinc filters to capture task-specific frequency bands (mu, beta, gamma).
- Projects high-dimensional EEG embeddings onto a compact Dirichlet parameter space for modeling time-varying categorical embeddings.
- Employs gradient-free Bayesian-style calibration: GMM clustering refines predictions without backpropagation at test time.

The framework achieves strong accuracy while operating at real-time speeds.

---

## Installation

### Dependencies
- Python 3.10
- PyTorch (see `requirements.txt`; currently pinned to **2.3.1** with matching `torchvision` / `torchaudio`)
- NumPy, SciPy, Pandas
- scikit-learn (GMM clustering)
- tqdm
- MOABB / MNE (dataset download and handling; see `download_data.py`)

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note:** If `pip install -r requirements.txt` fails (e.g. Conda-specific `file://...` entries or platform-specific wheels in a frozen export), create a clean virtual environment and install the core stack manually (PyTorch, NumPy, SciPy, Pandas, scikit-learn, tqdm, MOABB, MNE), or install `torch` from the [official PyTorch instructions](https://pytorch.org/) for your CUDA version.

## Dataset
Supported datasets:
- **BNCI2014001**: 9 subjects, 22 channels, 250Hz
- **BNCI2014002**: 14 subjects, 15 channels, 512Hz
- **BNCI2015001**: 12 subjects, 13 channels, 512Hz

Download and preprocess (run from the repository root; writes `X.npy`, `labels.npy`, and `meta.csv` under `./data/{dataset_name}/`):

```bash
python download_data.py
```

Keep data under `./data/{dataset_name}/` as produced by the script.

## Code Structure
### Key Components
- **SincAdaptNet**: Implements learnable band-pass filters (SincConv) for frequency-specific feature extraction.
Located in `SincAdaptNet.py`.

- **BTTA-DG Core**:

  - `BTTA_DG.py`: Implements BTTA-DG method that combines Dirichlet feature projection (`DirichletEstimator`) and GMM-driven inference (`OnlineClustererGMM`).

  - `pretrain_SincAdaptNet.py`: Pre-trains SincAdaptNet on source domains.


## Usage

**Hardware:** Training and adaptation scripts use **CUDA** (`.cuda()`) and set `CUDA_VISIBLE_DEVICES` in each script’s `if __name__ == '__main__'` block. For CPU-only runs, change device handling accordingly.

**Recommended pipeline:** download data → pretrain → test-time adaptation.

### 1. Pre-Training
Train SincAdaptNet on source domains. By default, `pretrain_SincAdaptNet.py` loops over `BNCI2014001`, `BNCI2014002`, and `BNCI2015001`; edit the script to run a single dataset if needed.

```bash
python pretrain_SincAdaptNet.py
```

### 2. Test-Time Adaptation
Run BTTA-DG on the held-out subject. Set `data_name` in `BTTA_DG.py` and ensure pretrained checkpoints exist under `./baseline_SincAdaptNet_{data_name}/` from the pretraining step.

```bash
python BTTA_DG.py
```



## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
We thank the authors of the [MOABB](https://github.com/NeuroTechX/moabb) datasets for providing the data used in our experiments. Special thanks to the [DeepTransferEEG](https://github.com/sylyoung/DeepTransferEEG) and [SincNet](https://github.com/mravanelli/SincNet) of the open-source libraries that facilitated the development of this framework.
