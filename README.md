# MelonSignature
Melonplaylist MelScale signature.ipynb:
* Finds exact mel spectrogram signature from MelonPlaylist mel files.

Interpolation.ipynb:
* Measures the impact of using MelonPlaylist melScale signature when computing Essentia-tensorflow embeddings instead of audio (which is unavailable due to copyright) in a genre classification task on GTZAN.
| Model        | Loss           | Accuracy  |
| ------------- |:-------------:| -----:|
| Random embeddings      | 2.31 | 8.07% |
| Random musiCNN      | 1.88 | 40,58% |
| musiCNN waveform      |    0.67   |   80.50% |
| musiCNN melonMEL | 0.92      |    74.57% |
VENV: MELON_VENV
