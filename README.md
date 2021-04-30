# MelonSignature
Melonplaylist MelScale signature.ipynb:
* Finds exact mel spectrogram signature from MelonPlaylist mel files.

Interpolation.ipynb:
* Measures the impact of using MelonPlaylist melScale signature when computing Essentia-tensorflow embeddings instead of waveforms (which is unavailable due to copyright) in a genre classification task on GTZAN.


    | Model        | Loss           | GTZAN accuracy  |
    | ------------- |:-------------:| -----:|
    | Random embeddings      | 2.31 | 9.37% |
    | Random musiCNN      | 1.88 | 40,58% |
    | musiCNN waveform      |    0.67   |   80.50% |
    | musiCNN linear interpolation | 0.92      |    74.57% |
    | <strong>musiCNN nearest interpolation</strong> | <strong>0.87    </strong>  |  <strong>  77.28% </strong>|
    | musiCNN bilinear interpolation | 1.09     |    69.30% |
    | musiCNN bicubic interpolation |   0.94   |    72.83% |
    | musiCNN bicubic aligned_corners |   0.95   |    74.23% |
    | musiCNN area interpolation |      0.97 |    75.04% |
    | musiCNN trilinear interpolation |  1.04     |    70.40% |
    | musiCNN MEL_to_audio librosa |  2.58     |    18.75% |
    | musiCNN MEL_to_STFT librosa |    1.30   |    63.41% |

Final methods:


```
import essentia.standard as es
import numpy as np
import torch

def adapt_melonInput_TensorflowPredictMusiCNN(melon_sample):
    """
    Adapts (by treating the spectrogram as an image and using Computer 
    Vision interpolation methods) the MelonPlaylist mel spectrograms to patches
    suitable for using the Essentia-Tensorflow TensorflowPredict algorithm.

    Input:
    melon_samples (frames, 48bands) dtype=np.float32
    mode: 'linear', 'nearest'  'bilinear', 'bicubic', , 'area', 'trilinear'
    Output:(batch, 187, 1, 96bands)
    """
    #First we de-normnalize (from dB to linear):
    db2amp = es.UnaryOperator(type='db2lin', scale=2)
    renormalized = np.zeros_like(melon_sample).astype(np.float32)
    for k in range(len(melon_sample)):
        renormalized[k,:] = np.log10(1 + (db2amp(melon_sample[k])*10000))
    #We add dimensions for convolution
    renormalized = torch.from_numpy(renormalized).unsqueeze(0).unsqueeze(0)
    #Oversample with pytorch   
    oversampled=torch.nn.functional.interpolate(input=renormalized, 
                                        size=[melon_sample.shape[0],melon_sample.shape[1]*2],
                                        mode='nearest').squeeze()
    oversampled = oversampled.numpy()
    # Now we cut again, but with hop size of 93 frames as in default TensorflowPredictMusiCNN
    new = np.zeros((int(len(oversampled) / 93) - 1, 187, 96)).astype(np.float32)
    for k in range(int(len(oversampled) / 93) - 1):
        new[k]=oversampled[k*93:k*93+187]
    return np.expand_dims(new, 2)
```

```
def melspectrogram(audio):
    """
    From a 16kH sample, computes the mel spectrogram with the same signature as done in MelonPlaylist dataset.
    
    Input:
    audio (samples) sampled at 16kHz and between [-1,1], dtype=np.float32
    Output:(frames, 48bands)
    """    
    windowing = es.Windowing(type='hann', normalized=False, zeroPadding=0)
    spectrum = es.Spectrum()
    melbands = es.MelBands(numberBands=48,
                                   sampleRate=16000,
                                   lowFrequencyBound=0,
                                   highFrequencyBound=16000/2,
                                   inputSize=(512+0)//2+1,
                                   weighting='linear',
                                   normalize='unit_tri',
                                   warpingFormula='slaneyMel',
                                   type='power')
    amp2db = es.UnaryOperator(type='lin2db', scale=2)
    result = []
    for frame in es.FrameGenerator(audio, frameSize=512, hopSize=256,
                                   startFromZero=False):
        spectrumFrame = spectrum(windowing(frame))

        melFrame = melbands(spectrumFrame)
        result.append(amp2db(melFrame))
    return np.array(result)
```
Now we can feed it to TensorFlowPredict:
```
#Simulate a MelonPlaylist arena_mel file:
loader = es.MonoLoader(filename='576120.mp3')
audio = loader()
melon_sample = melspectrogram(audio)
adapted_sample = adapt_melonInput_TensorflowPredictMusiCNN(melon_sample)

modelName='msd-musicnn-1.pb'
output_layer='model/dense/BiasAdd'
input_layer='model/Placeholder'
predict = es.TensorflowPredict(graphFilename=modelName,
                               inputs=[input_layer],
                               outputs=[output_layer])
in_pool = Pool()
in_pool.set('model/Placeholder', adapt_melonInput_TensorflowPredictMusiCNN(adapted_sample))
output = predict(in_pool)
embedding = output['model/dense/BiasAdd'][:,0,0,:]
```
Additional requirements: torch

Local VENV: MELON_VENV
