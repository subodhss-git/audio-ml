

import librosa
import librosa.display
import numpy

class PyTrial:
    def __init__(self):
        print('init PyTrial... ')



    def plotWavSpectra(self,wavFile):
        y, sr = librosa.core.load(path=wavFile, sr=None)

        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=numpy.max)

        import matplotlib.pyplot as plt

        # Make a new figure
        plt.figure(figsize=(12, 4))

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram ' + str(wavFile))

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()

    def energyAnalysis(self,wavFile,cutOffBin=512):
        y, sr = librosa.core.load(path=wavFile, sr=None)
        mono_y = librosa.to_mono(y=y)
        stftCoeff = librosa.core.stft(y=mono_y)
        stftShape = numpy.shape(stftCoeff)
        for cols in range(0,stftShape[1]):
            v = stftCoeff[:,cols]
            vn = numpy.linalg.norm(x=v)
            v1 = stftCoeff[:cutOffBin, cols]
            v1n = numpy.linalg.norm(x=v1)
            print(vn,v1n,v1n*100/vn)



    def supressBins(self,wavFile,cutOffBins=[512]):
        y, sr = librosa.core.load(path=wavFile, sr=None)
        mono_y = librosa.to_mono(y=y)
        print('running stft...')
        stftCoeff = librosa.core.stft(y=mono_y)
        stftShape = numpy.shape(stftCoeff)

        path = '/Users/subodhss/Downloads/song_masked_mono.wav'
        print('writing mono file...'+path)
        librosa.output.write_wav(path, mono_y, sr, norm=True)

        for cutOffBin in cutOffBins:
            masks = numpy.full(shape=stftShape, fill_value=1.0, dtype=numpy.float32)
            masks[cutOffBin:,:] = 0
            print('masking stft coeffs...cutOff='+str(cutOffBin))
            maskedCoeffs = stftCoeff * masks
            print('running inverse stft...')
            y = librosa.core.istft(maskedCoeffs)
            path = '/Users/subodhss/Downloads/song_masked_'+str(cutOffBin)+'.wav'
            print('writing file...',path)
            librosa.output.write_wav(path, y, sr, norm=True)
            print('file written...')

            masks = numpy.full(shape=stftShape, fill_value=0, dtype=numpy.float32)
            masks[cutOffBin:, :] = 1
            maskedCoeffs = stftCoeff * masks
            print('running inverse stft...')
            y = librosa.core.istft(maskedCoeffs)
            path = '/Users/subodhss/Downloads/song_inv_masked_' + str(cutOffBin) + '.wav'
            print('writing file...', path)
            librosa.output.write_wav(path, y, sr, norm=True)
            print('file written...')



        print('all done...')



if __name__ == '__main__':

    p = PyTrial()
    cutOffs = [512,256,128,64,32,16]
    p.supressBins(wavFile='/Users/subodhss/Downloads/song.wav',cutOffBins=cutOffs)
    # p.energyAnalysis(wavFile='/Users/subodhss/Downloads/song.wav',cutOffBin=256)
    # p.plotWavSpectra(wavFile='/Users/subodhss/Downloads/song.wav')