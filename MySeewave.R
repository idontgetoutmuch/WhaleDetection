library(seewave)
library(tuneR)
library(sound)


fftFile <- function(fileName) {
    sndSample <- loadSample(fileName)
    sampFreq  <- rate(sndSample)
    nBits     <- bits(sndSample)
    ## extract numerical values from sample object
    snd       <- sound(sndSample)

    timeArray <- (0:(4000-1)) / sampFreq
    timeArra1 <- timeArray * 1000

    n <- length(snd)
    p <- fft(snd)

    nUniquePts <- ceiling((n + 1)/2)
    ## select just the first half since the second half
    ## is a mirror image of the first
    p <- p[1:nUniquePts]
    ## take the absolute value, or the magnitude
    p <- abs(p)
    ## scale by the number of points so that
    ## the magnitude does not depend on the length
    ## of the signal or on its sampling frequency
    p <- p / n
    ## square it to get the power
    p <- p^2

    ## multiply by two (see
    ## http://www.mathworks.co.uk/support/tech-notes/1700/1702.html
    ## for details) odd nfft excludes Nyquist point
    if (n %% 2 > 0){
        ## we've got odd number of points fft
        p[2:length(p)] <- p[2:length(p)] * 2
    } else {
        ## we've got even number of points fft
        p[2: (length(p) - 1)] <- p[2: (length(p) - 1)] * 2
    }

    ##  create the frequency array
    freqArray <- (0:(nUniquePts - 1)) * (sampFreq / n)

    return(list(snd, p, timeArray, freqArray))
}

fftData1 <- fftFile('data/train/train1.wav')

snd1       <- unlist(fftData1[1])
p1         <- unlist(fftData1[2])
timeArray1 <- unlist(fftData1[3])
freqArray1 <- unlist(fftData1[4])

fftData6 <- fftFile('data/train/train6.wav')

snd6       <- unlist(fftData6[1])
p6         <- unlist(fftData6[2])
timeArray6 <- unlist(fftData6[3])
freqArray6 <- unlist(fftData6[4])

jpeg("sample.jpg")

par(mfrow=c(2,2))

plot(timeArray1, snd1, type='l', ylim=range(c(snd1, snd6)), main='Non-whale time (1)', col='black', xlab='Time (ms)', ylab='Amplitude')

plot(freqArray1/1000, 10*log10(p1), type='l', ylim=range(c(10*log10(p1), 10*log10(p6))), main='Non-whale freq (1)', col='black', xlab='Frequency (kHz)', ylab='Power (dB)')

plot(timeArray6, snd6, type='l', ylim=range(c(snd1, snd6)), main='Whale time (6)', col='black', xlab='Time (ms)', ylab='Amplitude')

plot(freqArray6/1000, 10*log10(p6), type='l', ylim=range(c(10*log10(p1), 10*log10(p6))), main='Whale freq (6)', col='black', xlab='Frequency (kHz)', ylab='Power (dB)')

dev.off()
