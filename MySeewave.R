library(seewave)
library(tuneR)
library(sound)

sndSample1 <- loadSample('data/train/train1.wav')
sampFreq1  <- rate(sndSample1)
nBits1     <- bits(sndSample1)
 ## extract numerical values from sample object
snd1       <- sound(sndSample1)

## FIXME: Cut'n'paste - the swiss army knife of software engineering
sndSample6 <- loadSample('data/train/train6.wav')
sampFreq6  <- rate(sndSample6)
nBits6     <- bits(sndSample6)
## extract numerical values from sample object
snd6       <- sound(sndSample6)

timeArray1 <- (0:(4000-1)) / sampFreq1
timeArray1 <- timeArray1 * 1000

n1 <- length(snd1)
p1 <- fft(snd1)

nUniquePts1 <- ceiling((n1 + 1)/2)
## select just the first half since the second half
## is a mirror image of the first
p1 <- p1[1:nUniquePts1]
## take the absolute value, or the magnitude
p1 <- abs(p1)
## scale by the number of points so that
## the magnitude does not depend on the length
## of the signal or on its sampling frequency
p1 <- p1 / n1
## square it to get the power
p1 <- p1^2

# multiply by two (see http://www.mathworks.co.uk/support/tech-notes/1700/1702.html for details)
# odd nfft excludes Nyquist point
if (n1 %% 2 > 0){
    ## we've got odd number of points fft
    p1[2:length(p1)] <- p1[2:length(p1)] * 2
} else {
    ## we've got even number of points fft
    p1[2: (length(p1) - 1)] <- p1[2: (length(p1) - 1)] * 2
}

##  create the frequency array
freqArray1 <- (0:(nUniquePts1 - 1)) * (sampFreq1 / n1)


## FIXME: More cut'n'paste yuk

timeArray6 <- (0:(4000-1)) / sampFreq6
timeArray6 <- timeArray6 * 1000

n6 <- length(snd6)
p6 <- fft(snd6)

nUniquePts6 <- ceiling((n6 + 1)/2)
## select just the first half since the second half
## is a mirror image of the first
p6 <- p6[1:nUniquePts6]
## take the absolute value, or the magnitude
p6 <- abs(p6)
## scale by the number of points so that
## the magnitude does not depend on the length
## of the signal or on its sampling frequency
p6 <- p6 / n6
## square it to get the power
p6 <- p6^2

# multiply by two (see http://www.mathworks.co.uk/support/tech-notes/1700/1702.html for details)
# odd nfft excludes Nyquist point
if (n6 %% 2 > 0){
    ## we've got odd number of points fft
    p6[2:length(p6)] <- p6[2:length(p6)] * 2
} else {
    ## we've got even number of points fft
    p6[2: (length(p6) - 1)] <- p6[2: (length(p6) - 1)] * 2
}

##  create the frequency array
freqArray6 <- (0:(nUniquePts6 - 1)) * (sampFreq6 / n6)

jpeg("sample.jpg")

par(mfrow=c(2,2))

plot(timeArray1, snd1, type='l', ylim=range(c(snd1, snd6)), main='Non-whale time (1)', col='black', xlab='Time (ms)', ylab='Amplitude')

plot(freqArray1/1000, 10*log10(p1), type='l', ylim=range(c(10*log10(p1), 10*log10(p6))), main='Non-whale freq (1)', col='black', xlab='Frequency (kHz)', ylab='Power (dB)')

plot(timeArray6, snd6, type='l', ylim=range(c(snd1, snd6)), main='Whale time (6)', col='black', xlab='Time (ms)', ylab='Amplitude')

plot(freqArray6/1000, 10*log10(p6), type='l', ylim=range(c(10*log10(p1), 10*log10(p6))), main='Whale freq (6)', col='black', xlab='Frequency (kHz)', ylab='Power (dB)')

dev.off()
