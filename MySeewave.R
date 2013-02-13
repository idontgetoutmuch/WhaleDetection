library(seewave)

s1<-sin(2*pi*440*seq(0,1,length.out=8000))

library(tuneR)

s6<-readWave("data/train/train6.wav")

library(sound)

sndSample <- loadSample('data/train/train6.wav')
sampFreq <- rate(sndSample)
nBits <- bits(sndSample)
snd <- sound(sndSample) # extract numerical values from sample object

timeArray <- (0:(4000-1)) / sampFreq
timeArray <- timeArray * 1000

n <- length(snd)
p <- fft(snd)

nUniquePts <- ceiling((n+1)/2)
p <- p[1:nUniquePts] #select just the first half since the second half
                     # is a mirror image of the first
p <- abs(p)  #take the absolute value, or the magnitude

P <- p / n # scale by the number of points so that
           # the magnitude does not depend on the length
           # of the signal or on its sampling frequency
p <- p^2  # square it to get the power

# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
if (n %% 2 > 0){
    p[2:length(p)] <- p[2:length(p)]*2 # we've got odd number of points fft
} else {
    p[2: (length(p) -1)] <- p[2: (length(p) -1)]*2 # we've got even number of points fft
}

freqArray <- (0:(nUniquePts-1)) * (sampFreq / n) #  create the frequency array

plot(freqArray/1000, 10*log10(p), type='l', col='black', xlab='Frequency (kHz)', ylab='Power (dB)')
