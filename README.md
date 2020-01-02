# CUDA Pitch Detector
A GPU accelerated autocorrelation pitch detector. This program compares the runtimes of the pitch detection using the normal method and the parallel method. The pitches, or frequencies (in Hertz), are then output to the console.

## Dectecting the pitches of melody.wav
Before proceeding, you will need an Nvidia graphics card and the CUDA compiler.

While in the pitchGPU directory, run:
```
./pitch melody.wav
```
The output would then be the runtime (in milliseconds) series of frequencies produced by the normal method

**Disclaimer**: Only `.wav` files with a sample rate of 44100 can have their pitch detected with this program.


## Research Paper
[Parallelizing Pitch Detection Autocorrelation Using GPU](https://drive.google.com/file/d/1rwtYk0nfTehvrJkJP1twFi0neStNdzZU/view?usp=sharing)
