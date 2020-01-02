# CUDA Pitch Detector
A GPU accelerated autocorrelation pitch detector. This program compares the runtimes of pitch detection using the [normal method](https://gerrybeauregard.wordpress.com/2013/07/15/high-accuracy-monophonic-pitch-estimation-using-normalized-autocorrelation/) and the parallel method. This project provides an implementation for parallelizing autocorrection using a GPU.

## Dectecting the pitches of melody.wav
Before proceeding, you will need an Nvidia graphics card and the CUDA compiler.

While in the pitchGPU directory, run:
```
./pitch melody.wav
```
The output would then be the normal runtime (in milliseconds) followed by a series of frequencies (in Hertz), produced by the detection, of melody.wav:
```
Calculating pitches normally... Normal: 6965.701000 ms
Estimated frequencies of melody.wav by NORMAL:
196.865219,
197.027130,
196.821457,
196.731506,
196.936630,
196.663422,
98.346306,
98.202042,
...
```
Then the wav file's frequencies are detected using the GPU:
```
Calculating pitches using GPU... GPU: 5245.698000 ms
Estimated frequencies of melody.wav by GPU:
196.865219,
197.027130,
196.821457,
196.731506,
196.936630,
196.663422,
98.346306,
98.202042,
...
```
For this audio file, the frequencies were detected faster using the GPU. Try it out for yourself :)
>**Disclaimer**: Only `.wav` files with a sample rate of 44100 can have their pitch detected with this program. Any other file or sample rate may break the program. Error checking was kept to a minimal in the creation of this program.


## Research Paper
[Parallelizing Pitch Detection Autocorrelation Using GPU](https://drive.google.com/file/d/1rwtYk0nfTehvrJkJP1twFi0neStNdzZU/view?usp=sharing)
### Contributers
Kevin Louis-Jean (kloui032@fiu.edu)  
Jessela Baniqued (jelbaniqued26@gmail.com)  
Christian Agosto (cagos003@fiu.edu)  

