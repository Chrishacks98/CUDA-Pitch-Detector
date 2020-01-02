#include <iostream>
#include <cstring>
#include <math.h>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <emmintrin.h>

#define totalNumPeriods 1598
#define SR 44100
#define MINP 9
#define MAXP 1604

using namespace std;

typedef unsigned char uchar;

struct timeval before, after; 

void readWavBytes(string path, vector<char> *wavdata) {
    ifstream inFile;

    inFile.open(path, ifstream::binary);

    if (!inFile) {
        cout << "Can't find file!\n";
        exit(1);
    }

    // get length of file:
    inFile.seekg (0, inFile.end);
    int length = inFile.tellg();
    inFile.seekg (0, inFile.beg); 

    char *buffer = new char[length];

    cout << "Reading " << length << " bytes from wave file... ";

    // read data as a block:
    inFile.read (buffer,length);

    if (inFile) {
        cout << "all bytes read successfully.\n";
    } else {
        cout << "error: only " << inFile.gcount() << " could be read";
        inFile.close();
        exit(1);
    }

    inFile.close();

    //push buffer array into wavdata vector
    for (int i = 0; i < length; i++) {
        wavdata->push_back(buffer[i]);
    }
	
	delete[] buffer;
}


void audioRead(string path, vector<float> *buffer, int &sampleRate, int &numChannels) {
	//Get bytes from wave file
	vector<char> wavdata;
	readWavBytes(path, &wavdata);
	
	sampleRate = (int)((uchar)wavdata[24]) + (int)((uchar)wavdata[25]*256);
	
	numChannels = (int)wavdata[22];

	int floatSize = (((int)((uchar)wavdata[52]) + (int)((uchar)wavdata[53])*256 + 
		(int)((uchar)wavdata[54])*256*256 + (int)((uchar)wavdata[55])*256*256*256)/4) / numChannels;
	
	if (numChannels == 1) {
		//If audio is mono:
		for (int i = 0; i < floatSize; i++) {
			int block = (14+i)*4;
			uchar bytes[] = {(uchar)wavdata[block], (uchar)wavdata[block+1], 
                (uchar)wavdata[block+2], (uchar)wavdata[block+3]};
			float sample;
			memcpy(&sample, &bytes, sizeof(sample));
			buffer->push_back(sample);
		}

	} else {
		//Else, audio is stereo. Read left channel:
		for (int i = 0; i/2 < floatSize; i+=2) {
			int block = (14+i)*4;
			uchar bytes[] = {(uchar)wavdata[block], (uchar)wavdata[block+1], 
				(uchar)wavdata[block+2], (uchar)wavdata[block+3]};
			float sample;
			memcpy(&sample, &bytes, sizeof(sample));
			buffer->push_back(sample);
		}
	}              
} 


void starttime() {
  gettimeofday( &before, 0 );
}

void endtime(const char* c) {
   gettimeofday( &after, 0 );
   double elapsed = ( after.tv_sec - before.tv_sec ) * 1000.0 + ( after.tv_usec - before.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

/*
*   This funciton is built from Beauregard's EstimatePeriod()
*/
float detectFrequency(vector<float> *x, int start) {
	vector<float> nac (MAXP + 2);	// Normal Autocorrelation array
    int frameSize = MAXP*2;

	//Find the autocorrelation value for each period
	for (int p = MINP - 1; p <= MAXP + 1; p++) {
		float ac = 0.0;		// Standard auto-correlation
      	float sumSqBeg = 0.0;	// Sum of squares of beginning part
      	float sumSqEnd = 0.0;	// Sum of squares of ending part
        int audioShift = frameSize + start;

		for (int i = start; i < audioShift - p; i++) {
			ac += x->at(i) * x->at(i + p);
			sumSqBeg += x->at(i) * x->at(i);
			sumSqEnd += x->at(i + p) * x->at(i + p);
		}

		float sumSqrt = sqrt(sumSqBeg * sumSqEnd);
		if (sumSqrt == 0) {sumSqrt = 1;}
		nac[p] = ac / sumSqrt;
	}

	//  Get the highest value
	int bestP = MINP;
	for (int p = MINP; p <= MAXP; p++) {
		if (nac[p] > nac[bestP]) {
            bestP = p;
        }
	}

  	//  For accuracy, interpolate based on neighboring values
	float mid = nac[bestP];
  	float left = nac[bestP - 1];
  	float right = nac[bestP + 1];
	float div = 2 * mid - left - right;
	int error = 1;

	if (div == 0 || (nac[bestP] < nac[bestP - 1] && nac[bestP] < nac[bestP + 1])) {
		error = -1;
	}
	
	float shift = 0.5 * (right - left) / div;
	float pEst = error * (bestP + shift);

	//  Check for octave mutiple errors
	const float k_subMulThreshold = 0.90f;
	int maxMul = bestP / MINP;
	bool found = false;

	for (int mul = maxMul; !found && mul >= 1; mul--) {
		bool subsAllStrong = true;

		for (int k = 1; k < mul; k++) {
			int subMulP = int (k * pEst / mul + 0.5);

			if (nac[subMulP] < k_subMulThreshold * nac[bestP]) {
	    		subsAllStrong = false;
				break;
			}

		}

		if (subsAllStrong == true) {
			found = true;
			pEst = pEst / mul;
		}
	}

	float fEst = 0;
	if (pEst > 0) {
		fEst = SR / pEst;
	}
    
	//returning frequency
	return fEst;
}


/*
*   This method takes in audio x and detect the frequency for audio frame
*/
float *normalPitchDetection(vector<float> *x) {
    int audioSize = x->size();
    int fEstsSize = audioSize / (MAXP*2);   // The amount of audio frames that will be detected for frequencies
    float *fEsts;

    // Allocate Memory on CPU
    fEsts = (float *)malloc(fEstsSize * sizeof(float));

    // For each audio frame, detect the frequency and store it in fests array
    for (int i = 0; i < fEstsSize; i++) {
        int audioIdx = (MAXP*2) * i;
        fEsts[i] = detectFrequency(x, audioIdx);
    }

    // Return the array
    return fEsts;
}

/*
*   This is the GPU implementation of the normal pitch detection. 
*   Each block in the gpu will detect the pitch of one frame of audio
*       -frame of audio: a subsection (or "slice") of the audio. Its about MAXP*2 samples long    
*        (3,208 samples)
*   For example: if an audio is 250ms long, the amount is audio frames being detected
*   for pitch will be 250 / 70 (which is 3)
*/
__global__ void gpu_PitchDetection (float *gpu_audio, float *gpu_fEsts) {

    // Storing blockIdx.x and threadIdx.x in variables for temporal locality
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    // The size of the "slice" of audio being detected for pitch
    int frameSize = MAXP * 2;

    /*
        The Folling variables are for frequency detection calculations
        The process for detecting a frequency goes as follows:
            1.  Sounds that have a pitch mean that there are sections in the waveform 
                that are periodic. In order to discover the size of this period, there
                would have to be a one to one correlation a time t away. 

            2.  My range of periods to search for are from notes A0 to C8. Each period's
                correlation value are stored in an array called "nac" which stands for
                "normal autocorrelation". A period of a waveform is determined if the nac 
                value close ot 1 (about 0.99880)

            3.  This method will also have a one to correlation of a multiple of the period.
                So an additional part of the process will involve discovering which multiple
                of the period will produce the most fundamental "period"
    */

    // Each thread in a block will detect the correlation values for 4 periods
    __shared__ float nac[MAXP + 2];                 // Will store the normal autocorrelation values 

    int lowPThreshold = (tid * 4) + (MINP - 1);     // Since each thread will compute 4 periods, this varible is the smallest period that the thread will compute for
    int highPThreshold = lowPThreshold + 4;         // This is the maximum period that a thread will compute for


    /*
        These variables are for loading an audio frame into a gpu block. 
    */

    // Each thread will load 8 samples into thier shared memory block
    __shared__ float sharedAudioFrame[MAXP*2];  // The "slice" of audio that the gpu block will detect the pitch for
    int loadMin = tid * 8;                      // This is the minimum sample that will be loaded by a thread
    int loadMax = loadMin + 8;                  // The max sample that will load a thread
    int audioIdx = bid * frameSize;             // This varible holds the beginning index of the global audio to load the blocks frame



    // This block loads the audio frame into shared memory
    if (loadMin < frameSize) {  
        // Ensure that the threadId doesn't cause audio array to go out of bounds
        // Only threads 0 - 400 should run this section
        
        // Ensure loadMax doesn't index audio array out of bounds
        if (loadMax > frameSize) {
            loadMax = frameSize;
        }

        // Each thread will load 8 samples from the global audio
        for (int i = loadMin; i < loadMax; i++) {
            sharedAudioFrame[i] = gpu_audio[audioIdx + i];
        }
    }

    __syncthreads(); // Wait for all threads to finish loading their samples into shared memory

    // Each thread will find the correlation value of 4 periods
    if (lowPThreshold <= totalNumPeriods) {
        
        // If the highest period exceeds MAXP + 2, set highest period variable to MAXP + 2
        if (highPThreshold > MAXP + 2) {
            highPThreshold = MAXP + 2;
        }

        // This for loop is built from Beauregard's code for calculating the 
        // Normal autocorrelation value
        // Each thread will calculate 4 period values
        for (int p = lowPThreshold; p < highPThreshold; p++) {
                float ac = 0.0;		// Standard auto-correlation
                float sumSqBeg = 0.0;	// Sum of squares of beginning part
                float sumSqEnd = 0.0;	// Sum of squares of ending part

            for (int i = 0; i < frameSize - p; i++) {
                ac += sharedAudioFrame[i] * sharedAudioFrame[i + p];
                sumSqBeg += sharedAudioFrame[i] * sharedAudioFrame[i];
                sumSqEnd += sharedAudioFrame[i + p] * sharedAudioFrame[i + p];
            }

            float sumSqrt = sqrt(sumSqBeg * sumSqEnd);
            if (sumSqrt == 0) {sumSqrt = 1;}
            nac[p] = ac / sumSqrt;
        }
        
    }

    __syncthreads(); // Wait for threads to finish calculating nac values

    // The following block is built from  Beauregard's code
    // Use thread 0 to find the greatest value and store it in nac[0]
    if (tid == 0) {
        int bestP = MINP;
        for (int p = MINP; p <= MAXP; p++) {
            if (nac[p] > nac[bestP]) {bestP = p;}
        }
        nac[0] = (float)bestP;
    } 

    __syncthreads(); // Wait for the largest number nac value to be found   


    // This entire section until the next __syncthreads() was built heavily from Beauregard's code
    // This section finds the fundamental period be check each multiple's correlation value
    int bestP = nac[0];
    float mid = nac[bestP];
    float left = nac[bestP - 1];
    float right = nac[bestP + 1];
    float div = 2 * mid - left - right;

    // If error, terminate this block and return frequency 0
    if (div == 0 || (nac[bestP] < nac[bestP - 1] && nac[bestP] < nac[bestP + 1])) {
        if (tid == 0) {
            gpu_fEsts[bid] = 0;
        }
        return;
    }


    float shift = 0.5 * (right - left) / div;
    float pEst = (bestP + shift);
    float k_subMulThreshold = 0.90f;
    int maxMul = bestP / MINP;
    __shared__ bool subsAllStrong[(MAXP/MINP)+1];

    if (tid > 0 && tid <= maxMul) {
        int mul = tid;
        subsAllStrong[mul] = true;
        for (int k = 1; k < mul; k++) {
            int subMulP = int (k * pEst / mul + 0.5);
            if (nac[subMulP] < k_subMulThreshold * nac[bestP]) {
                subsAllStrong[mul] = false;
                break;
            }
        }
    }

    __syncthreads(); // Wait for the subs to be calculated

    // Thread 0 will scan the subsAllStrong array backward to find the strongest multiple
    if (tid == 0) {
        for (int mul = maxMul; mul >= 1; mul--) {
            if (subsAllStrong[mul] == true) {
                pEst = pEst / mul;
                break;
            }
        }

        float fEst = 0;
        if (pEst > 0) {
            fEst = SR / pEst;
        }
        
        gpu_fEsts[bid] = fEst;
    }

}

float *gpuPitchDetection (vector<float> *x) {
    const int threadsPerBlock = 416;
    int audioSize = x->size();
    int blockSize = audioSize / (MAXP*2);
    int gpu_audioSize = audioSize - (audioSize % (MAXP*2));

    float *gpu_audio;
    float *gpu_fEsts;
    float *h_fEsts;

    //Allocate memory on CPU
    h_fEsts = (float *)malloc(blockSize*sizeof(float));

    // Allocate memory on the GPU
    cudaMalloc(&gpu_audio, gpu_audioSize*sizeof(float)); 
    cudaMalloc(&gpu_fEsts, blockSize*sizeof(float));

    // Copy the audio into the gpu_audio
    cudaMemcpy(gpu_audio, x->data(), gpu_audioSize*sizeof(float), cudaMemcpyHostToDevice); 
    
    // Run kernal pitch detection on entire audio (samples)
    gpu_PitchDetection <<< blockSize, threadsPerBlock >>>(gpu_audio, gpu_fEsts);

    // Copy all frequnecy estimates back to host
    cudaMemcpy(h_fEsts, gpu_fEsts, blockSize*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&gpu_audio); // Free the memory on the GPU
    cudaFree(&gpu_fEsts); // Free the memory on the GPU

    return h_fEsts;
}


int main(int argc, char** argv) {
    if (argc == 1) {
        printf("Must input wave file name!\n");
        return 1;
    }

	const string path = argv[1];

	vector<float> audioData;
	int sampleRate;
	int numChannels;
	audioRead (path, &audioData, sampleRate, numChannels);
	
    // This block just shows how the #define values above were obtained: MINP, MAXP, and totalNumPeriods (which is MAXP - MINP)
	const double minF = 27.5;	//  Lowest pitch of interest (27.5 = A0, lowest note on piano.)
	const double maxF = 4186.0;	//  Highest pitch of interest (4186 = C8, highest note on piano.)
	const int minP = int (sampleRate / maxF - 1);	//  Minimum period
	const int maxP = int (sampleRate / minF + 1);	//  Maximum period
	const int numOfSamples = 2 * maxP;	//  Number of samples.  For best results, should be at least 2 x maxP

	if (audioData.size() < numOfSamples) {
		printf("Audio too small!\n");
		return 1;
	}

    if (sampleRate != SR) {
        printf("Sample rate must be exacly 44100!\n");
        return 1;
    }


    int n = audioData.size() / (MAXP*2); //Number of frequency estimations

    printf("Calculating pitches normally... ");
    float *pitchNormal;
    starttime();
    pitchNormal = normalPitchDetection(&audioData);
    endtime("Normal");
    printf("Estimated frequencies of %s by NORMAL:\n", path.c_str());
    for (int i = 0; i < n; i++) {
        printf("%f,\n", pitchNormal[i]);
    }
    printf("\n\n");


    printf("Calculating pitches using GPU... ");
    float *pitchGPU;
    starttime();
    pitchGPU = gpuPitchDetection(&audioData);
    endtime("GPU");
    printf("Estimated frequencies of %s by GPU:\n", path.c_str());
    for (int i = 0; i < n; i++) {
        printf("%f,\n", pitchGPU[i]);
    }
    printf("\n\n");


	return 0;
}