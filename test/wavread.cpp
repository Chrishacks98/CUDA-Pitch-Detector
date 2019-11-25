#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>

using namespace std;

typedef unsigned char uchar;

void audioRead (string path, vector<float> *buffer, int *sampleRate, int *numChannels) {
	//Get bytes from wave file
	ifstream file(path);
	string wavdata { istreambuf_iterator<char>(file),
			istreambuf_iterator<char>() };
	
	*sampleRate = (int)((uchar)wavdata[24]) + (int)((uchar)wavdata[25]*256);
	
	*numChannels = (int)wavdata[22];

	int floatSize = ((int)((uchar)wavdata[52]) + (int)((uchar)wavdata[53])*256 + 
		(int)((uchar)wavdata[54])*256*256 + (int)((uchar)wavdata[55])*256*256*256)/4;
	
	if (*numChannels == 1) {
		//If audio is mono
		for (int i = 0; i < floatSize; i++) {
			int block = (14+i)*4;
			uchar bytes[] = {(uchar)wavdata[block], (uchar)wavdata[block+1], 
					(uchar)wavdata[block+2], (uchar)wavdata[block+3]};
			float sample;
			memcpy(&sample, &bytes, sizeof(sample));
			buffer->push_back(sample);
		}

	} else {
		//Else, audio is stereo
		//Default data from left channel
		
		for (int i = 0; i < floatSize; i+=2) {
		      	int block = (14+i)*4;
	             	uchar bytes[] = {(uchar)wavdata[block], (uchar)wavdata[block+1], 
                      			(uchar)wavdata[block+2], (uchar)wavdata[block+3]};
			
			float sample;
			memcpy(&sample, &bytes, sizeof(sample));
			buffer->push_back(sample);
		}
	}
	              
}                     
                      
int main() {    
	string path = "sample_stereo.wav";
	vector<float> audioData;
	int sampleRate;
	int numChannels;
	audioRead (path, &audioData, &sampleRate, &numChannels);
	
	for (float sample : audioData) {
		cout << sample << endl;
	}

	return 0;
}
