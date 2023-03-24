import os
import librosa #audio processing library
import numpy as np
import math
import json

DATASET_PATH = "dataset"
JSON_PATH = "genre.json"

SAMPLE_RATE = 22050 #customary value for music processing
DURATION = 240 #change to duration of your audio
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
def right_pad(array,num_missing_items):
    padded_array = np.pad(array,(0, num_missing_items),mode = "constant")
    return padded_array

def save_mfcc(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5):
    # building a dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vector_per_segment = math.ceil(num_samples_per_segment / hop_length) #round up these values

    # loop through all the genres in the dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)): # dirpath is the current folder path, dirnames is the names of all the sub folders and file names are all the file names
        # if we are at the dataset folder (not the subfolders for genres/ need to be in genre level)
        if dirpath is not dataset_path:
            # save the labels
            label = os.path.split(dirpath)[-1] #in the genre folder get the genre IE path=genre/blues => blues
            data["mapping"].append(label)
            print("\n processing: {}".format(label))
            # process the files for the specific genre
            for f in filenames:
                file_path = os.path.join(dirpath, f) #load audio file
                signal, samplerate = librosa.load(file_path, sr=SAMPLE_RATE)
                if len(signal) < num_samples_per_segment:
                    num_missing_samples = num_samples_per_segment - len(signal)
                    signal = right_pad(signal, num_missing_samples)
                    print("padding...")
                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s #s is current segment we are in
                    finish_sample = start_sample + num_samples_per_segment
                    mfcc = librosa.feature.mfcc(y = signal[start_sample:finish_sample]
                                                ,sr = samplerate
                                                ,n_fft = n_fft
                                                ,n_mfcc = n_mfcc
                                                ,hop_length = hop_length) #analysing a slice of the audio from range start sample to finish sample
                    mfcc = mfcc.T #Transpose it
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vector_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))

        with open(json_path, "w") as fp:
            json.dump(data,fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
