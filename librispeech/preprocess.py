import os
import librosa
import numpy as np
from scikits.audiolab import Sndfile, play
import collections
import re
import h5py
import pickle
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--root',help='root dir',dest='root',default='LibriSpeech')
parser.add_option('--save',help='save directory',dest='save',default='LibriSpeech/preprocessed/')
parser.add_option('--train',help='train directory',dest='train-clean-100',default='LibriSpeech/preprocessed/')
(options, args) = parser.parse_args()

rootdir         = options.root
savedir         = options.save
traindir        = os.path.join(rootdir,'train-clean-100')
validdir        = os.path.join(rootdir,'dev-clean')
testdir         = os.path.join(rootdir,'test-clean')

print '\n'
print 'rootdir     = %s' % rootdir
print 'savedir     = %s' % savedir
print '\n'

os.system('mkdir -p %s' % savedir)

#------------------- organize files and process transcriptions --------------------
def loadTxtFile(filepath):
    with open(filepath,'r') as f:
        lines = [l.split(' ') for l in f.read().split('\n')]
    while len(lines[-1]) <= 1:
        #print 'removing',lines[-1]
        lines = lines[:-1]
    lines = {l[0]:{'transcription':' '.join(l[1:]),'txtfile':filepath} for l in lines}
    return lines

def organizeFiles(rootdir):
    lines = {}
    audio = {}
    unknown = {}
    for dirname, subdirlist, filelist in os.walk(rootdir):
        for f in filelist:
            filepath = os.path.join(dirname,f)
            if len(f) >= 4:
                if f[-4:] == '.txt':
                    lines.update(loadTxtFile(filepath))
                elif f[-5:] == '.flac':
                    audio[f.replace('.flac','')] = {'audiofile':filepath}
                else:
                    unknown[f] = {'filepath':filepath}
    files = lines
    for k,v in audio.iteritems():
        files[k]['audiofile'] = v['audiofile']
    
    return files

def getCharMap(files):
    # files should be a list, e.g. [trainfiles, validfiles, testfiles]
    charlist = collections.Counter()
    wordlist = collections.Counter()
    for lines in files:
        for key,data in lines.iteritems():
            transcription = data['transcription']
            for c in transcription:
                charlist[c] += 1
            for w in transcription.split():
                wordlist[w] += 1
    # 1-based indexing for torch
    charmap = {k:i+1 for i,k in enumerate(sorted(charlist.keys()))}
    wordmap = {k:i+1 for i,k in enumerate(sorted(wordlist.keys()))}
    
    charEOS = len(charmap)+1
    wordEOS = len(wordmap)+1
    
    charmap['<eos>'] = charEOS
    wordmap['<eos>'] = wordEOS
    
    return charmap, wordmap

def processTranscriptions(lines,charmap,wordmap):
    charEOS = charmap['<eos>']
    wordEOS = wordmap['<eos>']
    
    for key,data in lines.iteritems():
        transcription = data['transcription']
        data['chars'] = np.array([charmap[c] for c in transcription]+[charEOS])
        data['words'] = np.array([wordmap[w] for w in transcription.split()]+[wordEOS])

#------------------- preprocess audio --------------------
def logmel(filename,n_fft=2048,hop_length=512,nfreqs=None):
    f = Sndfile(filename, 'r')
    data = f.read_frames(f.nframes)
    melspectrogram = librosa.feature.melspectrogram(y=data, sr=f.samplerate, n_fft=n_fft, hop_length=hop_length)
    logmel = librosa.core.logamplitude(melspectrogram)
    if nfreqs != None:
        logmel = logmel[:nfreqs,:]
    energy = librosa.feature.rmse(y=data)
    spectr = np.vstack((logmel,energy))
    delta1 = librosa.feature.delta(spectr,order=1)
    delta2 = librosa.feature.delta(spectr,order=2)

    features = np.vstack((spectr,delta1,delta2))
    return features.T

def CQT(filename, fmin=None, n_bins=84, hop_length=512,nfreqs=None):
    f = Sndfile(filename, 'r')
    data = f.read_frames(f.nframes)
    cqt = librosa.cqt(data, sr=f.samplerate, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    if nfreqs != None:
        cqt = cqt[:nfreqs,:]
    delta1 = librosa.feature.delta(cqt,order=1)
    delta2 = librosa.feature.delta(cqt,order=2)
    energy = librosa.feature.rmse(y=data)
    features = np.vstack((cqt,delta1,delta2,energy))
    return features.T

def getFeatures(files,func=logmel,**kwargs):
    for k,f in files.iteritems():
        filepath = f['audiofile']
        f['features'] = func(filepath,**kwargs)

def normalizeFeatures(train,valid,test,pad=1):
    maxlength = 0
    featurelist = []
    for k,f in train.iteritems():
        maxlength = max(maxlength,len(f['features']))
        featurelist.append(f['features'])
    featurelist = np.vstack(featurelist)
    mean = featurelist.mean(axis=0)
    std = featurelist.std(axis=0)

    def normalize_and_pad(files):
        for k,f in files.iteritems():
            mylen = len(f['features'])
            padding = np.zeros((pad,f['features'].shape[1]))
            f['features'] = (f['features']-mean)/std
            f['features'] = np.vstack([padding,f['features'],padding])

    normalize_and_pad(train)
    normalize_and_pad(valid)
    normalize_and_pad(test)

    return mean, std

def pickleIt(X,outputName):
    with open(outputName,'wb') as f:
        pickle.dump(X,f)
        
def toHDF5(allfiles,filename):
    with h5py.File(filename,'w') as h:
        for g,files in allfiles.iteritems():
            grp = h.create_group(g)
            template = files[files.keys()[0]]

            for k,f in enumerate(files.values()):
                mygrp           = grp.create_group(str(k))
                mygrp['x']      = f['features']
                mygrp['chars']  = np.array(f['chars'])
                mygrp['words']  = np.array(f['words'])


#----------------- run -----------------------
print 'organize files and generate maps'
trainfiles = organizeFiles(traindir)
validfiles = organizeFiles(validdir)
testfiles  = organizeFiles(testdir)
charmap, wordmap = getCharMap([trainfiles,validfiles,testfiles])

print 'save charmap and wordmap'
with open(os.path.join(savedir,'charmap.txt'),'w') as f:
    for k,v in sorted(list(charmap.iteritems()),key=lambda x:x[1]):
        f.write('%s %s\n' % (k,v))

with open(os.path.join(savedir,'wordmap.txt'),'w') as f:
    for k,v in sorted(list(wordmap.iteritems()),key=lambda x:x[1]):
        f.write('%s %s\n' % (k,v))


print 'process transcriptions'
processTranscriptions(trainfiles,charmap,wordmap)
processTranscriptions(validfiles,charmap,wordmap)
processTranscriptions(testfiles,charmap,wordmap)

# logmel
print 'generate logmel features'
getFeatures(trainfiles,func=logmel,nfreqs=40)
getFeatures(validfiles,func=logmel,nfreqs=40)
getFeatures(testfiles,func=logmel,nfreqs=40)

print 'normalize logmel features'
logmel_mean, logmel_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=1)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

print 'save to disk'
toHDF5(allfiles,os.path.join(savedir,'logmel.h5'))
pickleIt([logmel_mean, logmel_std],os.path.join(savedir,'logmel_mean_std.pkl'))

# CQT
print 'generate CQT features'
getFeatures(trainfiles,func=CQT)
getFeatures(validfiles,func=CQT)
getFeatures(testfiles,func=CQT)

print 'normalize CQT features'
CQT_mean, CQT_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=1)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

print 'save to disk'
toHDF5(allfiles,os.path.join(savedir,'CQT.h5'))
pickleIt([CQT_mean, CQT_std],os.path.join(savedir,'CQT_mean_std.pkl'))
