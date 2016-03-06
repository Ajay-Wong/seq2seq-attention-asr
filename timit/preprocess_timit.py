import librosa
import numpy as np
from scikits.audiolab import Sndfile, play
import os
import h5py
import pickle
from optparse import OptionParser
import sklearn.decomposition

parser = OptionParser()
parser.add_option('--root',help='root TIMIT dir',dest='root',default='TIMIT')
parser.add_option('--save',help='save directory',dest='save',default='TIMIT')
parser.add_option('--valid',help='validation speakerIDs',dest='valid',default='valid_spkrid.txt')
parser.add_option('--samelenfeat',help='if True, all features will have same length',dest='samelenfeat',default=False)
parser.add_option('--samelenlabs',help='if True, all labels will have same length',dest='samelenlabs',default=False)
(options, args) = parser.parse_args()

featuresSameLength = options.samelenfeat
phonemesSameLength = options.samelenlabs
rootdir            = options.root
savedir            = options.save
rootdirtrain       = os.path.join(rootdir,'TRAIN')
rootdirtest        = os.path.join(rootdir,'TEST')

print '\n'
print 'featuresSameLength = %s' % featuresSameLength
print 'phonemesSameLength = %s' % phonemesSameLength
print 'rootdir            = %s' % rootdir
print 'savedir            = %s' % savedir
print 'validIDs           = %s' % options.valid
print '\n'

os.system('mkdir -p %s' % savedir)

#------------------- Load Files --------------------
print 'loading files'
def getFiles(rootdir,printFiles=False):
    files = {}
    ignored = 0
    for root, dirnames, filenames in os.walk(rootdir):
        for f in filenames:
            myname = '/'.join(root.split('/')[-2:] + [f])
            myid,filetype = myname.split('.')
            if filetype in ['PHN','TXT','WAV','WRD']:
                dr,spkr,st = myid.split('/')
                if st[:2] != 'SA':
                    if not myid in files.keys():
                        files[myid] = {}
                        files[myid]['dr'] = dr
                        files[myid]['spkr'] = spkr
                        files[myid]['sent'] = st
                        files[myid]['root'] = rootdir
                    files[myid][filetype] = myname
                    if printFiles:
                        print myname
                else:
                    ignored += 1
    print 'ignored %s files in %s' % (ignored,rootdir)
    return files


alltrainfiles = getFiles(rootdirtrain)
testfiles = getFiles(rootdirtest)

#------------------- Load Validation SpeakerIDs --------------------
print 'load validation speakerIDs'
vaids = np.loadtxt(options.valid,dtype=str)

trainfiles = {}
validfiles = {}
for k,v in alltrainfiles.iteritems():
    if v['spkr'] in vaids:
        validfiles[k] = v
    else:
        trainfiles[k] = v

### check there are only 50 spkrs in validfiles, 
### and check that there is no overlap between trainfiles and validfiles
vaspkrs = set([v['spkr'] for k,v in validfiles.iteritems()])
print "num valid speakers = %s" % len(vaspkrs)

trspkrs = set([v['spkr'] for k,v in trainfiles.iteritems()])
print 'len(vaspkrs.intersection(trspkrs)) = %s' % len(vaspkrs.intersection(trspkrs))

#------------------- Process Phonemes & Words --------------------
print 'parse phonemes & words'
def parseFile(x,filetype):
    with open(os.path.join(x['root'],x[filetype])) as f:
        lines = f.read().split('\n')
    return zip(*[l.split() for l in lines if len(l) > 0])

def parseAllFiles(files,filetype,keyname):
    for k,f in files.iteritems():
        start, finish, key = parseFile(f,filetype)
        f[keyname] = list(key)
        f['%s_start' % keyname] = start
        f['%s_finish' % keyname] = finish
        
def addEOStag(files):
    eos = '<EOS>'
    for k,f in files.iteritems():
        f['phonemes'].append(eos) 

def makeLabelsSameLength(train,valid,test,eos='<EOS>'):
    maxlength = 0
    for k,f in train.iteritems():
        maxlength = max(maxlength,len(f['phonemes']))
    
    for k,f in valid.iteritems():
        assert len(f['phonemes']) <= maxlength, 'uh-oh'
    for k,f in test.iteritems():
        assert len(f['phonemes']) <= maxlength, 'uh-oh'
        
    maxlength = maxlength
    for k,f in train.iteritems():
        mylen = len(f['phonemes'])
        f['phonemes'] = f['phonemes'] + [eos]*(maxlength-mylen)  
        f['label_flag'] = np.zeros(maxlength)  
        f['label_flag'][:mylen] = 1
    for k,f in valid.iteritems():
        mylen = len(f['phonemes'])
        f['phonemes'] = f['phonemes'] + [eos]*(maxlength-mylen) 
        f['label_flag'] = np.zeros(maxlength)  
        f['label_flag'][:mylen] = 1   
    for k,f in test.iteritems():
        mylen = len(f['phonemes'])
        f['phonemes'] = f['phonemes'] + [eos]*(maxlength-mylen)
        f['label_flag'] = np.zeros(maxlength)  
        f['label_flag'][:mylen] = 1
        
parseAllFiles(trainfiles,'PHN','phonemes')
parseAllFiles(validfiles,'PHN','phonemes')
parseAllFiles(testfiles,'PHN','phonemes')

parseAllFiles(trainfiles,'WRD','words')
parseAllFiles(validfiles,'WRD','words')
parseAllFiles(testfiles,'WRD','words')

addEOStag(trainfiles)
addEOStag(validfiles)
addEOStag(testfiles)

if phonemesSameLength:
    makeLabelsSameLength(trainfiles,validfiles,testfiles)
      
#------------------- generate phoneme vocab --------------------
print 'generate phoneme vocab'
phonemesTrain = set()
for k,f in trainfiles.iteritems():
    for p in f['phonemes']:
        phonemesTrain.add(p)
for k,f in validfiles.iteritems():
    for p in f['phonemes']:
        phonemesTrain.add(p)

phonemesTest = set()
for k,f in testfiles.iteritems():
    for p in f['phonemes']:
        phonemesTest.add(p)

phonemes = {p:i+1 for i,p in enumerate(phonemesTrain)}

with open('phones.60-48-39.map','r') as f:
    kaldi_phonemes = f.readlines()
kaldi_phonemes = [e.strip('\n').split('\t') for e in kaldi_phonemes]
kaldi_phonemes.append(['<EOS>']*3)

p60,p48,p39 = zip(*kaldi_phonemes)

vocab48 = {p:i for i,p in enumerate(set(p48))}
vocab39 = {p:i for i,p in enumerate(set(p39))}

map48 = {k[0]:{'index':vocab48[k[1]],'phoneme':k[1]} for k in kaldi_phonemes}
map39 = {k[0]:{'index':vocab39[k[2]],'phoneme':k[2]} for k in kaldi_phonemes}

#------------------- save phoneme dictionary --------------------
with open(os.path.join(savedir,'phonemes.txt'),'w') as f:
    f.write('index60,phoneme60,index48,phoneme48,index39,phoneme39\n')
    for k,v in sorted(list(phonemes.iteritems()),key = lambda x:x[1]):
        f.write('%s,%s,%s,%s,%s,%s\n' % (v,k,map48[k]['index']+1,map48[k]['phoneme'],map39[k]['index']+1,map39[k]['phoneme']))

#------------------- digitize phonemes --------------------
print 'digitize phonemes'
def digitizePhonemes(files):
    lengths = set()
    for k,f in files.iteritems():
        f['phonemeLabels'] = [phonemes[p] for p in f['phonemes']]
        f['phonemeLabels39'] = [map39[p]['index'] for p in f['phonemes']]
        lengths.add(len(f['phonemeLabels']))

digitizePhonemes(trainfiles)
digitizePhonemes(validfiles)
digitizePhonemes(testfiles)

#------------------- generate features --------------------
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

def logmel_stacked(filename,n_fft=2048,hop_length=512,nfreqs=None):
    f = Sndfile(filename, 'r')
    data = f.read_frames(f.nframes)
    melspectrogram = librosa.feature.melspectrogram(y=data, sr=f.samplerate, n_fft=n_fft, hop_length=hop_length)
    logmel = librosa.core.logamplitude(melspectrogram)
    if nfreqs != None:
        logmel = logmel[:nfreqs,:]
    delta1 = librosa.feature.delta(logmel,order=1)
    delta2 = librosa.feature.delta(logmel,order=2)
    d,L    = logmel.shape
    logmel = logmel.T.reshape(1,L,d)
    delta1 = delta1.T.reshape(1,L,d)
    delta2 = delta2.T.reshape(1,L,d)
    features = np.vstack((logmel,delta1,delta2))
    return features


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

def CQT_stacked(filename, fmin=None, n_bins=84, hop_length=512,nfreqs=None):
    f = Sndfile(filename, 'r')
    data = f.read_frames(f.nframes)
    cqt = librosa.cqt(data, sr=f.samplerate, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    if nfreqs != None:
        cqt = cqt[:nfreqs,:]
    delta1 = librosa.feature.delta(cqt,order=1)
    delta2 = librosa.feature.delta(cqt,order=2)
    d,L    = cqt.shape
    cqt = cqt.T.reshape(1,L,d)
    delta1 = delta1.T.reshape(1,L,d)
    delta2 = delta2.T.reshape(1,L,d)
    features = np.vstack((cqt,delta1,delta2))
    return features


def getFeatures(files,func=logmel,**kwargs):
    for k,f in files.iteritems():
        filename = os.path.join(f['root'],f['WAV'])
        f['features'] = func(filename,**kwargs)
        
def normalizeFeatures(train,valid,test,pad=10,use_samelength=False):
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
            if use_samelength:
                if mylen < maxlength:
                    extra_padding = np.zeros((maxlength-mylen,f['features'].shape[1]))
                    f['features'] = np.vstack([f['features'],extra_padding])
    
    normalize_and_pad(train)
    normalize_and_pad(valid)
    normalize_and_pad(test)
    
    return mean, std 

def normalizeStackedFeatures(train,valid,test,pad=10,use_samelength=True):
    maxlength = 0
    featurelist = []
    for k,f in train.iteritems():
        maxlength = max(maxlength,len(f['features']))
        featurelist.append(f['features'])
    featurelist = np.concatenate(featurelist,axis=1)
    a,b,c = featurelist.shape
    mean = featurelist.mean(axis=1).reshape(a,1,c)
    std = featurelist.std(axis=1).reshape(a,1,c)
    
    def normalize_and_pad(files):
        for k,f in files.iteritems():
            mylen = len(f['features'])
            padding = np.zeros((3,pad,f['features'].shape[2]))
            f['features'] = (f['features']-mean)/std
            f['features'] = np.concatenate([padding,f['features'],padding],axis=1)
            if use_samelength:
                if mylen < maxlength:
                    extra_padding = np.zeros((maxlength-mylen,f['features'].shape[1]))
                    f['features'] = np.concatenate([f['features'],extra_padding],axis=1)
    
    normalize_and_pad(train)
    normalize_and_pad(valid)
    normalize_and_pad(test)
    
    return mean, std 

def PCA(train,valid,test):
    featurelist = []
    for k,f in train.iteritems():
        featurelist.append(f['features'])
    featurelist = np.vstack(featurelist)
    pca = sklearn.decomposition.PCA()
    pca.fit(featurelist)

    for k,f in train.iteritems():
        f['features'] = pca.transform(f['features'])
    
    for k,f in valid.iteritems():
        f['features'] = pca.transform(f['features'])
    
    for k,f in test.iteritems():
        f['features'] = pca.transform(f['features'])

    return pca

#------------------- pickle --------------------
def pickleIt(X,outputName):
    with open(outputName,'wb') as f:
        pickle.dump(X,f)

#------------------- To HDF5 --------------------
def toHDF5(allfiles,filename):
    with h5py.File(filename,'w') as h:
        for g,files in allfiles.iteritems():
            grp = h.create_group(g)
            template = files[files.keys()[0]]
            
            if featuresSameLength and phonemesSameLength:
                sizes = list(template['features'].shape)
                sizes = [1] + sizes
                features = [f['features'].reshape(sizes) for f in files.values()]
                labels = [np.array(f['phonemeLabels']).reshape(1,-1) for f in files.values()]
                label_flags = [np.array(f['label_flag']).reshape(1,-1) for f in files.values()]
                grp['x'] = np.vstack(features)
                grp['y'] = np.vstack(labels)
                grp['ymask'] = np.vstack(label_flags)
            else:
                for k,f in enumerate(files.values()):
                    mygrp      = grp.create_group(str(k))
                    mygrp['x'] = f['features']
                    mygrp['y'] = np.array(f['phonemeLabels'])
                    mygrp['y39'] = np.array(f['phonemeLabels39'])
                    mygrp['start'] = np.array(f['phonemes_start']).astype('int')
                    mygrp['finish'] = np.array(f['phonemes_finish']).astype('int')

#------------------- logmel features --------------------
print 'generating logmel features'
getFeatures(trainfiles,func=logmel)
getFeatures(validfiles,func=logmel)
getFeatures(testfiles,func=logmel)
logmel_mean, logmel_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=10,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'logmel.h5'))
pickleIt([logmel_mean, logmel_std],os.path.join(savedir,'logmel_mean_std.pkl'))

#------------------- logmel123 features --------------------
print 'generating logmel features'
getFeatures(trainfiles,func=logmel,nfreqs=40)
getFeatures(validfiles,func=logmel,nfreqs=40)
getFeatures(testfiles,func=logmel,nfreqs=40)
logmel_mean, logmel_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=10,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'logmel123.h5'))
pickleIt([logmel_mean, logmel_std],os.path.join(savedir,'logmel123_mean_std.pkl'))

#------------------- logmel-stacked features --------------------
print 'generating logmel-stacked features'
getFeatures(trainfiles,func=logmel_stacked)
getFeatures(validfiles,func=logmel_stacked)
getFeatures(testfiles,func=logmel_stacked)
logmel_mean, logmel_std = normalizeStackedFeatures(trainfiles,validfiles,testfiles,pad=10,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'logmel_stacked.h5'))
pickleIt([logmel_mean, logmel_std],os.path.join(savedir,'logmel_stacked_mean_std.pkl'))

#------------------- logmel-pca features --------------------
print 'generating logmel-pca features'
getFeatures(trainfiles,func=logmel)
getFeatures(validfiles,func=logmel)
getFeatures(testfiles,func=logmel)
logmel_pca = PCA(trainfiles,validfiles,testfiles)
logmel_pca_mean, logmel_pca_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=10,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'logmel_pca.h5'))
pickleIt([logmel_pca, logmel_pca_mean, logmel_pca_std],os.path.join(savedir,'logmel_pca_mean_std.pkl'))

#------------------- CQT features --------------------
print 'generating cqt features'
getFeatures(trainfiles,func=CQT)
getFeatures(validfiles,func=CQT)
getFeatures(testfiles,func=CQT)
cqt_mean, cqt_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=10,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'cqt.h5'))
pickleIt([cqt_mean, cqt_std],os.path.join(savedir,'cqt_mean_std.pkl'))

#------------------- CQT-stacked features --------------------
print 'generating cqt-stacked features'
getFeatures(trainfiles,func=CQT_stacked)
getFeatures(validfiles,func=CQT_stacked)
getFeatures(testfiles,func=CQT_stacked)
cqt_mean, cqt_std = normalizeStackedFeatures(trainfiles,validfiles,testfiles,pad=10,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'cqt_stacked.h5'))
pickleIt([cqt_mean, cqt_std],os.path.join(savedir,'cqt_stacked_mean_std.pkl'))

#------------------- CQT-pca features --------------------
print 'generating CQT-pca features'
getFeatures(trainfiles,func=CQT)
getFeatures(validfiles,func=CQT)
getFeatures(testfiles,func=CQT)
cqt_pca = PCA(trainfiles,validfiles,testfiles)
cqt_pca_mean, cqt_pca_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=10,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'cqt_pca.h5'))
pickleIt([cqt_pca, cqt_pca_mean, cqt_pca_std],os.path.join(savedir,'cqt_pca_mean_std.pkl'))


