require 'optim';
require 'utils_librispeech';
package.path = package.path .. ';../?.lua'
TrainUtils = require 'TrainUtils'

-- setup experiment file structure
opt                 = {}
opt.device          = 1
opt.resume          = false
opt.datadir         = 'LibriSpeech/preprocessed/clean100_chunk2_samples3/CQT'
opt.savedir         = 'LibriSpeech/output/exp0_scriptcheck'

sys.execute('mkdir -p ' .. opt.savedir)
opt.trainfile       = 'train.lua'
opt.modelfile       = 'model_chorowski_baseline.lua'
opt.runfile         = string.sub(tostring(debug.getinfo(1,'S').source),2)
sys.execute('cp ' .. opt.trainfile .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.runfile   .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.modelfile .. ' ' .. opt.savedir)

-- train settings
opt.batchSize       = 1
opt.maxnorm         = 1e20
opt.weightDecay     = 0
opt.maxnumsamples   = 1e20

optimMethod         = optim.adadelta
optimConfig         = {eps=1e-8,rho=0.95}
--optimConfigResets   = {}
--optimConfigResets[13]= {eps=1e-10,rho=0.95}
gradnoise           = {eta=0,gamma=0.55,t=0}
opt.weightnoise     = 0
opt.init_std        = 0.01
opt.orthogonalize   = true

-- load filepaths and metadata
filepaths           = loadfilepath(opt.datadir)
meta                = loadmeta(opt.datadir)

-- load model
opt.inputFrameSize  = meta.inputFrameSize
opt.outputDepth     = meta.numchars
dofile(opt.modelfile)
model               = loadmodel(opt)
model.gradnoise     = gradnoise
model.optimConfig   = optimConfig

-- initialize model
model.autoencoder:reset(opt.init_std)
if opt.orthogonalize then
    TrainUtils.orthogonalizeGraph(model.autoencoder)
end

-- run
dofile(opt.trainfile)
