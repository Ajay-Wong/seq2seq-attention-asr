require 'optim';
package.path = package.path .. ';../?.lua'
TrainUtils = require 'TrainUtils'
require 'cunn'

-- setup experiment file structure
opt                 = {}
opt.device          = 1
opt.resume          = false
opt.datadir         = 'TIMIT/dynamic/'
opt.datafile        = paths.concat(opt.datadir,'logmel123.h5')
opt.savedir         = 'TIMIT/output/exp_logmel7_baseline_normNLL_colnorm'
opt.phonemes        = 'phonemes.txt'

sys.execute('mkdir -p ' .. opt.savedir)
opt.trainfile       = 'timit.lua'
opt.modelfile       = 'model_chorowski_baseline.lua'
opt.runfile         = string.sub(tostring(debug.getinfo(1,'S').source),2)
sys.execute('cp ' .. opt.trainfile .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.runfile   .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.modelfile .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.phonemes  .. ' ' .. opt.savedir)

-- train settings
opt.batchSize       = 1
opt.maxnorm         = 1e20
opt.normalizeNLL    = true
opt.weightDecay     = 0
opt.maxnumsamples   = 1e20
opt.colnormconstr   = true

optimMethod         = optim.adadelta
optimConfig         = {eps=1e-8,rho=0.95}
optimConfigResets   = {}
--optimConfigResets[13]= {eps=1e-10,rho=0.95}
gradnoise           = {eta=0,gamma=0.55,t=0}
opt.weightnoise     = 0
opt.init_std        = 0.01
opt.orthogonalize   = true
--opt.dropout         = 0.5

-- load model
opt.numPhonemes     = 62
dofile(opt.modelfile)
model               = loadmodel(opt) 
model.gradnoise     = gradnoise
model.optimConfig   = optimConfig

-- run
dofile(opt.trainfile)
