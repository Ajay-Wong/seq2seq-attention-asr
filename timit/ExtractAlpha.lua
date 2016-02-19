require 'nn';
package.path = package.path .. ';../?.lua'
require 'Attention';
require 'LSTM';
require 'nngraph';
require 'RNN';
require 'GRU';
require 'LSTM';
require 'Maxout';
require 'hdf5';
require 'xlua';
require 'optim';
require 'cunn';
require 'timit/utils_timit';

cutorch.setDevice(1)

rundir = 'TIMIT/output/exp_logmel4_chorowski_baseline_adadelta_ortho/'
model = torch.load(paths.concat(rundir,'model.t7'))

autoencoder = model.autoencoder
encoder     = model.encoder
decoder     = model.decoder
optimConfig = model.optimConfig
optimState  = model.optimState
gradnoise   = model.gradnoise
--model.outputDepth = model.outputDepth or numPhonemes


opt = model.opt
opt.datafile = 'TIMIT/dynamic3/logmel123.h5'
print(opt)

------------------ Data ------------------
local file = hdf5.open(opt.datafile)
data = file:all()
file:close()

numPhonemes = 62

function processData(data)
    if type(data) == 'table' then
        if opt.batchSize == 1 then
            print('resetting opt.batchSize to 1 since data is a table, which is assumed to have variable length sequences')
            opt.batchSize = 1
        end
        dataset   = {}
        dataset.x = {}
        dataset.y = {}
		dataset.start = {}
		dataset.finish = {}
        i=0
        for k,f in pairs(data) do
            i=i+1
            dataset.x[i] = f.x:cuda()
            dataset.y[i] = f.y:cuda()
			dataset.start[i] = f.start
			dataset.finish[i] = f.finish
        end
        dataset.numSamples = i
        return dataset
    else
        data.numSamples = data.x:size(1)
        return data
    end
end

train = processData(data.train)
valid = processData(data.valid)
test  = processData(data.test)


function getAlpha(dataset,writeFile,datasetname)
	local start = sys.clock()
	local numsamples = 20--,#dataset.x
	local alpha = {}
	for i = 1, numsamples do
		xlua.progress(i,numsamples)
		local X = dataset.x[i]
		local Y = dataset.y[i]
		local T = Y:size(1)
		local labelmask = torch.zeros(T,model.outputDepth):cuda():scatter(2,Y:view(T,1),1)
		local logprobs  = autoencoder:forward({X,labelmask})
		alpha[i] = decoder:alpha():float()
	end
	return alpha, dataset.x, dataset.y
end

function toHD5format(dataset)
	local alpha, X, Y = getAlpha(dataset)
	local output = {}
	for i = 1, #alpha do
		local start = dataset.start[i]
		local finish = dataset.finish[i]
		local myoutput = {}
		myoutput['alpha'] = alpha[i]
		myoutput['start'] = start
		myoutput['finish'] = finish
		myoutput['X'] = X[i]:float()
		myoutput['Y'] = Y[i]:float()
		output[tostring(i)] = myoutput
	end
	return output
end

writeFile = hdf5.open(paths.concat(rundir,'alpha_sample.h5'),'w')
print('extract alpha on train')
tr = toHD5format(train)
writeFile:write('train',tr)
print('extract alpha on validation')
va = toHD5format(valid)
writeFile:write('valid',va)
writeFile:close()

