require 'nn';
package.path = package.path .. ';../?.lua'
require 'Attention';
require 'LSTM';
require 'nngraph';
require 'RNN';
require 'hdf5';
require 'xlua';
require 'optim';
require 'cunn';
require 'utils';
require 'utils_librispeech';
TrainUtils = require 'TrainUtils'

opt                 = opt                or {}
opt.device          = opt.device         or 1
opt.datadir         = opt.datadir        or 'LibriSpeech/preprocessed/clean100_chunked40'
opt.savedir         = opt.savedir        or 'LibriSpeech/output/CQT'
opt.batchSize       = opt.batchSize      or 1
opt.numEpochs       = opt.numEpochs      or 100
opt.maxnorm         = opt.maxnorm        or 1
opt.normalizeNLL    = opt.normalizeNLL   or false
opt.weightDecay     = opt.weightDecay    or 1e-4
opt.colnormconstr   = opt.colnormconstr  or false -- column norm constraint
opt.weightnoise     = opt.weightnoise    or 0
opt.K               = opt.K              or 5 -- beam search width
opt.modelfile       = opt.modelfile      or 'model.lua'
opt.labelset        = opt.labelset       or 'chars'
print(opt)

cutorch.setDevice(opt.device)

------------------ Data ------------------

filepaths = filepaths or loadfilepaths(opt.datadir)
meta      = meta or loadmeta(opt.datadir)

------------------ Model ------------------
if model then
	print('loading model')
	model.opt = opt
else
	dofile(opt.modefile)
	model = loadmodel(opt)
	opt = model.opt
end

autoencoder = model.autoencoder
encoder     = model.encoder
decoder     = model.decoder
optimConfig = model.optimConfig
optimState  = model.optimState
gradnoise   = model.gradnoise

autoencoder           = autoencoder:cuda()
parameters, gradients = autoencoder:getParameters()

if initialization then
	initialization(parameters)
end

------------------ Train ------------------
optimMethod = optimMethod or optim.adadelta
optimConfig = optimConfig or {
	eps = 1e-8,
	rho = 0.95
}
optimState  = optimState or nil
gradnoise   = gradnoise or {
	eta   = 0,
	gamma = 0.55,
	t     = 0
}
if opt.weightnoise > 0 then
	weightnoise = torch.randn(parameters:size()):cuda()
end

currentChunk = 0
currentIndex = 0
function Train()
	local numChunks       = #filepaths.train
	local totalNumSamples = meta.trainsamples
	local shuffleChunks   = torch.randperm(numChunks)
	local NLL             = 0
	local numCorrect      = 0
	local numPredictions  = 0
	local gradnorms       = {}
	local sampleCount     = 0
	autoencoder:training()

	-- load each chunk separately
	for k = 1, #filepaths.train do 
		currentChunk         = shuffleChunks[k]
		local filepath       = filepaths.train[shuffleChunks[k]]
		local train          = loaddata(filepath,opt.labelset)
		local numSamples     = train.numSamples
		local shuffle        = torch.randperm(numSamples):long()
		for t=1,numSamples,opt.batchSize do
			sampleCount = sampleCount + 1
			collectgarbage()
			xlua.progress(sampleCount,totalNumSamples)
			local optimfunc = function(x)
				if x ~= parameters then
					parameters:copy(x)
				end

				autoencoder:zeroGradParameters()
				-- nll keeps track of neg log likelihood of mini-batch
				local nll = 0

				-- since data is variable length, we do each sample individually
				for b=1,opt.batchSize do

					-- weight noise
					if opt.weightnoise > 0 then
						weightnoise:randn(parameters:size()):mul(opt.weightnoise)
						parameters:add(weightnoise)
					end

					-- grab data
					local index     = shuffle[t+b-1]
					local X         = train.x[index]
					local Y         = train.y[index]
					local T         = Y:size(1)
					currentIndex    = index

					-- labelmask is a one-hot encoding of y
					local labelmask = torch.zeros(T,model.outputDepth):cuda():scatter(2,Y:view(T,1),1)

					-- forward prop
					local logprobs  = autoencoder:forward({X,labelmask})

					-- nll keeps track of neg log likelihood of mini-batch
					if opt.normalizeNLL then
						nll         = -torch.cmul(labelmask,logprobs):sum()/T + nll
					else
						nll         = -torch.cmul(labelmask,logprobs):sum() + nll
					end

					-- NLL keeps track of neg log likelihood of entire dataset
					NLL             = NLL + nll

					-- backprop
					local dLdlogp   = -labelmask
					autoencoder:backward({X,labelmask},dLdlogp)

					-- keep track of accuracy of predictions
					local _, pred   = logprobs:max(2)
					pred = pred:squeeze()
					numCorrect      = numCorrect + torch.eq(pred,Y):sum()
					numPredictions  = numPredictions + Y:nElement()
				end

				-- normalize according to size of mini-batch
				if opt.batchSize > 1 then
					nll = nll/opt.batchSize
					gradients:div(opt.batchSize)
				end

				-- gradient clipping
				local gradnorm  = gradients:norm()
				table.insert(gradnorms,gradnorm)
				if gradnorm > opt.maxnorm then
					gradients:mul(opt.maxnorm/gradnorm)
				end
				
				-- L2 regularization
				if opt.weightDecay > 0 then
					nll = nll + 0.5*opt.weightDecay*(parameters:norm()^2)
					gradients:add(opt.weightDecay, parameters)
				end

				-- gradient noise
				if gradnoise.eta ~= 0 then
					gradnoise.t = (gradnoise.t or 0) + 1
					local sigma     = (gradnoise.eta/(1+gradnoise.t)^gradnoise.gamma)^0.5
					gradients:add(torch.randn(gradients:size()):cuda()*sigma)
				end

				return nll, gradients
			end

			-- optimize
			optimMethod(optimfunc, parameters, optimConfig, optimState)
			if opt.colnormconstr then
				TrainUtils.columnNormConstraintGraph(autoencoder)
			end

			-- report useful stats
			if t % 2000 == 0 then
				print('\ntrain gradnorm =', gradnorms[#gradnorms])
				local accuracy = numCorrect/numPredictions
				print('train accuracy =',torch.round(100*100*accuracy)/100 .. '%')
			end
		end
	end

	local accuracy = numCorrect/numPredictions
	local NLL      = NLL/totalNumSamples
	return accuracy, NLL, gradnorms
end

------------------ Evaluate ------------------
function Evaluate(filepath)
	local data           = loaddata(filepath,opt.labelset)
	local numSamples     = data.numSamples
	local NLL            = 0
	local numCorrect     = 0
	local numPredictions = 0
	local per            = torch.Tensor(numSamples)
	local predlist       = {}
	autoencoder:evaluate()
	for t=1,numSamples do
		xlua.progress(t,numSamples)
		local index      = t
		local X          = data.x[index]
		local Y          = data.y[index]
		local T          = Y:size(1)
		local labelmask  = torch.zeros(T,model.outputDepth):cuda():scatter(2,Y:view(T,1),1)
		local logprobs   = autoencoder:forward({X,labelmask})
		local nll        = -torch.cmul(labelmask,logprobs):sum()
		NLL              = NLL + nll
		local _, pred    = logprobs:max(2)
		pred = pred:squeeze()
		numCorrect       = numCorrect + torch.eq(pred,Y):sum()
		numPredictions   = numPredictions + Y:nElement()

		-- calculate CER
		local annotation = encoder.output
		local eos        = Y[T]
		local K          = opt.K
		local maxseqlen  = X:size(1)*2
		local prediction = decoder:BeamSearch(annotation,eos,K,maxseqlen)
		local dist       = WagnerFischer(prediction,Y)
		per[t]           = dist/T
		predlist[index]  = prediction:clone()
	end
	local accuracy = numCorrect/numPredictions
	NLL            = NLL/numSamples
	local CER      = per:mean()
	return accuracy, NLL, CER, predlist
end

------------------ Run ------------------
function updateLog(val,list)
	if not list then
		if type(val) == 'number' then
			return torch.Tensor(1):fill(val)
		elseif type(val) == 'table' then
			return torch.Tensor(val)
		else
			local view = val:view(1,unpack(val:size():totable()))
			return torch.Tensor(view:size()):fill(view)
		end
	else
		if type(val) == 'number' then
			return torch.cat(list,torch.Tensor(1):fill(val),1)
		elseif type(val) == 'table' then
			return torch.cat(list,torch.Tensor(val))
		else
			local view = val:view(1,unpack(val:size():totable()))
			return torch.cat(list,torch.Tensor(view:size()):fill(val),1)
		end
	end
end

	
-- training and validation loop
local trainLog     = {}
local validLog     = {}
local bestAccuracy = 0
local bestCER      = 1e20
for epoch = 1, opt.numEpochs do
	print('---------- epoch ' .. epoch .. '----------')
	local start = sys.clock()
	if optimConfigResets then
		if optimConfigResets[epoch] then
			for k,v in pairs(optimConfigResets[epoch]) do
				optimConfig[k] = v
			end
		end
	end
	print('device      = ' .. tostring(opt.device))
	print('optimConfig = ')
	print(optimConfig)
	print('optimState  = ')
	print(optimState)
	print('gradnoise   = ')
	print(gradnoise)
	print('')

	-- training
	local start = sys.clock()
	local trainAccuracy, trainNLL, gradnorms = Train()
	collectgarbage()
	local trainTime = (sys.clock()-start)/60
	print('\ntraining time   =', torch.round(trainTime) .. ' minutes')
	
	-- update training logs
	trainLog.accuracy  = updateLog(trainAccuracy, trainLog.accuracy)
	trainLog.nll       = updateLog(trainNLL, trainLog.nll)
	trainLog.gradnorms = updateLog(gradnorms, trainLog.gradnorms)

	-- print useful statistics
	print('train Accuracy  =', torch.round(100*100*trainAccuracy)/100 .. '%')
	print('train NLL       =', torch.round(100*trainNLL)/100)
	print('||grad||        =', gradients:norm())

	-- collect some useful outputs
	local alpha_train = decoder:alpha():float()
	local Ws_train    = decoder:Ws():float()
	local Vh_train    = decoder.Vh.output:float()
	
	-- validation
	local start = sys.clock()
	local validAccuracy, validNLL, validCER, predictions = Evaluate(filepaths.valid)
	collectgarbage()
	local validTime = (sys.clock() - start)/60
	print('\nvalidation time =', torch.round(validTime) .. ' minutes')

	-- update validation logs
	local start = sys.clock()
	validLog.accuracy  = updateLog(validAccuracy, validLog.accuracy)
	validLog.nll       = updateLog(validNLL, validLog.nll)
	validLog.CER       = updateLog(validCER, validLog.CER)

	-- print useful statistics
	print('valid Accuracy  =', torch.round(100*100*validAccuracy)/100 .. '%')
	print('valid NLL       =', torch.round(100*validNLL)/100)
	print('valid CER       =', torch.round(100*100*validCER)/100 .. '%')

	-- collect some useful outputs
	local alpha_valid = decoder:alpha():float()
	local Ws_valid    = decoder:Ws():float()
	local Vh_valid    = decoder.Vh.output:float()
	
	-- save logs and outputs
	print('saving to ' .. opt.savedir)
	sys.execute('mkdir -p ' .. opt.savedir)
	local writeFile = hdf5.open(paths.concat(opt.savedir,'log.h5'),'w')
	writeFile:write('train',trainLog)
	writeFile:write('valid',validLog)
	writeFile:write('alpha_train',alpha_train)
	writeFile:write('alpha_valid',alpha_valid)
	writeFile:write('Ws_train',Ws_train)
	writeFile:write('Ws_valid',Ws_valid)
	writeFile:write('Vh_train',Vh_train)
	writeFile:write('Vh_valid',Vh_valid)
	writeFile:write('output',autoencoder.output:float())
	writeFile:close()

	-- save current model
	torch.save(paths.concat(opt.savedir,'model.t7'),model)
	torch.save(paths.concat(opt.savedir,'predictions.t7'),predictions)

	-- save best model according to accuracy
	if validAccuracy > bestAccuracy then
		bestAccuracy = validAccuracy
		torch.save(paths.concat(opt.savedir,'model_best_valid_accuracy.t7'),model)
		torch.save(paths.concat(opt.savedir,'predictions_best_valid_accuracy.t7'),predictions)
	end

	-- save best model according to error rate
	if validCER < bestCER then
		bestCER = validCER
		torch.save(paths.concat(opt.savedir,'model_best_valid_CER.t7'),model)
		torch.save(paths.concat(opt.savedir,'predictions_best_valid_CER.t7'),predictions)
	end

	-- report total epoch time
	local totalTime = trainTime + validTime + (sys.clock() - start)/60
	print('\nepoch ' .. epoch .. ' completed in ' .. torch.round(totalTime) .. ' minutes\n')
end
