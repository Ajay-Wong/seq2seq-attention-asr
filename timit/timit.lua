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
require 'utils_timit';
require 'AdaptiveWeightNoise';
require 'WeightNoise';
TrainUtils = require 'TrainUtils'

opt                 = opt                or {}
opt.device          = opt.device         or 1
opt.datafile        = opt.datafile       or 'TIMIT/logmel.h5'
opt.savedir         = opt.savedir        or 'TIMIT/output/logmel'
opt.phonemes        = opt.phonemes       or 'TIMIT/phonemes.txt'
opt.batchSize       = opt.batchSize      or 1
opt.maxnorm         = opt.maxnorm        or 1
opt.weightDecay     = opt.weightDecay    or 1e-4
opt.maxnumsamples   = opt.maxnumsamples  or 1e20
opt.colnormconstr   = opt.colnormconstr  or false -- column norm constraint
opt.weightnoise     = opt.weightnoise    or 0
opt.numPhonemes     = opt.numPhonemes    or 62
opt.decode39        = opt.decode39       or true
opt.predict39       = opt.predict39      or false
opt.K               = opt.K              or 5 -- beam search width
opt.normalizeNLL    = opt.normalizeNLL   or false
opt.normalizeGrad   = opt.normalizeGrad  or false
opt.adaweightnoise  = opt.adaweightnoise or false
opt.adalambda       = opt.adalambda      or 1.0 -- complexity multiplier for adaptive weight noise
opt.adasigmainit    = opt.adasigmainit   or 0.075
print(opt)

cutorch.setDevice(opt.device)

------------------ Data ------------------
local file = hdf5.open(opt.datafile)
data = file:all()
file:close()

function processData(data)
    if type(data) == 'table' then
		if opt.batchSize == 1 then
			print('resetting opt.batchSize to 1 since data is a table, which is assumed to have variable length sequences')
			opt.batchSize = 1
		end
        dataset   = {}
        dataset.x = {}
        dataset.y = {}
        i=0
        for k,f in pairs(data) do
            i=i+1
            dataset.x[i] = f.x:cuda()
			if opt.predict39 then
				dataset.y[i] = f.y39:cuda()
			else
				dataset.y[i] = f.y:cuda()
			end
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
vocabs,maps = loadPhonemeVocabs(opt.phonemes)

function mapPhonemes(sequence,map)
    local new_sequence = sequence:clone()
    new_sequence:apply(function(x) return map[x] end)
    return new_sequence
end

------------------ Model ------------------
if model then
	print('loading model')
	autoencoder = model.autoencoder
	encoder     = model.encoder
	decoder     = model.decoder
	optimConfig = model.optimConfig
	optimState  = model.optimState
	gradnoise   = model.gradnoise
	AWN         = model.AWN

	model.opt = opt
else
	model = {}
	------------------ Encoder ------------------
	model.seqLength       = train.x[1]:size(1)
	model.inputFrameSize  = train.x[1]:size(2)
	model.hiddenFrameSize = 256
	model.outputFrameSize = 128
	model.kW              = 3

	L         = model.seqLength
	enc_inp   = nn.Identity()()
	
	convlayer = nn.Sequential()
	convlayer:add(nn.TemporalConvolution(model.inputFrameSize, model.hiddenFrameSize, model.kW))
	convlayer:add(nn.ReLU())
	convlayer:add(nn.TemporalMaxPooling(2,2))
	L         = (L - model.kW + 1)/2
	convlayer:add(nn.TemporalConvolution(model.hiddenFrameSize, model.hiddenFrameSize, model.kW))
	convlayer:add(nn.ReLU())
	convlayer:add(nn.TemporalMaxPooling(2,2))
	L         = (L - model.kW + 1)/2
	convlayer:add(nn.TemporalConvolution(model.hiddenFrameSize, model.hiddenFrameSize, model.kW))
	convlayer:add(nn.ReLU())
	convlayer:add(nn.TemporalMaxPooling(2,2))
	L         = (L - model.kW + 1)/2
	
	fRNN      = nn.RNN(nn.LSTM(model.hiddenFrameSize, model.outputFrameSize, False), false)(convlayer(enc_inp))
	bRNN      = nn.RNN(nn.LSTM(model.hiddenFrameSize, model.outputFrameSize, False), true)(convlayer(enc_inp))
	concat    = nn.JoinTable(2,2)({fRNN,bRNN})
	encoder   = nn.gModule({enc_inp},{concat})

	------------------ Decoder ------------------
	model.scoreDepth              = 150
	model.hybridAttendFilterSize  = 5
	model.hybridAttendFeatureMaps = 16
	model.stateDepth              = 400
	model.annotationDepth 		  = model.outputFrameSize*2
	model.outputDepth             = opt.numPhonemes
	model.peepholes               = false
	model.mlpDepth                = opt.numPhonemes*2 

	decoder_recurrent = nn.LSTM(model.stateDepth, model.stateDepth, model.peepholes)
	dec_mlp_inp = nn.Identity()()
	mlp_inp = nn.JoinTable(1,1)(dec_mlp_inp)
	mlp     = nn.Sequential()
	mlp:add(nn.Linear(model.stateDepth + model.annotationDepth,model.mlpDepth))
	mlp:add(nn.ReLU())
	mlp:add(nn.Linear(model.mlpDepth, model.outputDepth))
	mlp:add(nn.LogSoftMax())
	decoder_mlp = nn.gModule({dec_mlp_inp},{mlp(mlp_inp)})
	decoder = nn.Attention(decoder_recurrent,
						   decoder_mlp,
						   model.scoreDepth,
						   model.hybridAttendFilterSize,
						   model.hybridAttendFeatureMaps,
						   model.stateDepth,
						   model.annotationDepth,
						   model.outputDepth,
						   true,
						   opt.penalty)

	------------------ Autoencoder ------------------
	autoenc_inp           = nn.Identity()()
	enc_x, y              = autoenc_inp:split(2) 
	autoencoder           = nn.gModule({autoenc_inp},{decoder({encoder(enc_x),y})})

	model.autoencoder = autoencoder
	model.encoder     = encoder
	model.decoder     = decoder
	model.optimConfig = optimConfig
	model.optimState  = optimState
	model.gradnoise   = gradnoise
	model.opt         = opt
end

autoencoder           = autoencoder:cuda()
parameters, gradients = autoencoder:getParameters()

--if initialization then
--	initialization(parameters)
--end

------------------ Train ------------------
optimMethod = optimMethod or optim.adadelta
optimConfig = optimConfig or {
	eps = 1e-8,
	rho = 0.95
}
optimState  = optimState or nil
gradnoise   = gradnoise or {
	eta   = 1e-3,
	gamma = 0.55,
	t     = 0
}
if opt.weightnoise > 0 then
	print('initializing weight noise,','sigma =',opt.weightnoise)
	WeightNoise = nn.WeightNoise(parameters,opt.weightnoise):cuda()
	model.WeightNoise = WeightNoise
	wnparameters, wngradients = WeightNoise:getParameters()
end
if opt.adaweightnoise then
	print('initializing adaptive weight noise,','lambda =',opt.adalambda,'init std =',opt.adasigmainit)
	if AWN then
		AWN.lambda = opt.adalambda
	else
		AWN = nn.AdaptiveWeightNoise(parameters:clone(),opt.adalambda,opt.adasigmainit):cuda()
		model.AWN = AWN
	end
	adaparameters,adagradients = AWN:getParameters()
end
function Train()
	local numSamples     = math.min(opt.maxnumsamples,train.numSamples)
	local shuffle        = torch.randperm(numSamples):long()
	local NLL            = 0
	local numCorrect     = 0
	local numPredictions = 0
	local gradnorms      = {}
	local L_bound        = 0
	autoencoder:training()
	for t=1,numSamples,opt.batchSize do
		collectgarbage()
		xlua.progress(t+opt.batchSize-1,numSamples)
		local optimfunc = function(x)
			if opt.adaweightnoise then
				if x ~= adaparameters then
					adaparameters:copy(x)
				end
			elseif opt.weightnoise > 0 then
				if x ~= wnparameters then
					wnparameters:copy(x)
				end
			else
				if x ~= parameters then
					parameters:copy(x)
				end
			end

			autoencoder:zeroGradParameters()
			local nll = 0
			if opt.weightnoise > 0 then
				WeightNoise:zeroGradParameters()
			end

			-- since data is variable length, we do each sample individually
			for b=1,opt.batchSize do

				-- weight noise
				if opt.weightnoise > 0 then
					parameters:copy(WeightNoise:Sample())
				end

				-- adaptive weight noise
				if opt.adaweightnoise then
					--print('\nparameters:mean()',parameters:mean())
					--print('parameters:norm()',parameters:norm())
					parameters:copy(AWN:Sample())
					--print('parameters:norm()',parameters:norm())
				end

				-- grab data
				local index     = shuffle[t+b-1]
				local X         = train.x[index]
				local Y         = train.y[index]
				local T         = Y:size(1)

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
				if opt.normalizeGrad then
					dLdlogp     = dLdlogp/T
				end
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

			-- weight noise
			if opt.weightnoise > 0 then
				WeightNoise:forward(nll);
				WeightNoise:backward(nll,gradients)
				return nll, wngradients
			end

			-- adaptive weight noise
			if opt.adaweightnoise then
				AWN:zeroGradParameters()
				local L = AWN:forward(nll)
				L_bound = L_bound + L
				AWN:backward(nll,gradients)
				return L, adagradients
			else
				return nll, gradients
			end
		end

		-- optimization step
		if opt.adaweightnoise then
			optimMethod(optimfunc, adaparameters, optimConfig, optimState)
		elseif opt.weightnoise > 0 then
			optimMethod(optimfunc, wnparameters, optimConfig, optimState)
		else
			optimMethod(optimfunc, parameters, optimConfig, optimState)
		end

		-- column norm constraints
		if opt.colnormconstr then
			TrainUtils.columnNormConstraintGraph(autoencoder)
		end
		if t % 100 == 0 then
			print('\ntrain gradnorm =', gradnorms[#gradnorms])
			print('parameters:norm() =',parameters:norm())
			print('parameters:std() =',parameters:std())
			local accuracy = numCorrect/numPredictions
			print('train accuracy =',torch.round(100*100*accuracy)/100 .. '%')
			if opt.adaweightnoise then
				print('train L =',L_bound/t)
			else
				print('train nll =',NLL/t)
			end
		end
	end
	local accuracy = numCorrect/numPredictions
	NLL            = NLL/numSamples
	return accuracy, NLL, gradnorms
end

------------------ Evaluate ------------------
function Evaluate(data)
	local numSamples     = math.min(opt.maxnumsamples,data.numSamples)
	local NLL            = 0
	local numCorrect     = 0
	local numPredictions = 0
	local per            = torch.Tensor(numSamples)
	local predlist       = {}
	if opt.adaweightnoise then
		parameters:copy(AWN:Mode())
	elseif opt.weightnoise > 0 then
		parameters:copy(WeightNoise:Mode())
	end
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

		-- calculate PER
		local annotation = encoder.output
		local eos        = Y[T]
		local K          = opt.K
		local maxseqlen  = X:size(1)
		local prediction = decoder:BeamSearch(annotation,eos,K,maxseqlen)
		local dist
		if opt.decode39 and (opt.predict39 == false) then
			local pred39     = mapPhonemes(prediction,maps.map39)
			local target39   = mapPhonemes(Y,maps.map39)
			dist   	         = WagnerFischer(pred39,target39)
		else
			dist             = WagnerFischer(prediction,Y)
		end
		per[t]           = dist/T
		predlist[index]  = prediction:clone()
	end
	local accuracy = numCorrect/numPredictions
	NLL            = NLL/numSamples
	local PER      = per:mean()
	return accuracy, NLL, PER, predlist
end

------------------ Run ------------------
function updateList(val,list)
	if not list then
		return torch.Tensor(1):fill(val)
	else
		return torch.cat(list,torch.Tensor(1):fill(val),1)
	end
end

function updateLog(log,accuracy,nll,gradnorms)
	if log then
		log.accuracy = torch.cat(log.accuracy,torch.Tensor(1):fill(accuracy),1)
		log.nll      = torch.cat(log.nll,torch.Tensor(1):fill(nll),1)
	else
		log = {}
		log.accuracy = torch.Tensor(1):fill(accuracy)
		log.nll      = torch.Tensor(1):fill(nll)
	end
	if gradnorms then
		if log.gradnorms then
			log.gradnorms = torch.cat(log.gradnorms,torch.Tensor(gradnorms))
		else
			log.gradnorms = torch.Tensor(gradnorms)
		end
	end
	return log
end
	
--print('---------- evaluate initialization ----------')
--local validAccuracy, validNLL = Evaluate({numSamples=1,x={[1]=valid.x[1]},y={[1]=valid.y[1]}})
--print('valid Accuracy  =', torch.round(100*100*validAccuracy)/100 .. '%')
--print('valid NLL       =', torch.round(100*validNLL)/100)
--local alpha_valid = decoder:alpha():float()
--print('saving to ' .. opt.savedir)
sys.execute('mkdir -p ' .. opt.savedir)
--local writeFile = hdf5.open(paths.concat(opt.savedir,'alpha0.h5'),'w')
--writeFile:write('alpha_valid',alpha_valid)
--writeFile:write('output',autoencoder.output:float())
--writeFile:close()

numEpochs = opt.numEpochs or 100
local trainLog, validLog
local bestAccuracy = 0
local bestPER      = 1e20

function formatPercent(val,d)
	local d = d or 2
	return torch.round(10^2*val)/(10^2)
end

if opt.resume then
	local file = hdf5.open(paths.concat(opt.resumedir,'log.h5'),'r')
	local data = file:all()
	file:close()
	trainLog   = data.train
	validLog   = data.valid
	bestAccuracy = validLog.accuracy:max()
	bestPER      = validLog.PER:min()

	print('resuming from....')
	print(opt.resumedir)
	print('best accuracy =',formatPercent(bestAccuracy))
	print('best PER =',formatPercent(bestPER))

	local start = sys.clock()
	--local validAccuracy, validNLL, validPER, predictions = Evaluate(valid)
	local validTime = (sys.clock() - start)/60
	--print('\nvalidation time =', torch.round(validTime) .. ' minutes')
	--print('valid Accuracy  =', torch.round(100*100*validAccuracy)/100 .. '%')
	--print('valid NLL       =', torch.round(100*validNLL)/100)
	--print('valid PER       =', torch.round(100*100*validPER)/100 .. '%')
end


for epoch = 1, numEpochs do
	print('---------- epoch ' .. epoch .. '----------')
	local start = sys.clock()
	if optimConfigResets then
		if optimConfigResets[epoch] then
			for k,v in pairs(optimConfigResets[epoch]) do
				optimConfig[k] = v
			end
		end
	end
	print('device =',opt.device)
	print('optimConfig = ')
	print(optimConfig)
	print('optimState  = ')
	print(optimState)
	print('gradnoise   = ')
	print(gradnoise)
	print('')
	local start = sys.clock()
	local trainAccuracy, trainNLL, gradnorms = Train()
	local trainTime = (sys.clock()-start)/60
	print('\ntraining time   =', torch.round(trainTime) .. ' minutes')
	trainLog = updateLog(trainLog, trainAccuracy, trainNLL, gradnorms)
	print('train Accuracy  =', torch.round(100*100*trainAccuracy)/100 .. '%')
	print('train NLL       =', torch.round(100*trainNLL)/100)
	print('||grad||        =', gradients:norm())
	local alpha_train = decoder:alpha():float()
	local Ws_train    = decoder:Ws():float()
	local Vh_train    = decoder.Vh.output:float()
	
	local start = sys.clock()
	local validAccuracy, validNLL, validPER, predictions = Evaluate(valid)
	local validTime = (sys.clock() - start)/60
	print('\nvalidation time =', torch.round(validTime) .. ' minutes')

	local start = sys.clock()
	validLog = updateLog(validLog, validAccuracy, validNLL)
	validLog.PER = updateList(validPER, validLog.PER)
	print('valid Accuracy  =', torch.round(100*100*validAccuracy)/100 .. '%')
	print('valid NLL       =', torch.round(100*validNLL)/100)
	print('valid PER       =', torch.round(100*100*validPER)/100 .. '%')
	local alpha_valid = decoder:alpha():float()
	local Ws_valid    = decoder:Ws():float()
	local Vh_valid    = decoder.Vh.output:float()
	
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
	torch.save(paths.concat(opt.savedir,'model.t7'),model)
	torch.save(paths.concat(opt.savedir,'predictions.t7'),predictions)
	if validAccuracy > bestAccuracy then
		bestAccuracy = validAccuracy
		torch.save(paths.concat(opt.savedir,'model_best_valid_accuracy.t7'),model)
		torch.save(paths.concat(opt.savedir,'predictions_best_valid_accuracy.t7'),predictions)
	end
	if validPER < bestPER then
		bestPER = validPER
		torch.save(paths.concat(opt.savedir,'model_best_valid_PER.t7'),model)
		torch.save(paths.concat(opt.savedir,'predictions_best_valid_PER.t7'),predictions)
	end
	local totalTime = trainTime + validTime + (sys.clock() - start)/60
	print('\nepoch ' .. epoch .. ' completed in ' .. torch.round(totalTime) .. ' minutes\n')
end
