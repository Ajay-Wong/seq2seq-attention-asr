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
	--model.outputDepth = model.outputDepth or opt.numPhonemes


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
	eta   = 1e-3,
	gamma = 0.55,
	t     = 0
}
if opt.weightnoise > 0 then
	weightnoise = torch.randn(parameters:size()):cuda()
end
function Train()
	local numSamples     = math.min(opt.maxnumsamples,train.numSamples)
	local shuffle        = torch.randperm(numSamples):long()
	local NLL            = 0
	local numCorrect     = 0
	local numPredictions = 0
	local gradnorms      = {}
	autoencoder:training()
	for t=1,numSamples,opt.batchSize do
		collectgarbage()
		xlua.progress(t+opt.batchSize-1,numSamples)
		local optimfunc = function(x)
			if x ~= parameters then
				parameters:copy(x)
			end

			autoencoder:zeroGradParameters()
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

				-- labelmask is a one-hot encoding of y
				local labelmask = torch.zeros(T,model.outputDepth):cuda():scatter(2,Y:view(T,1),1)

				-- forward prop
				local logprobs  = autoencoder:forward({X,labelmask})

				-- nll keeps track of neg log likelihood of mini-batch
				nll             = -torch.cmul(labelmask,logprobs):sum() + nll

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
			nll = nll/opt.batchSize
			gradients:div(opt.batchSize)

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
		optimMethod(optimfunc, parameters, optimConfig, optimState)
		if opt.colnormconstr then
			TrainUtils.columnNormConstraintGraph(autoencoder)
		end
		if t % 500 == 0 then
			print('\ntrain gradnorm =', gradnorms[#gradnorms])
			local accuracy = numCorrect/numPredictions
			print('train accuracy =',torch.round(100*100*accuracy)/100 .. '%')
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
	end
	local accuracy = numCorrect/numPredictions
	NLL            = NLL/numSamples
	local PER      = per:mean()
	return accuracy, NLL, PER
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
	
print('---------- evaluate initialization ----------')
local validAccuracy, validNLL = Evaluate({numSamples=1,x={[1]=valid.x[1]},y={[1]=valid.y[1]}})
print('valid Accuracy  =', torch.round(100*100*validAccuracy)/100 .. '%')
print('valid NLL       =', torch.round(100*validNLL)/100)
local alpha_valid = decoder:alpha():float()
print('saving to ' .. opt.savedir)
sys.execute('mkdir -p ' .. opt.savedir)
local writeFile = hdf5.open(paths.concat(opt.savedir,'alpha0.h5'),'w')
writeFile:write('alpha_valid',alpha_valid)
writeFile:write('output',autoencoder.output:float())
writeFile:close()

numEpochs = opt.numEpochs or 100
local trainLog, validLog
local bestAccuracy = 0
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
	print('optimConfig = ')
	print(optimConfig)
	print('optimState  = ')
	print(optimState)
	print('gradnoise   = ')
	print(gradnoise)
	print('')
	local start = sys.clock()
	local trainAccuracy, trainNLL, gradnorms = Train()
	local time = 
	print('\ntraining time   =', torch.round(10*(sys.clock()-start)/60)/10 .. ' minutes')
	trainLog = updateLog(trainLog, trainAccuracy, trainNLL, gradnorms)
	print('train Accuracy  =', torch.round(100*100*trainAccuracy)/100 .. '%')
	print('train NLL       =', torch.round(100*trainNLL)/100)
	print('||grad||        =', gradients:norm())
	local alpha_train = decoder:alpha():float()
	local Ws_train    = decoder:Ws():float()
	local Vh_train    = decoder.Vh.output:float()
	
	local start = sys.clock()
	local validAccuracy, validNLL, validPER = Evaluate(valid)
	print('\nvalidation time =', torch.round(10*(sys.clock()-start)/60)/10 .. ' minutes')
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
	if validAccuracy > bestAccuracy then
		bestAccuracy = validAccuracy
		torch.save(paths.concat(opt.savedir,'model_best.t7'),model)
	end
	local time = math.floor((sys.clock() - start)/60)
	print('\nepoch ' .. epoch .. ' completed in ' .. time .. ' minutes\n')
end
