local WeightNoise, parent  = torch.class('nn.WeightNoise', 'nn.Module')

local eps = 1e-12

function WeightNoise:__init(parameters,sigma)
	parent.__init(self)
	self.sigma = sigma or 1e-3
	self.weight = parameters:clone()
	self.gradWeight = parameters:clone()
	self.sample = parameters:clone()
end

function WeightNoise:getWeights()
	return self.weight
end

function WeightNoise:Sample()
	self.sample:randn(self.sample:size())
	self.sample:mul(self.sigma)
	self.sample:add(self.weight)
	return self.sample
end

function WeightNoise:Mode()
	return self.weight
end

function WeightNoise:updateOutput(nll)
	self.nll = nll
	return self.nll
end

function WeightNoise:accGradParameters(input, gradOutput)
	self.gradWeight:add(gradOutput)
end
	


