local AdaptiveWeightNoise, parent  = torch.class('nn.AdaptiveWeightNoise', 'nn.Module')

function AdaptiveWeightNoise:__init(parameters,lambda)
	parent.__init(self)
	self.alpha_mu = 0
	self.alpha_sigma2 = 1
	self.n = parameters:size(1)
	self.weight = torch.Tensor(2*self.n):type(parameters:type())
	local mu,sigma2 = unpack(self.weight:split(self.n))
	mu:copy(parameters)
	sigma2:fill(1)
	self.sigma = sigma2:clone()
	self.sample = parameters:clone()
	self.gradWeight = self.weight:clone():zero()
	self.gradOutput2 = parameters:clone():zero()
	self.lambda = lambda or 1 -- equals 1/B, the multiplier on Lc
end

function AdaptiveWeightNoise:Sample()
	local n = self.n
	local mu,sigma2 = unpack(self.weight:split(n))
	self.sigma:copy(sigma2):sqrt()
	self.sample:randn(self.sample:size())
	self.sample:cmul(self.sigma)
	self.sample:add(mu)
	return self.sample
end

function AdaptiveWeightNoise:updateOutput(nll)
	local n = self.n
	local mu,sigma2 = unpack(self.weight:split(n))
	self.nll = nll
	if self.lambda > 0 then
		self.alpha_mu = mu:mean()
		self.alpha_sigma2 = sigma2:mean() + (mu-self.alpha_mu):pow(2):sum()/n
		self.KL  = 0.5*(n*torch.log(self.alpha_sigma2) - torch.log(sigma2):sum())
		self.KL  = self.KL + 0.5/self.alpha_sigma2*((mu-self.alpha_mu):pow(2):sum())
		self.KL  = self.KL + 0.5/self.alpha_sigma2*(sigma2:sum()-n*self.alpha_sigma2)
		self.L   = self.lambda*self.KL + self.nll
	else
		self.L = self.nll
	end
	return self.L
end

function AdaptiveWeightNoise:accGradParameters(input, gradOutput)
	local n = self.n
	local mu,sigma2 = unpack(self.weight:split(n))

	self.gradOutput2:copy(gradOutput):pow(2)
	
	local gradmu,gradsigma2 = unpack(self.gradWeight:split(self.n))
	if self.lambda > 0 then
		self.alpha_mu = mu:mean()
		self.alpha_sigma2 = sigma2:mean() + (mu-self.alpha_mu):pow(2):sum()/n
		gradmu:copy(mu):add(-self.alpha_mu):div(self.alpha_sigma2):mul(self.lambda)
		gradsigma2:copy(sigma2):pow(-1):mul(-1):add(1/self.alpha_sigma2):mul(0.5*self.lambda)
	end
	gradmu:add(gradOutput)
	gradsigma2:add(self.gradOutput2);
end
	


