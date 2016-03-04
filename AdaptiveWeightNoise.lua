local AdaptiveWeightNoise, parent  = torch.class('nn.AdaptiveWeightNoise', 'nn.Module')

local eps = 1e-12

function AdaptiveWeightNoise:__init(parameters,lambda,sigma_init)
	parent.__init(self)
	self.alpha_mu = 0
	self.alpha_sigma2 = 1
	self.n = parameters:size(1)
	self.weight = torch.Tensor(2*self.n):type(parameters:type())
	local sigma_init = sigma_init or 1
	self:initialize(parameters,sigma_init)
	self.sigma = parameters:clone():zero()
	self.sigma2 = parameters:clone():zero()
	self.sample = parameters:clone():zero()
	self.gradWeight = self.weight:clone():zero()
	self.dLNds = parameters:clone():zero()
	self.lambda = lambda or 1 -- equals 1/B, the multiplier on Lc
end

function AdaptiveWeightNoise:getWeights()
	local n = self.n
	local mu,sigma2 = unpack(self.weight:split(n))
	return mu,sigma
end

function AdaptiveWeightNoise:Sample()
	local n = self.n
	local mu,s = unpack(self.weight:split(n))
	local sigma2 = self.sigma2:copy(s):exp()
	self.sigma:copy(sigma2):sqrt()
	self.sample:randn(self.sample:size())
	self.sample:cmul(self.sigma)
	self.sample:add(mu)
	--print('\nsample:mean()',self.sample:mean())
	--print('sample:std()',self.sample:std())
	return self.sample
end

function AdaptiveWeightNoise:initialize(mu_init,sigma_init)
	local n = self.n
	-- s = log(sigma^2)
	local mu,s = unpack(self.weight:split(n))
	if type(mu) == 'number' then
		mu:fill(mu_init)
	else
		mu:copy(mu_init)
	end
	if type(sigma_init) == 'number' then
		--sigma2:fill(sigma_init^2)
		s:fill(sigma_init^2):log()
	else
		--sigma2:copy(sigma_init):pow(2)
		sigma2:copy(sigma_init):pow(2):log()
	end
end

function AdaptiveWeightNoise:Mode()
	local mu,_ = unpack(self.weight:split(self.n))
	return mu
end

function AdaptiveWeightNoise:updateOutput(nll)
	self.nll = nll
	if self.lambda > 0 then
		local n = self.n
		-- s = log(sigma^2)
		local mu,s = unpack(self.weight:split(n))
		local sigma2 = self.sigma2:copy(s):exp()
		self.alpha_mu = mu:mean()
		self.alpha_sigma2 = math.max(eps,sigma2:mean() + (mu-self.alpha_mu):pow(2):sum()/n)
		self.KL  = 0.5*(n*torch.log(self.alpha_sigma2) - s:sum())
		self.KL  = self.KL + 0.5/self.alpha_sigma2*((mu-self.alpha_mu):pow(2):sum())
		self.KL  = self.KL + 0.5/self.alpha_sigma2*(sigma2:sum()) - n/2
		self.L   = self.lambda*self.KL + self.nll
	else
		self.L = self.nll
	end
	return self.L
end

function AdaptiveWeightNoise:accGradParameters(input, gradOutput)
	local n = self.n
	-- s = log(sigma^2)
	local mu,s = unpack(self.weight:split(n))
	local sigma2 = self.sigma2:copy(s):exp()
	self.dLNds:copy(gradOutput):pow(2):cmul(sigma2):mul(0.5)
	local gradmu,grads = unpack(self.gradWeight:split(self.n))
	if self.lambda > 0 then
		self.alpha_mu = mu:mean()
		self.alpha_sigma2 = math.max(eps,sigma2:mean() + (mu-self.alpha_mu):pow(2):sum()/n)
		gradmu:copy(mu):add(-self.alpha_mu):div(self.alpha_sigma2):mul(self.lambda)
		grads:copy(sigma2):mul(self.lambda*0.5/self.alpha_sigma2):add(-self.lambda*0.5)
		--print('\n||dLEdmu|| =',gradmu:norm())
		--print('||dLEds|| =',grads:norm())
		gradmu:add(gradOutput)
		grads:add(self.dLNds);
	else
		gradmu:copy(gradOutput)
		grads:copy(self.dLNds);
	end
	--print('||dLNdmu|| =',gradOutput:norm())
	--print('||dLNds|| =',self.dLNds:norm(),'\n')
end
	


