require 'nn';
require 'nngraph';
require 'Recurrent';
require 'LinearZeroBias';

local GRU, parent = torch.class('nn.GRU','nn.Recurrent')

function GRU:__init(diminput,dimoutput)
	assert(diminput  ~= nil, "diminput must be specified")
	assert(dimoutput ~= nil, "dimoutput must be specified")

	self.diminput  = diminput
	self.dimoutput = dimoutput


	local x      = nn.Identity()()
	local prev_h = nn.Identity()()
	
	self.x       = x
	self.prev_h  = prev_h

	local hx     = nn.JoinTable(1,1)({prev_h,x})
	local z      = nn.Sigmoid()(nn.LinearZeroBias(diminput+dimoutput,dimoutput)(hx))
	local r      = nn.Sigmoid()(nn.LinearZeroBias(diminput+dimoutput,dimoutput)(hx))
	local rh_x   = nn.JoinTable(1,1)({nn.CMulTable()({r,prev_h}),x})
	local h_     = nn.Tanh()(nn.LinearZeroBias(diminput+dimoutput,dimoutput)(rh_x))
	local zh_    = nn.CMulTable()({z,h_})
	local v1     = nn.AddConstant(1)(nn.MulConstant(-1)(z))
	local v2     = nn.CMulTable()({v1,prev_h})
	local h      = nn.CAddTable()({v2,zh_})

	self.z       = z
	self.zh_     = zh_
	self.v1      = v1
	self.v2      = v2
	self.h       = h

	local gru    = nn.gModule({x,prev_h},{h})
	self.gru     = gru

	parent.__init(self,gru,dimoutput)

end

function GRU:updateGradInput(input,gradOutput)
	if type(gradOutput) == 'table' then
		return parent.updateGradInput(self,input,gradOutput[1])
	else
		return parent.updateGradInput(self,input,gradOutput)
	end
end
