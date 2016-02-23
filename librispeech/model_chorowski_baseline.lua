require 'nn';
package.path = package.path .. ';../?.lua'
require 'Attention';
require 'GRU';
require 'nngraph';
require 'RNN';
require 'Maxout';
TrainUtils = require 'TrainUtils';

function loadmodel(opt)
	local model = {}

	------------------ Encoder ------------------
	model.inputFrameSize  = opt.inputFrameSize or 123
	model.hiddenFrameSize = opt.hiddenFrameSize or 256
	model.outputFrameSize = opt.outputFrameSize or 256
	model.enc_peepholes   = opt.enc_peepholes or false
	model.window          = opt.window or 5

	local enc_inp   = nn.Identity()()

	local fRNN1     = nn.RNN(nn.GRU(model.inputFrameSize,model.hiddenFrameSize,model.enc_peepholes),false)(enc_inp)
	local bRNN1     = nn.RNN(nn.GRU(model.inputFrameSize,model.hiddenFrameSize,model.enc_peepholes),true)(enc_inp)
	local biRNN1    = nn.JoinTable(2,2)({fRNN1,bRNN1})

	local fRNN2     = nn.RNN(nn.GRU(model.hiddenFrameSize*2,model.hiddenFrameSize,model.enc_peepholes),false)(biRNN1)
	local bRNN2     = nn.RNN(nn.GRU(model.hiddenFrameSize*2,model.hiddenFrameSize,model.enc_peepholes),true)(biRNN1)
	local biRNN2    = nn.JoinTable(2,2)({fRNN2,bRNN2})

	local fRNN3     = nn.RNN(nn.GRU(model.hiddenFrameSize*2,model.outputFrameSize,model.enc_peepholes),false)(biRNN2)
	local bRNN3     = nn.RNN(nn.GRU(model.hiddenFrameSize*2,model.outputFrameSize,model.enc_peepholes),true)(biRNN2)
	local biRNN3    = nn.JoinTable(2,2)({fRNN3,bRNN3})

	local encoder   = nn.gModule({enc_inp},{biRNN3})


	------------------ Decoder ------------------
	model.scoreDepth              = opt.scoreDepth or 512
	model.hybridAttendFilterSize  = opt.hybridAttendFilterSize or 10
	model.hybridAttendFeatureMaps = opt.hybridAttendFeatureMaps or 0
	model.stateDepth              = opt.stateDepth or 256
	model.annotationDepth         = model.outputFrameSize*2
	model.outputDepth             = opt.outputDepth or 62
	model.peepholes               = opt.peepholes or false
	model.mlpDepth                = opt.mlpDepth or 64
	model.penalty                 = opt.penalty or 0

	local dr_inp                    = nn.Identity()()
	local gru_inp,prev_s,prev_mem   = dr_inp:split(3)
	local gru                       = nn.GRU(model.stateDepth,model.stateDepth)({gru_inp,prev_s})
	local decoder_recurrent         = nn.gModule({dr_inp},{gru,nn.Identity()(prev_mem)})

	local dec_mlp_inp = nn.Identity()()
	local mlp_inp = nn.JoinTable(1,1)(dec_mlp_inp)
	local mlp     = nn.Sequential()
	mlp:add(nn.Maxout(model.stateDepth+model.annotationDepth,model.mlpDepth,7))
	mlp:add(nn.Linear(model.mlpDepth,model.outputDepth))
	mlp:add(nn.LogSoftMax())
	local decoder_mlp = nn.gModule({dec_mlp_inp},{mlp(mlp_inp)})

	local decoder = nn.Attention(decoder_recurrent,
						   decoder_mlp,
						   model.scoreDepth,
						   model.hybridAttendFilterSize,
						   model.hybridAttendFeatureMaps,
						   model.stateDepth,
						   model.annotationDepth,
						   model.outputDepth,
						   true,
						   model.penalty)

	------------------ Autoencoder ------------------
	local autoenc_inp           = nn.Identity()()
	local enc_x, y              = autoenc_inp:split(2)
	local autoencoder           = nn.gModule({autoenc_inp},{decoder({encoder(enc_x),y})})


	model.autoencoder  = autoencoder
	model.encoder      = encoder
	model.decoder      = decoder

	return model
end
