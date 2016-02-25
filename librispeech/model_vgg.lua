require 'nn';
package.path = package.path .. ';../?.lua'
require 'Attention';
require 'GRU';
require 'nngraph';
require 'RNN';
require 'Maxout';
require 'Transpose2';
TrainUtils = require 'TrainUtils';

function loadmodel(opt)
	local model = {}

	------------------ Encoder ------------------
	model.inputFrameSize  = opt.inputFrameSize or 123
	model.hiddenFrameSize = opt.hiddenFrameSize or 256
	model.outputFrameSize = opt.outputFrameSize or 512
	model.enc_peepholes   = opt.enc_peepholes or false
	model.window          = opt.window or 5

	local enc_inp   = nn.Identity()()

	local conv      = nn.Sequential()
	conv:add(   nn.SpatialConvolutionMM(  3, 64,  3,  3))
	conv:add(   nn.ReLU())
	conv:add(   nn.SpatialConvolutionMM( 64, 64,  3,  3))
	conv:add(   nn.ReLU())
	conv:add(   nn.SpatialMaxPooling( 2, 1, 2, 1))
	conv:add(   nn.SpatialConvolutionMM( 64, 128, 3, 3))
	conv:add(   nn.ReLU())
	conv:add(   nn.SpatialConvolutionMM(128, 128, 3, 3))
	conv:add(   nn.ReLU())
	conv:add(   nn.SpatialMaxPooling( 2, 2, 2, 2))

	-- L = L - 8
	-- L = math.floor(L/2)

	local H = model.inputFrameSize
	H = H - 4
	H = math.floor(H/2)
	H = H - 4
	H = math.floor(H/2)
	local view = 128*H

	conv:add(  nn.Transpose2({1,2},3)) -- nFeat x L x H -> L x nFeat x H
	conv:add(  nn.View(-1,view):setNumInputDims(3)) --> L x nFeat*H
	conv:add(  nn.TemporalConvolution(view,2048,1))
	conv:add(  nn.ReLU())
	conv:add(  nn.TemporalConvolution(2048,2048,1))
	conv:add(  nn.ReLU())
	conv:add(  nn.TemporalConvolution(2048,2048,1))
	conv:add(  nn.ReLU())
	conv:add(  nn.TemporalConvolution(2048,model.outputFrameSize,1))
	conv:add(  nn.ReLU())

	local encoder   = nn.gModule({enc_inp},{conv(enc_inp)})

	------------------ Decoder ------------------
	model.scoreDepth              = opt.scoreDepth or 512
	model.hybridAttendFilterSize  = opt.hybridAttendFilterSize or 10
	model.hybridAttendFeatureMaps = opt.hybridAttendFeatureMaps or 0
	model.stateDepth              = opt.stateDepth or 256
	model.annotationDepth         = model.outputFrameSize
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
	mlp:add(nn.Linear(model.mlpDepth,model.mlpDepth))
	mlp:add(nn.Maxout(model.mlpDepth,model.mlpDepth,7))
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
