function loadPhonemeVocabs(filepath)
	local filepath = filepath or 'phonemes.txt'
	local vocab60 = {}
	local vocab48 = {}
	local vocab39 = {}
	local map48 = {}
	local map39 = {}
	local lines = {}
	local file = io.open(filepath,'r')

	for line in file:lines() do
		table.insert(lines,line)
	end
	for l = 2,#lines do
		local line = lines[l]
		local split = line:split(',')
		local i60,p60,i48,p48,i39,p39 = unpack(split)
		i60 = tonumber(i60)
		i48 = tonumber(i48)
		i39 = tonumber(i39)
		vocab60[i60] = p60
		vocab48[i48] = p48
		vocab39[i39] = p39
		map48[i60] = i48
		map39[i60] = i39
	end
	local vocabs = {}
	vocabs['vocab60'] = vocab60
	vocabs['vocab48'] = vocab48
	vocabs['vocab39'] = vocab39
	local maps = {}
	maps['map48'] = map48
	maps['map39'] = map39
	return vocabs,maps
end
