require 'hdf5';

function loadfilepaths(datadir)
	local traindb = io.open(paths.concat(datadir,'train.db'),'r')
	local chunks = {}
	for line in traindb:lines() do
		table.insert(chunks,line)
	end
	local valid = paths.concat(datadir,'valid.h5')
	local test = paths.concat(datadir,'test.h5')
	local filepaths = {}
	filepaths.train = chunks
	filepaths.valid = valid
	filepaths.test  = test
	return filepaths
end

function split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
	 table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

function loadmeta(datadir)
	local file = io.open(paths.concat(datadir,'meta.txt'),'r')
	local meta = {}
	for line in file:lines() do
		k,v = unpack(split(line,' '))
		meta[k] = tonumber(v)
	end
	return meta
end


function loaddata(filepath,labelset)
	local labelset = labelset or 'chars'
	local file = hdf5.open(filepath,'r')
	local data = file:all()
	file:close()

	local dataset = {}
	dataset.x     = {}
	dataset.y     = {}
	local i=0
	for k,d in pairs(data) do
		i=i+1
		dataset.x[i] = d.x:cuda()
		dataset.y[i] = d[labelset]:cuda()
	end
	dataset.numSamples = i
	return dataset
end

