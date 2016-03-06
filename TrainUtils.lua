require 'nn';
require 'nngraph';


local orthogonalize = function(m)
    if m.weight and m.bias then
        local w = torch.cat(m.weight,torch.view(m.bias,m.bias:size(1),1))
        if w:size(1) < w:size(2) then
            q,_ = torch.qr(w:t())
            q = q:t()
        else
            q,_ = torch.qr(w)
        end
        m.weight:copy(q[{{},{1,m.weight:size(2)}}])
        m.bias:copy(q[{{},m.weight:size(2)+1}])
    elseif m.weight then
        local w = m.weight
        if w:size(1) < w:size(2) then
            q,_ = torch.qr(w:t())
            q = q:t()
        else
            q,_ = torch.qr(w)
        end
        m.weight:copy(q)
    end
end


local checkOrthogonalization_ = function(m)
	local prefix = prefix or ''
    --print(prefix .. torch.typename(m))
	local w
    if m.weight and m.bias then
        w = torch.cat(m.weight,torch.view(m.bias,m.bias:size(1),1))
    elseif m.weight then
        w = m.weight
    end
	if w then
		local check
		if w:size(1) > w:size(2) then
			check = torch.mm(w:t(),w)
		else
			check = torch.mm(w,w:t())
		end
		local n = check:size(1)
		--print(check)
		local check = check - torch.eye(n)
		return check:norm()
	end
end

function columnNormConstraint(m,maxval)
    local maxval = maxval or 1
    if m.weight then
		local nan_check = m.weight:norm()
		if nan_check ~= nan_check then
			print('\nmodule',m)
			print('prior to colnorm constraint')
			print('nan_check:',nan_check)
			__debug_module = m
			error('found a nan, module saved to __debug_module')
		end
        local norm = m.weight:norm(2,2):expandAs(m.weight) + 1e-8
        local lt = torch.lt(norm,maxval):type(norm:type())
        local ge = torch.ge(norm,maxval):type(norm:type())
        local unchanged = torch.ones(norm:size()):type(norm:type()):cmul(lt)
        --print(unchanged)
        local constrained = torch.cmul(ge,norm):div(maxval)
        --print(constrained)
        local div = unchanged + constrained
		local nan_check = div:norm()
		if nan_check ~= nan_check then
			print('\nmodule',m)
			print('during colnorm constraint')
			print('nan_check:',nan_check)
			__debug_module = m
			error('found a nan, module saved to __debug_module')
		end
		if div:eq(0):any() then
			print('\nmodule',m)
			print('zeros in divisor during colnorm constraint')
			__debug_module = m
			error('found a nan, module saved to __debug_module')
		end
        m.weight:cdiv(div)
		local nan_check = m.weight:norm()
		if nan_check ~= nan_check then
			print('\nmodule:',m)
			print('norm:',norm:norm())
			print('nan_check:',nan_check)
			__debug_module = m
			error('found a nan, module saved to __debug_module')
		end
        --print(m.weight:norm(2,1))
    end
	--[[
    if m.bias then
        local norm = m.bias:norm(2)
        if norm > maxval then
            m.bias:div(norm)
        end
        --print(m.bias:norm(2))
    end]]
end

function checkColumnNormConstraint(m,maxval)
    local maxval = maxval or 1
    if m.weight then
        print(m.weight:norm(2,2))
    end
	--[[
    if m.bias then
        print(m.bias:norm(2))
    end]]
end

local getnorms
getnorms = function(t)
	if type(t) == 'table' then
		local norms = {}
		for k,v in pairs(t) do
			norms[k] = getnorms(v)
		end
		return norms
	else
		return t:norm()
	end
end

local checkoutput
checkoutput = function(m)
	if m.output then
		return getnorms(m.output)
	end
end

local apply2graph
apply2graph = function(graph,func,toggleprint,prefix)
	local prefix = prefix or ''
	local list
	local typename = torch.typename(graph) or ''
	if typename == 'nn.gModule' then
		list = graph.forwardnodes
	elseif graph.modules then
		list = graph.modules
	else
		if graph.weight then
			if toggleprint then
				local printname = typename
				if graph.__tostring__ then
					printname = graph:__tostring__()
				end
				print(prefix .. printname)
			end
		end
		local returnval = func(graph)
		if returnval then
			if toggleprint then
				print(prefix,returnval)
			end
		end
	end
	if list then
		if toggleprint then
			local printname = typename
			if graph.__tostring__ then
				printname = graph:__tostring__()
			end
			print(prefix .. printname)
		end
		local prefix = prefix .. '  '
		for i,n in pairs(list) do
			local m
			if torch.typename(n) == 'nngraph.Node' then
				m = n.data.module
			else
				m = n 
			end
			if m ~= nil then
				apply2graph(m,func,toggleprint,prefix)
			end
		end
	end
end

local orthogonalizeGraph = function(graph)
	apply2graph(graph,orthogonalize)
end

local checkOrthogonalization = function(graph)
	apply2graph(graph,checkOrthogonalization_,true)
end

local columnNormConstraintGraph = function(graph)
	apply2graph(graph,columnNormConstraint)
end

local checkColumnNormConstraintGraph = function(graph)
	apply2graph(graph,checkColumnNormConstraint,true)
end

TrainUtils = {
			['orthogonalize'] = orthogonalize,
			['orthogonalizeGraph'] = orthogonalizeGraph,
			['checkOrthogonalization'] = checkOrthogonalization,
			['columnNormConstraint'] = columnNormConstraint,
			['columnNormConstraintGraph'] = columnNormConstraintGraph,
			['checkColumnNormConstraint'] = checkColumnNormConstraint,
			['checkColumnNormConstraintGraph'] = checkColumnNormConstraintGraph,
			['apply2graph'] = apply2graph,
			['getnorms'] = getnorms,
			['checkoutput'] = checkoutput
		}

return TrainUtils
