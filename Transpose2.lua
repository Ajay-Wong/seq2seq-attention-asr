local Transpose2, parent = torch.class('nn.Transpose2', 'nn.Module')

-- transpose dimensions:
-- n = nn.Transpose2({1,4},{1,3})
-- will transpose dims 1 and 4, then 1 and 3...
-- n = nn.Transpose2({1,4},{1,3},nDim)
-- use nDim for determining if input is mini-batch

function Transpose2:__init(...)
   parent.__init(self)
   self.permutations = {}
   for _,elem in ipairs({...}) do
	  if type(elem) == 'table' then
		 table.insert(self.permutations,elem)
      else
		 self.nDim = elem
      end
   end
end

function Transpose2:updateOutput(input)
   local batchadj
   if input:nDimension() == self.nDim or self.nDim == nil then	
      batchadj = 0
   elseif input:nDimension() == self.nDim + 1 then
      batchadj = 1
   else
	  error('inconsistent tensor size')
   end
   for _,perm in ipairs(self.permutations) do
      input = input:transpose(perm[1]+batchadj,perm[2]+batchadj)
   end
   self.output:resizeAs(input):copy(input)
   return self.output
end

function Transpose2:updateGradInput(input, gradOutput)
   local batchadj
   if input:nDimension() == self.nDim or self.nDim == nil then	
      batchadj = 0
   elseif input:nDimension() == self.nDim + 1 then
      batchadj = 1
   else
	  error('inconsistent tensor size')
   end
   for i = #self.permutations,1,-1 do
      local perm = self.permutations[i]
      gradOutput = gradOutput:transpose(perm[1]+batchadj,perm[2]+batchadj)
   end
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end

