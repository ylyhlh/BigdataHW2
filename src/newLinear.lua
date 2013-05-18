local newLinear, parent = torch.class('nn.newLinear', 'nn.Module')

function newLinear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize):zero()
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize):zero()
   
   self:reset()
end

function newLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv):zero()
   end
end

function newLinear:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      self.output:resize(nframe, nunit)
      self.output:zero():addr(1, input.new(nframe):fill(1), self.bias)
      self.output:addmm(1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function newLinear:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function newLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput):zero()    
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      print(self.bias:max())
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1)):zero()
   end

end

-- we do not need to accumulate parameters when sharing
newLinear.sharedAccUpdateGradParameters = newLinear.accUpdateGradParameters
