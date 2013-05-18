local Pos, parent = torch.class('nn.Pos', 'nn.Module')

function Pos:__init(inputSize)
   parent.__init(self)
end

function Pos:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input)
   self.output[torch.lt(input,0)] = 0
   return self.output 
end

function Pos:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(input):fill(1)
   self.gradInput[torch.lt(input,0)] = 0
   --print("NEg"..torch.norm(self.gradInput))
   return self.gradInput
end
