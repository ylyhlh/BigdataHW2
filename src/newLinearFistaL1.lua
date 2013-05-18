local newLinearFistaL1, parent = torch.class('unsup.newLinearFistaL1','unsup.UnsupModule')


-- inputSize   : size of input
-- outputSize  : size of code
-- lambda      : sparsity coefficient
-- params      : optim.FistaLS parameters
function newLinearFistaL1:__init(inputSize, outputSize, lambda, params)

   parent.__init(self)

   -- sparsity coefficient
   self.lambda = lambda
   -- dictionary is a linear layer so that I can train it
   self.D = nn.newLinear(outputSize, inputSize)
   -- L2 reconstruction cost
   self.Fcost = nn.MSECriterion()
   self.Fcost.sizeAverage = false;
   -- L1 sparsity cosr
   self.Gcost = nn.L1Cost()

   -- this is going to be set at each forward call.
   self.input = nil
   -- this is going to be passed to unsup.FistaLS
   self.code = torch.Tensor(outputSize):fill(0)

   -- Now I need a function to pass along as f
   -- input is code, do reconstruction, calculate cost
   -- and possibly derivatives too
   self.f = function(x, mode)
               local code = x
               local gradx = nil
               local input = self.input

               -- forward function evaluation
               local reconstruction = self.D:updateOutput(code)
               local fval = self.Fcost:updateOutput(reconstruction, input)

               -- derivative wrt code
               if mode and mode:match('dx') then
                  local gradr = self.Fcost:updateGradInput(reconstruction, input)
                  gradx = self.D:updateGradInput(code, gradr)
               end
               return fval, gradx
            end

   -- Next, we need function g that will be the non-smooth function
   self.g = function(x)
               local code = x
               local gradx = nil
               local fval = self.lambda * self.Gcost:updateOutput(code)
               if mode and mode:match('dx') then
                  gradx = self.Gcost:updateGradInput(code)
                  gradx:mul(self.lambda)
               end
               return fval, gradx
            end

   -- Finally we need argmin_x Q(x,y)
   self.pl = function(x, L)
                local code = x
                code:shrinkage(self.lambda/L)
             end

   -- this is for keeping parameters related to fista algorithm
   self.params = params or {}
   -- related to FISTA
   self.params.L = self.params.L or 0.1
   self.params.Lstep = self.params.Lstep or 1.5
   self.params.maxiter = self.params.maxiter or 50
   self.params.maxline = self.params.maxline or 20
   self.params.errthres = self.params.errthres or 1e-4
   self.params.doFistaUpdate = true

   self.gradInput = nil
   self:reset()
end

function newLinearFistaL1:reset(stdv)
   self.D:reset(stdv)
   self.D.bias:fill(0)
end

function newLinearFistaL1:parameters()
   --return {self.D.weight},{self.D.gradWeight}
   return self.D:parameters()
end

function newLinearFistaL1:initDiagHessianParameters()
   self.D:initDiagHessianParameters()
end

-- we do inference in forward
function newLinearFistaL1:updateOutput(input)
   self.input = input
   -- init code to all zeros
   self.code:fill(0)
   -- do fista solution
   local oldL = self.params.L
   local code, h = optim.FistaLS(self.f, self.g, self.pl, self.code, self.params)
   local fval = h[#h].F

   -- let's just half the params.L (eq. to double learning rate)
   if oldL == self.params.L then
      self.params.L = self.params.L / 2
   end

   return fval, h
end

-- no grad output, because we are unsup
-- d(||Ax-b||+lam||x||_1)/dx
function newLinearFistaL1:updateGradInput(input)
   -- calculate grad wrt to (x) which is code.
   if self.gradInput then
      local fval, gradf = self.fista.f(self.code,'dx')
      local gval, gradg = self.fista.g(self.code,'dx')
      self.gradInput:resizeAs(gradf):copy(gradf):add(gradg)
   end
   return self.gradInput
end

function newLinearFistaL1:zeroGradParameters()
   self.D:zeroGradParameters()
end

-- no grad output, because we are unsup
-- d(||Ax-b||+lam||x||_1)/dA
function newLinearFistaL1:accGradParameters(input)
   self.Fcost:updateGradInput(self.D.output,input)
   self.D:accGradParameters(self.code, self.Fcost.gradInput)
   self.D.gradBias:fill(0)
end

--[[function newLinearFistaL1:normalize()
   -- normalize the dictionary
end]]

function newLinearFistaL1:updateParameters(learningRate)
   self.D:updateParameters(learningRate)
   self.D.bias:fill(0)
end

function newLinearFistaL1:normalize()
   -- normalize the dictionary
   local w = self.D.weight
   for i=1,w:size(2) do
      w:select(2,i):div(w:select(2,i):norm()+1e-12)
   end
end

