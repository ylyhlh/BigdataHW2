dofile 'Pos.lua'
local specialPSD, parent = torch.class('unsup.specialPSD','unsup.PSD')

-- inputSize   : size of input
-- outputSize  : size of code
-- lambda      : sparsity coefficient
-- beta	       : prediction coefficient
-- params      : optim.FistaLS parameters
function specialPSD:__init(inputSize, outputSize, lambda, beta, params)
   
   -- prediction weight
   self.beta = beta

   -- decoder is L1 solution
   self.decoder = unsup.LinearFistaL1(inputSize, outputSize, lambda, params)

   -- encoder
   params = params or {}
   self.params = params
   self.params.encoderType = params.encoderType or 'softPlus'
   print(inputSize, outputSize)
   if params.encoderType == 'linear' then
      self.encoder = nn.Linear(inputSize,outputSize)
   elseif params.encoderType == 'softPlus' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.Linear(inputSize,outputSize))
      self.encoder:add(nn.SoftPlus())
   elseif params.encoderType == 'Pos' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.Linear(inputSize,outputSize))
      self.encoder:add(nn.Pos())
   elseif params.encoderType == 'tanh_shrink' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.Linear(inputSize,outputSize))
      self.encoder:add(nn.TanhShrink())
      self.encoder:add(nn.Diag(outputSize))
   else
      error('params.encoderType unknown " ' .. params.encoderType)
   end

   parent.__init(self, self.encoder, self.decoder, self.beta, self.params)

end
