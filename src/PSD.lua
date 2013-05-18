require 'unsup'
require 'image'
require 'gnuplot'

dofile  'ImageTable.lua'
dofile  'getPatches.lua'
dofile  'mnist.lua'
dofile  'newLinear.lua'
dofile  'newLinearFistaL1.lua'
dofile  'specialPSD.lua'


if not arg then arg = {} end

cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-dir','outputs', 'subdirectory to save experimens in')
cmd:option('-seed', 1198981, 'initial random seed')
cmd:option('-kernelsize', 5, 'size of convolutional kernels')
cmd:option('-inputsize', 11, 'size of each input patch')
cmd:option('-lambda', 0.1, 'sparsity coefficient')
cmd:option('-beta',0.1, 'prediction error coefficient')
cmd:option('-file_list', 'file_list')
cmd:option('-eta',0.1,'learning rate')
cmd:option('-eta_encoder',0,'encoder learning rate')
cmd:option('-encoderType','Pos','encoder type')
cmd:option('-decay',0,'weigth decay')
cmd:option('-maxiter',80000,'max number of updates')
cmd:option('-statinterval',1000,'interval for saving stats and models')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:option('-conv', false, 'force convolutional dictionary')
cmd:option('-channel', 1, 'the channel to learn')
cmd:option('-display_encoder', true, 'display the encoder when program running')
cmd:option('-display_decoder', true, 'display the decoder when program running')
cmd:option('-err_weighted ', false, 'train 3 more times when error is big')
cmd:text()

local params = cmd:parse(arg)

local rundir = cmd:string('psd', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   --error('This experiment is already done!!!')
end

os.execute('mkdir -p ' .. params.rundir)
cmd:log(params.rundir .. '/log', params)

-- init random number generator
torch.manualSeed(params.seed)

-- create the dataset
--data = mnist:getDatasets(10000, 1000)
data = ImageTable(params.file_list,10000).dataset


-- creat unsup stuff
mlp = unsup.specialPSD(params.inputsize*params.inputsize, 256 , params.lambda, params.beta )

-- learning rates
if params.eta_encoder == 0 then params.eta_encoder = params.eta end
params.eta = torch.Tensor({params.eta_encoder, params.eta})

-- do learrning rate hacks
-- kex.nnhacks()

function train(module,dataset,channel)

   local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local avFistaIterations = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local currentLearningRate = params.eta
   
   local function updateSample(input, target, eta)
      local err,h = module:updateOutput(input)
      --print(err)
      module:zeroGradParameters()
      module:updateGradInput(input, err)
      module:accGradParameters(input, err)
      module:updateParameters(eta)
      module:normalize()
      return err, #h
   end
   mlp.decoder.D.weight=torch.rand( mlp.decoder.D.weight:size(1), mlp.decoder.D.weight:size(2))
   mlp.decoder:normalize()
   display(mlp.decoder.D.weight)

   local err = 0
   local iter = 0
   local img_iter = 1
   local img_inner_iter = 1
   img = getPatches(dataset[img_iter],channel,params.inputsize,params.kernelsize)
  
   for t = 1,params.maxiter do
----[[
      img_inner_iter = img_inner_iter + 1
      if( img_inner_iter > img.size()) then
         img_inner_iter = 1
         img_iter = img_iter+1
         img = getPatches(dataset[img_iter],channel,params.inputsize,params.kernelsize)

      end
      --print(img[img_inner_iter ]:size())
      local example ={ img[img_inner_iter ]}
--]]
      --local example = dataset[t]
      local serr, siter = updateSample(example[1],example[1] ,currentLearningRate)
      err = err + serr
      iter = iter + siter
      if(t>params.statinterval ) then
         if(serr> 2*avTrainingError[t/params.statinterval] and params.err_weighted ) then
            --print('d')
            ----[[
            updateSample(example[1],example[1] ,currentLearningRate)
            updateSample(example[1],example[1] ,currentLearningRate)
            updateSample(example[1],example[1] ,currentLearningRate)
		--]]--
         end
      	
      end
      if math.fmod(t , params.statinterval) == 0 then
         --module:updateParameters(currentLearningRate)
         --module:zeroGradParameters()
	 avTrainingError[t/params.statinterval] = err/params.statinterval
	 avFistaIterations[t/params.statinterval] = iter/params.statinterval

	 -- report
	 print('# iter=' .. t .. ' eta = ( ' .. currentLearningRate[1] .. ', ' .. currentLearningRate[2] .. ' ) current error = ' .. err)

	 -- plot training error
	 gnuplot.pngfigure(params.rundir .. '/error.png')
	 gnuplot.plot(avTrainingError:narrow(1,1,math.max(t/params.statinterval,2)))
	 gnuplot.title('Training Error')
	 gnuplot.xlabel('# iterations / ' .. params.statinterval)
	 gnuplot.ylabel('Cost')
	 -- plot training error
	 gnuplot.pngfigure(params.rundir .. '/iter.png')
	 gnuplot.plot(avFistaIterations:narrow(1,1,math.max(t/params.statinterval,2)))
	 gnuplot.title('LassoFista Iterations')
	 gnuplot.xlabel('# iterations / ' .. params.statinterval)
	 gnuplot.ylabel('Fista Iterations')
	 gnuplot.plotflush()
	 gnuplot.closeall()

	 -- plot filters
----[[
	 local dd,de
	 if torch.typename(mlp) == 'unsup.specialPSD' then
	    dd = image.toDisplayTensor{input=mlp.decoder.D.weight:transpose(1,2):unfold(2,11,11),padding=1,nrow=16}
	    de = image.toDisplayTensor{input=mlp.encoder:parameters()[1]:unfold(2,11,11),padding=1,nrow=16,symmetric=true}
	 else
	    de = image.toDisplayTensor{input=mlp.encoder.weight,padding=1,nrow=16}
	    dd = image.toDisplayTensor{input=mlp.decoder.D.weight,padding=1,nrow=16}
	 end
	 image.saveJPG(params.rundir .. '/filters_dec_' .. t .. '.jpg',dd)
	 image.saveJPG(params.rundir .. '/filters_enc_' .. t .. '.jpg',de)
--]]
         --display the dictionary
         if(params.display_decoder)  then
            display(mlp.decoder.D.weight)
         end
         if(params.display_encoder)  then
         display(mlp.encoder:parameters()[1]:t())
         end

	 -- store model
	 local mf = torch.DiskFile(params.rundir .. '/model_' .. t .. '.bin','w'):binary()
	 mf:writeObject(module)
	 mf:close()

	 -- write training error
	 local tf = torch.DiskFile(params.rundir .. '/error.mat','w'):binary()
	 tf:writeObject(avTrainingError:narrow(1,1,t/params.statinterval))
	 tf:close()

	 -- write # of iterations
	 local ti = torch.DiskFile(params.rundir .. '/iter.mat','w'):binary()
	 ti:writeObject(avFistaIterations:narrow(1,1,t/params.statinterval))
	 ti:close()

	 -- update learning rate with decay
	 currentLearningRate = params.eta/(1+(t/ params.statinterval)*params.decay)
	 err = 0
	 iter = 0
      end
   end
end
	function display(D)
		patches = {}
		local mean = torch.mean(D)
		local stdd = torch.std(D)
		local mi = mean-1*stdd
		local ma = mean+1*stdd
		mi = torch.min(D)
		ma = torch.max(D)
		for i = 1,256 do
			--mi = torch.min(D[{{},i}])
			--ma = torch.max(D[{{},i}])
			patches[i] = D[{{},i}]:clone():add(-mi):div(ma-mi)--:pow(0.4)
			patches[i]:resize(params.inputsize,params.inputsize)	
			--image.hflip(patches[i])
		end
		
		image.display({image=patches,nrow=16,min=0.3,max=0.7, scaleeach=false,padding=2})
		--image.display({image=patches,nrow=16,min=0.3,max=0.6, scaleeach=false,padding=2})
		--image.display({image=patches,nrow=16, scaleeach=true,padding=2})
	end
	
train(mlp,data,params.channel)
