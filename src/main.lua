hw2 = {}

require 'torch'
require 'gnuplot'
require 'image'
require 'nn'
dofile  'Pos.lua'
dofile  'fista.lua'
dofile  'fistaLS.lua'
dofile  'fistaPS.lua'
dofile  'fistaL1.lua'
dofile  'mnist.lua'
dofile  'sparsecoding.lua'
dofile  'ImageTable.lua'
dofile  'getPatches.lua'


if not arg then arg = {} end

cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-dir','outputs', 'subdirectory to save experimens in')
cmd:option('-seed', 119981, 'initial random seed')
cmd:option('-inputsize',32, 'size of each input patch')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-datafile', 'tr-berkeley-N5K-M56x56-lcn.bin','Data set file')
cmd:option('-eta',0.1,'learning rate')
cmd:option('-encoderType','exp','encoder type')
cmd:option('-momentum',0,'gradient momentum')
cmd:option('-decay',1,'weigth decay')
cmd:option('-maxiter',50000,'max number of updates')
cmd:option('-statinterval',100,'interval for saving stats and models')
cmd:option('-doLinearSearch', true, 'do linear search or spectral approxiamation')
cmd:option('-doFistaUpdate', true, 'do fista update')
cmd:option('-verbose', false, 'be verbose')
cmd:option('-newdict', true, 'be verbose')
cmd:option('-direct', false, 'be verbose')
cmd:option('-dict_name', 'Dictionary.dict', 'be verbose')
cmd:text()

local params = cmd:parse(arg)

local rundir = cmd:string('Sparsecoding', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

os.execute('mkdir -p ' .. params.rundir)
cmd:log(params.rundir .. '/log', params)


torch.manualSeed(params.seed)
--[[
a = ImageTable('file_list',100).dataset
b = a[100]
c = getPatches(b,1,100,5,true)
image.display({image=c,nrow=16,padding=1})
--]]

-- write the array on disk
train_set,test_set = mnist:getDatasets(params.maxiter, 1)
local sparsecoding = L1sparsecoding( 256, params.lambda)

if(params.newdict) then
	
	local dictionary = sparsecoding:train(train_set, params.eta, params.statinterval, params.decay, params.direct)

	file = torch.DiskFile(params.rundir ..params.dict_name, 'w')
	file:writeObject(sparsecoding.W)
	file:close() -- make sure the data is written
else
	-- reload the array
	file = torch.DiskFile(params.rundir ..params.dict_name, 'r')
	sparsecoding.W = file:readObject()
end


--[[

seed = seed or 13

dofista = true

torch.manualSeed(seed)
math.randomseed(seed)
nc = 3
ni = 30
no = 100
x = torch.Tensor(ni):zero()


fistaparams = {}
fistaparams.doFistaUpdate = dofista
fistaparams.maxline = 20
fistaparams.maxiter = 200
fistaparams.verbose = true

D=torch.randn(ni,no)

for i=1,D:size(2) do
   D:select(2,i):div(D:select(2,i):std()+1e-12)
end

mixi = torch.Tensor(nc)
mixj = torch.Tensor(nc)
for i=1,nc do
   local ii = math.random(1,no)
   local cc = torch.uniform(0,1/nc)
   mixi[i] = ii;
   mixj[i] = cc;
   print(ii,cc)
   x:add(cc, D:select(2,ii))
end

fista = hw2.FistaL1(D,fistaparams)
code,h = fista.run(x,0.1)
--fista.reconstruction:addmv(0,1,D,code)
rec = fista.reconstruction
--code,rec,h = fista:forward(x);

gnuplot.figure(1)
gnuplot.plot({'data',mixi,mixj,'+'},{'code',torch.linspace(1,no,no),code,'+'})
gnuplot.title('Fista = ' .. tostring(fistaparams.doFistaUpdate))

gnuplot.figure(2)
gnuplot.plot({'input',torch.linspace(1,ni,ni),x,'+-'},{'reconstruction',torch.linspace(1,ni,ni),rec,'+-'});
gnuplot.title('Reconstruction Error : ' ..  x:dist(rec) .. ' ' .. 'Fista = ' .. tostring(fistaparams.doFistaUpdate))
--w2:axis(0,ni+1,-1,1)
--]]--

