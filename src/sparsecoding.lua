require 'image'
function L1sparsecoding(dictionarySize, lam)
	sparsecoding = {}
	fistaparams = {}
	fistaparams.doFistaUpdate = dofista
	fistaparams.maxline = 20
	fistaparams.maxiter = 30
	fistaparams.verbose = false
	fistaparams.lam = lam
	fistaparams.doLinearSearch= true
	
	sparsecoding.graddict = torch.Tensor()

--[[
	for i = 1,sparsecoding:dictionarySize()do
		sparsecoding.W[{{},i}]:copy(train_set[10*i+1000][1])
	end
--]]

	function sparsecoding:updateDictionary(x,code,learningrate)
		local M= torch.ger(code,code)
	sparsecoding.graddict:resizeAs(sparsecoding.W):zero():add(2,sparsecoding.W * M)
		sparsecoding.graddict:addr(-2,x,code)
		sparsecoding.W:add(-learningrate,sparsecoding.graddict)
		sparsecoding:normalizeW()
	end
	
	function sparsecoding:normalizeW()
		for i = 1,sparsecoding:dictionarySize() do
			local col = torch.norm(sparsecoding.W[{{},i}])
			sparsecoding.W[{{},i}]:div(col)
		end
	end

	function sparsecoding:randomInitial(train_set)
		for i = 1,sparsecoding:dictionarySize() do
			sparsecoding.W[{{},i}] =train_set[torch.random(1000)][1]
		end
	end


	function sparsecoding:train(train_set,learningrate,interval,decay, direct, initW)
		
		function sparsecoding:dictionarySize() return dictionarySize end
		function sparsecoding:features() return train_set:features() end
	
		sparsecoding.W = initW or torch.randn( sparsecoding:features(), sparsecoding:dictionarySize())
		sparsecoding:randomInitial(train_set)
		sparsecoding:normalizeW()

		sparsecoding.A = torch.zeros( sparsecoding:dictionarySize(),sparsecoding:dictionarySize())
		sparsecoding.B = torch.zeros( sparsecoding:features(),sparsecoding:dictionarySize())

		local fista = hw2.FistaL1( sparsecoding.W, fistaparams )


		sparsecoding:display()
		interval = interval or 100
		decay = decay or 1
		direct = direct or false
		local direct_flag = false
		
		local currentLR = learningrate
		for epoch = 1,30 do
			for k = 1,train_set:size() do
				--print(k)
				local x = train_set[k][1]
				local code = fista.run(x,fistaparams.lam )
				sparsecoding.A :addr(code,code)
				sparsecoding.B :addr(x,code)
				if( direct and direct_flag  ) then
					sparsecoding.W = torch.gesv(sparsecoding.B:t(),sparsecoding.A:t()):t()
				else
				sparsecoding:updateDictionary(x,code,currentLR)
				end
				if(k%interval==0) then
					print(k+(epoch-1)*train_set:size())
					direct_flag = true
					currentLR = learningrate/(1+decay*(k+(epoch-1)*train_set:size())/interval)
					sparsecoding:display()	
				end
			end
		end
		return sparsecoding.W
	end

	function sparsecoding:display()
		patches = {}
			local mean = torch.mean(sparsecoding.W)
		local stdd = torch.std(sparsecoding.W)
		local mi = mean-1*stdd
		local ma = mean+1*stdd
		mi = torch.min(sparsecoding.W)
		ma = torch.max(sparsecoding.W)
		for i = 1,sparsecoding:dictionarySize() do
			--mi = torch.min(sparsecoding.W[{{},i}])
			--ma = torch.max(sparsecoding.W[{{},i}])
			patches[i] = sparsecoding.W[{{},i}]:clone():add(-mi):div(ma-mi)
			patches[i]:resize(28,28)
		end

		
		image.display({image=patches,nrow=16, min =0 ,padding=2})
	end
	return sparsecoding
end
