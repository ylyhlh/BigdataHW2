function getPatches(img, channel,size,kernelsize,disp)
	patches = {}
	size = size+2*torch.floor(kernelsize/2)
	local patchNum = 0
	local imag = img[channel]
	local height = imag:size(1)
	local width = imag:size(2)
	patches.size = function() return torch.floor(height/size)*torch.floor(width/size) end
	local rorder = torch.randperm(patches.size())
	for i = 1,height-size+1,size do
		for j = 1,width-size+1,size do
			patchNum = patchNum + 1
			patches[rorder[patchNum]] = imag[{{i,i+size-1},{j,j+size-1}}]
			patches[rorder[patchNum]]=lcn(patches[rorder[patchNum]],kernelsize):clone()
			if(not disp) then
				patches[rorder[patchNum]]:resize((size-4)*(size-4))
			end
			--patches[rorder[patchNum]] = patches[rorder[patchNum]]:add(-1*torch.mean(patches[rorder[patchNum]] ))
			--patches[rorder[patchNum]] = patches[rorder[patchNum]]:div(torch.std(patches[rorder[patchNum]])+1e-12 )
			--print(patches[rorder[patchNum]]:std(),patches[rorder[patchNum]]:mean())
		end
	end
	--print(patchNum,patches.size(),height,width)
 	return patches
end


function convf(patch)
	N=11;
	fx=torch.Tensor(N,N)
	fy=torch.Tensor(N,N)
	f=torch.linspace(-N/2,N/2-1,N)
	for i=1,N do 
		fx[{{i},{}}]=f
		fy[{{},{i}}]=f
	end
	rho=torch.sqrt(torch.cmul(fx,fx)+torch.cmul(fy,fy))
	f_0=0.4*N
	filt=torch.cmul(rho,torch.exp(torch.div(rho,f_0):pow(4):mul(-1)))
	return torch.conv2(patch,filt,'F')[{{6,14},{6,14}}]
end

function lcn(im,kernelsize)
	local gs = kernelsize
	local imsq = torch.Tensor()
	local lmnh = torch.Tensor()
	local lmn = torch.Tensor()
 	local lmnsqh = torch.Tensor()
   	local lmnsq = torch.Tensor()
	local lvar = torch.Tensor()
	local gfh = image.gaussian{width=gs,height=1,normalize=true}
	local gfv = image.gaussian{width=1,height=gs,normalize=true}
	local gf = image.gaussian{width=gs,height=gs,normalize=true}
	
      local mn = im:mean()
      local std = im:std()
      if data_verbose then
	 print('im',mn,std,im:min(),im:max())
      end
      im:add(-mn)
      im:div(std+1e-12)
      if data_verbose then
	 print('im',im:min(),im:max(),im:mean(), im:std())
      end

      imsq:resizeAs(im):copy(im):cmul(im)
      if data_verbose then
	 print('imsq',imsq:min(),imsq:max())
      end
	
      lmnh=torch.conv2(im,gfh)
      lmn=torch.conv2(lmnh,gfv)
      if data_verbose then
	 print('lmn',lmn:min(),lmn:max())
      end

      --local lmn = torch.conv2(im,gf)
      torch.conv2(lmnsqh,imsq,gfh)
      torch.conv2(lmnsq,lmnsqh,gfv)
      if data_verbose then	 
	 print('lmnsq',lmnsq:min(),lmnsq:max())
      end

      lvar:resizeAs(lmn):copy(lmn):cmul(lmn)
      lvar:mul(-1)
      lvar:add(lmnsq)
      if data_verbose then      
	 print('2',lvar:min(),lvar:max())
      end

      --lvar:apply(function (x) if x<0 then return 0 else return x end end)
      lvar[torch.lt(lvar,0)] = 0
      if data_verbose then
	 print('2',lvar:min(),lvar:max())
      end
      

      local lstd = lvar
      lstd:sqrt()
      --lstd:apply(function (x) if x<1 then return 1 else return x end end)
      lstd[torch.lt(lstd,1)]=1
      if data_verbose then
	 print('lstd',lstd:min(),lstd:max())
      end

      local shift = (gs+1)/2
      local nim = im:narrow(1,shift,im:size(1)-(gs-1)):narrow(2,shift,im:size(2)-(gs-1))
      nim:add(-1,lmn)
      nim:cdiv(lstd)
      if data_verbose then
	 print('nim',nim:min(),nim:max())
      end
      return nim
   end
