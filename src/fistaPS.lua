hw2.PowerSearch = function(y,L,pl,f,xk,maxline,Lrate)
      local powersearchdone = false
      local vk = torch.rand(y:size())
      local Ltmp = torch.norm(vk)
      local Ltmp0 = torch.norm(vk)
      local npower = 0
      local fy,gfy = f(y,'dx')
      gfy0 = gfy:clone();
      while not powersearchdone do
          vk:div(Ltmp)
	  fy,gfy = f(y + vk,'dx')
          vk = gfy - gfy0
	  Ltmp = torch.norm(vk)  

	  if torch.abs(Ltmp - Ltmp0) < 1e-10 or npower >= maxline then
		L = (1 - Lrate) * Ltmp + Lrate * L
		fy,gfy = f(y,'dx')
		local ply = y:clone()
		ply:add(-1/L,gfy)
          	pl(ply,L)
         	xk:copy(ply)
		--print('ddd'..L)
		powersearchdone = true
	  else
		Ltmp0 = Ltmp
		npower = npower + 1
	  end
              
      end
      return L
end

