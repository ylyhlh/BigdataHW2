hw2.LinearSearch = function(y,L,pl,f,xk,xkm,maxline,Lstep)
      local fy,gfy = f(y,'dx')
      local maxline = maxline or 20
      local fply = 0
      local gply = 0
      local Q = 0
      local nline = 0
      local linesearchdone = false
      while not linesearchdone do
         -- take a step in gradient direction of smooth function
         local ply = y:clone()
         ply:add(-1/L,gfy)

         -- and solve for minimum of auxiliary problem
         pl(ply,L)
         -- this is candidate for new current iteration
         xk:copy(ply)

         -- evaluate this point F(ply)
         fply = f(ply)
         
         -- ply - y
         ply:add(-1, y)
         -- <ply-y , \Grad(f(y))>
         local Q2 = gfy:dot(ply)
         -- L/2 ||beta-y||^2
         local Q3 = L/2 * ply:dot(ply)
         -- Q(beta,y) = F(y) + <beta-y , \Grad(F(y))> + L/2||beta-y||^2 + G(beta)
         Q = fy + Q2 + Q3

         -- check if F(beta) < Q(pl(y),\t)
         if fply <= Q then --and Fply + Gply <= F then
            -- now evaluate G here  
            linesearchdone = true
         elseif  nline >= maxline then
            linesearchdone = true
            xk:copy(xkm) -- if we can't find a better point, current iter = previous iter
            --print('oops')
         else
            L = L * Lstep
         end
         nline = nline + 1
      end
      return L
end

