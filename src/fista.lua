-------------------------------------------------------------------------------
-- FISTA with backtracking line search
--
-- f  smooth function
-- g  non-smooth function
-- pl minimizer of intermediate problem Q(x,y)
-- xinit initial point
-- params   : table of parameters (**optional**)
-- params.L : 1/(step size) for ISTA/FISTA iteration (0.1)
-- params.Lstep : step size multiplier at each iteration (1.5)
-- params.maxiter : max number of iterations (50)
-- params.maxline : max number of line search iterations per iteration (20)
-- params.errthres: Error thershold for convergence check (1e-4)
-- params.doFistaUpdate : true : use FISTA, false: use ISTA (true)
-- params.verbose : store each iteration solution and print detailed info (false)
-- 
-- On output, params will contain these additional fields that can be reused.
-- params.L       : last used L value will be written.
-- These are temporary storages needed by the algo and if the same params object is 
-- passed a second time, these same storages will be used without new allocation.
-- params.xkm     : previous iterarion point
-- params.y       : fista iteration
-- params.ply     : ply = pl(y - 1/L grad(f))
-- Returns the solution x and history of {function evals, number of line search ,...}
-- Algorithm is published in 
-- @article{beck-fista-09,
--    Author = {Beck, Amir and Teboulle, Marc},
--    Journal = {SIAM J. Img. Sci.},
--    Number = {1},
--    Pages = {183--202},
--    Title = {A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems},
--    Volume = {2},
--    Year = {2009}}
function hw2.FistaLS(f, g, pl, xinit, params)

   local params = params or {}
   local L = params.L or 0.1
   local Lstep = params.Lstep or 1.5
   local Lrate = params.Lrate or 0.2
   local maxiter = params.maxiter or 50
   local maxline = params.maxline or 20
   local errthres = params.errthres or 1e-4
   local doFistaUpdate = params.doFistaUpdate
   local doLinearSearch = params.doLinearSearch 
   local verbose = params.verbose

   -- temporary allocations
   params.xkm = params.xkm or torch.Tensor()
   params.y   = params.y   or torch.Tensor()
   params.ply = params.ply or torch.Tensor()
   local xkm = params.xkm  -- previous iteration
   local y   = params.y    -- fista iteration
   local ply = params.ply  -- soft shrinked y

   -- we start from all zeros
   local xk = xinit
   xkm:resizeAs(xk):zero()
   ply:resizeAs(xk):zero()
   y:resizeAs(xk):zero()

   local history = {} -- keep track of stuff
   local niter = 0    -- number of iterations done
   local converged = false  -- are we done?
   local tk = 1      -- momentum param for FISTA
   local tkp = 0


   local gy = g(y)
   local fval = math.huge -- fval = f+g
   while not converged and niter < maxiter do

      -- run through smooth function (code is input, input is target)
      -- get derivatives from smooth function
      local fy,gfy = f(y,'dx')

      --Linear search or spectral approximation 
      if doLinearSearch then 
          L = hw2.LinearSearch(y,L,pl,f,xk,xkm,maxline,Lstep)
      elseif (niter==0) then
	  L = hw2.LinearSearch(y,L,pl,f,xk,xkm,maxline,Lstep)
      else
	  L = hw2.PowerSearch(y,L,pl,f,xk,maxline,Lrate) 
      end
      ---------------------------------------------
      -- FISTA
      ---------------------------------------------
      if doFistaUpdate then
         -- do the FISTA step
         tkp = (1 + math.sqrt(1 + 4*tk*tk)) / 2
         -- x(k-1) = x(k-1) - x(k)
         xkm:add(-1,xk)
         -- y(k+1) = x(k) + (1-t(k)/t(k+1))*(x(k-1)-x(k))
         y:copy(xk)
         y:add( (1-tk)/tkp , xkm)
         -- store for next iterations
         -- x(k-1) = x(k)
         xkm:copy(xk)
      else
         y:copy(xk)
      end

      -- t(k) = t(k+1)
      tk = tkp
      local fply = f(y)
      local gply = g(y)
      if verbose then
	 print(string.format('iter=%d eold=%g enew=%g',niter,fval,fply+gply))
      end

      niter = niter + 1

      -- bookeeping
      fval = fply + gply
      history[niter] = {}
      history[niter].nline = nline
      history[niter].L  = L
      history[niter].F  = fval
      history[niter].Fply = fply
      history[niter].Gply = gply
      params.L = L
      if verbose then
         history[niter].xk = xk:clone()
         history[niter].y  = y:clone()
      end

      -- are we done?
      if niter > 1 and math.abs(history[niter].F - history[niter-1].F) <= errthres then
         converged = true
	 xinit:copy(y)
         return y,history
      end

      if niter >= maxiter then
	 xinit:copy(y)
         return y,history
      end

   end
   error('not supposed to be here')
end

