local ImageTable = torch.class('ImageTable')

function ImageTable:__init(fileList,train_size)
   self.dataset ={}
  
   file = torch.DiskFile(fileList, 'r')
   local nameTable = self:getNameTable(train_size,file)

   setmetatable(self.dataset, {
 	 __index = function(t, key)
      local img = image.loadJPG('train256/'..nameTable[key%nameTable.size()+1])
      ---some pictures have only one channel
      if(img:size(1)==1) then
		local tmp = torch.Tensor(3,img:size(2),img:size(3))
		tmp[1]=img[1]
		tmp[2]=img[1]
		tmp[3]=img[1]
		img=tmp
      end    

      return image.rgb2yuv(img)
    end
    })

end

function getData() 
	return self.dataset
end


function ImageTable:features() return 121 end

function ImageTable:getNameTable(train_size,file)
   local trainName = {}
   function trainName:size() return train_size end
   function trainName:features() return ImageTable:features() end
   -- Randomize, vectorize and insert trainNameing data all into trainName
   local rorder = torch.randperm(trainName:size())
   for i = 1,trainName:size() do
      local name = file:readString('*l')
      trainName[rorder[i]] = name
   end
   return trainName 
end 
