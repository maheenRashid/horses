
do  
    local data = torch.class('data')

    function data:__init(args)
        self.file_path_horse=args.file_path_horse
        self.file_path_human=args.file_path_human
        self.limit=args.limit;
        self.humanImage=args.humanImage;
        self.augmentation=args.augmentation;
        print ('self.augmentation',self.augmentation);

        self.batch_size=10;
        -- self.batch_size_human=10;
        -- _seg=32;
        -- self.batch_size_positive_score=16;
        -- self.batch_size_negative_score=16;


        self.start_idx_horse=1;
        -- self.humanImage=false;
        self.params={input_size={32,32},
                    mean={122,117,104},
                    labels_shape={5,3},
                    angles={-10,-5,0,5,10}};

        self.training_set_horse={};
        self.training_set_human={};
        
        self.lines_horse=self:readDataFile(self.file_path_horse);
        if self.humanImage then
            self.lines_human=self:readDataFile(self.file_path_human);
        -- ,self.humanImage);
        else
            self.lines_human=self:readDataFileNoIm(self.file_path_human);
        end
        assert (#self.lines_horse == #self.lines_human)
        
        if self.augmentation then
            -- print ('TRUE');
            self.lines_horse,self.lines_human=self:shuffleLines(self.lines_horse,self.lines_human);
        end

        print (#self.lines_horse,#self.lines_human);
        -- print (self.lines_human[1]);
        if self.limit~=nil then
            local lines_horse=self.lines_horse;
            self.lines_horse={};

            local lines_human=self.lines_human;
            self.lines_human={};
            
            for i=1,self.limit do
                self.lines_horse[#self.lines_horse+1]=lines_horse[i];
                self.lines_human[#self.lines_human+1]=lines_human[i];
            end
        end
    end


    function data:shuffleLines(lines,lines2)
        local x=lines;
        local len=#lines;
        local y=lines2;

        local shuffle = torch.randperm(len)
        
        local lines={};
        local lines2={};
        for idx=1,len do
            lines[idx]=x[shuffle[idx]];
            lines2[idx]=y[shuffle[idx]];
        end
        return lines,lines2;
    end

    function data:getTrainingData()
        local start_idx_horse_before = self.start_idx_horse

        self.training_set_horse.data=torch.zeros(self.batch_size,3,self.params.input_size[1]
            ,self.params.input_size[2]);
        self.training_set_horse.label=torch.zeros(self.batch_size,self.params.labels_shape[1],self.params.labels_shape[2]);

        if self.humanImage then
            self.training_set_human.data=self.training_set_horse.data:clone();
            self.training_set_human.label=self.training_set_horse.label:clone();
            self.start_idx_horse=self:addTrainingData(self.training_set_horse,self.training_set_human,self.batch_size,
            self.lines_horse,self.lines_human,self.start_idx_horse,self.params)
        else
            self.training_set_human.label=self.training_set_horse.label:clone();
            self.start_idx_horse=self:addTrainingDataNoIm(self.training_set_horse,self.training_set_human,self.batch_size,
            self.lines_horse,self.lines_human,self.start_idx_horse,self.params)    
        end
        
        if self.start_idx_horse<start_idx_horse_before then
            print ('shuffling data'..self.start_idx_horse..' '..start_idx_horse_before )
            self.lines_horse,self.lines_human=self:shuffleLines(self.lines_horse,self.lines_human);
        end

    end

    function data:readDataFileNoIm(file_path)
        local file_lines = {};
        print (file_path);
        for line in io.lines(file_path) do 
            local start_idx, end_idx = string.find(line, ' ');
            local img_label=string.sub(line,1,start_idx-1);

            local string_temp=string.sub(line,end_idx+1,#line);
            local start_idx, end_idx = string.find(string_temp, ' ');
            local size_r=string.sub(string_temp,1,start_idx-1);
            local size_c=string.sub(string_temp,end_idx+1,#string_temp);
            file_lines[#file_lines+1]={img_label,tonumber(size_r),tonumber(size_c)};

            -- local img_label=string.sub(line,end_idx+1,#line);
            -- file_lines[#file_lines+1]={img_path,img_label};
        end 
        return file_lines
    end

    function data:readDataFile(file_path)
        local file_lines = {};
        for line in io.lines(file_path) do 
            local start_idx, end_idx = string.find(line, ' ');
            local img_path=string.sub(line,1,start_idx-1);
            local img_label=string.sub(line,end_idx+1,#line);
            file_lines[#file_lines+1]={img_path,img_label};
        end 
        return file_lines

    end


    function data:hFlipImAndLabel(im,label)
        
        -- print ('hflip');
        
        if im then
            image.hflip(im,im);
        end

        label[{{},2}]=-1*label[{{},2}]
        local temp=label[{1,{}}]:clone();
        label[{1,{}}]=label[{2,{}}]:clone()
        label[{2,{}}]=temp;

        temp=label[{4,{}}]:clone();
        label[{4,{}}]=label[{5,{}}]:clone()
        label[{5,{}}]=temp;
        -- for i=1,label:size(1) do
        --     label[i][2]=1-(label[i][2]+1)
        -- end
        
        return im,label;
    end

    function data:rotateImAndLabel(img_horse,label_horse,img_human,label_human,angles)
        
        -- print ('hflip');
        
        
        local rand=math.random(#angles);
        local angle=math.rad(angles[rand]);
        local img_horse=image.rotate(img_horse,angle,"bilinear");

        local img_human_new;
        if img_human then
            img_human_new=image.rotate(img_human,angle,"bilinear");
        else
            img_human_new=img_human;
        end
        
        local rotation_matrix=torch.zeros(2,2);
        rotation_matrix[1][1]=math.cos(angle);
        rotation_matrix[1][2]=math.sin(angle);
        rotation_matrix[2][1]=-1*math.sin(angle);
        rotation_matrix[2][2]=math.cos(angle);
        
        for i=1,label_horse:size(1) do
            if label_horse[i][3]>0 then
                local ans = rotation_matrix*torch.Tensor({label_horse[i][2],label_horse[i][1]}):view(2,1);
                label_horse[i][1]=ans[2][1];
                label_horse[i][2]=ans[1][1];

                ans = rotation_matrix*torch.Tensor({label_human[i][2],label_human[i][1]}):view(2,1);
                label_human[i][1]=ans[2][1];
                label_human[i][2]=ans[1][1];
            end
        end
        
        
        return img_horse,label_horse,img_human_new,label_human; 
    end

    function data:processImAndLabel(img_horse,img_human,label_horse,label_human,params)
        -- print ('processImAndLabel');
        
        -- print ('hflip');
        
        img_horse:mul(255);
        img_human:mul(255);
        
        local org_size_horse=img_horse:size();
        local org_size_human=img_human:size();

        local label_horse_org=label_horse:clone();
        local label_human_org=label_human:clone();

        
        
        img_horse = image.scale(img_horse,params.input_size[1],params.input_size[2]);
        img_human = image.scale(img_human,params.input_size[1],params.input_size[2]);
        

        label_horse[{{},1}]=label_horse[{{},1}]/org_size_horse[3]*params.input_size[1];
        label_horse[{{},1}]=(label_horse[{{},1}]/params.input_size[1]*2)-1;
        label_horse[{{},2}]=label_horse[{{},2}]/org_size_horse[2]*params.input_size[2];
        label_horse[{{},2}]=(label_horse[{{},2}]/params.input_size[2]*2)-1;
        local temp = label_horse[{{},1}]:clone();
        label_horse[{{},1}]=label_horse[{{},2}]
        label_horse[{{},2}]=temp

        label_human[{{},1}]=label_human[{{},1}]/org_size_human[3]*params.input_size[1];
        label_human[{{},1}]=(label_human[{{},1}]/params.input_size[1]*2)-1;
        label_human[{{},2}]=label_human[{{},2}]/org_size_human[2]*params.input_size[2];
        label_human[{{},2}]=(label_human[{{},2}]/params.input_size[2]*2)-1;
        local temp = label_human[{{},1}]:clone();
        label_human[{{},1}]=label_human[{{},2}]
        label_human[{{},2}]=temp

        if (torch.max(label_horse)>=params.input_size[1]) then
            print ('PROBLEM horse');
            print (label_horse_org);
            print (label_horse);
            print (org_size_horse);
        end

        if (torch.max(label_human)>=params.input_size[1]) then
            print ('PROBLEM human');
            print (label_human_org);
            print (label_human);
            print (org_size_human);
        end
        
        -- flip or rotate
        if self.augmentation then
            -- print ('TRUE');
            local rand=math.random(2);
            -- rand=0;
            if rand==1 then
                img_horse,label_horse = self:hFlipImAndLabel(img_horse,label_horse);
                img_human,label_human = self:hFlipImAndLabel(img_human,label_human);
            end

            img_horse,label_horse,img_human,label_human=self:rotateImAndLabel(img_horse,label_horse,img_human,label_human,params.angles);
        end

        -- subtract the mean
        for i=1,img_horse:size()[1] do
            img_horse[i]:csub(params.mean[i])
        end

        for i=1,img_human:size()[1] do
            img_human[i]:csub(params.mean[i])
        end

        return img_horse,img_human,label_horse,label_human

    end


    function data:addTrainingData(training_set_horse,training_set_human,batch_size,lines_horse,lines_human,start_idx_horse,params)
        
        local list_idx=start_idx_horse;
        local list_size=#lines_horse;
        -- print (list_size,lines_horse[1])

        local curr_idx=1;

        while curr_idx<= batch_size do
            -- count=count+1;
            -- add mod length here to make the batch size rotate
            -- print ('curr_idx',curr_idx,'list_idx ',list_idx);

            local img_path_horse=lines_horse[list_idx][1];
            local label_path_horse=lines_horse[list_idx][2];
            
            local img_path_human=lines_human[list_idx][1];
            local label_path_human=lines_human[list_idx][2];
                        
            -- print (lines_horse[list_idx]);
            
            local status_img_horse,img_horse=pcall(image.load,img_path_horse);
            local status_img_human,img_human=pcall(image.load,img_path_human);
            
            
            if status_img_horse and status_img_human then
                -- print (284)
                -- print (img:size(),mask:size())
                local label_horse=npy4th.loadnpy(label_path_horse):double();
                local label_human=npy4th.loadnpy(label_path_human):double();

                if img_horse:size()[1]==1 then
                    img_horse= torch.cat(img_horse,img_horse,1):cat(img_horse,1)
                end
                
                if img_human:size()[1]==1 then
                    img_human= torch.cat(img_human,img_human,1):cat(img_human,1)
                end

                img_horse,img_human,label_horse,label_human=self:processImAndLabel(img_horse,img_human,label_horse,label_human,params)

                training_set_horse.data[curr_idx]=img_horse:int();
                training_set_human.data[curr_idx]=img_human:int();
                
                training_set_horse.label[curr_idx]=label_horse;
                training_set_human.label[curr_idx]=label_human;
            else
                print ('PROBLEM READING INPUT');
                curr_idx=curr_idx-1;
            end
            
            list_idx=(list_idx%list_size)+1;
            curr_idx=curr_idx+1;
        end
        return list_idx;
    
    end

    function data:processImAndLabelNoIm(img_horse,org_size_human,label_horse,label_human,params)
        
        img_horse:mul(255);
        -- img_human:mul(255);
        
        local org_size_horse=img_horse:size();
        -- local org_size_human=img_human:size();

        local label_horse_org=label_horse:clone();
        local label_human_org=label_human:clone();

        
        assert (img_horse:size(2)==params.input_size[1])
        assert (img_horse:size(3)==params.input_size[2])

        -- img_horse = image.scale(img_horse,params.input_size[1],params.input_size[2]);
        -- img_human = image.scale(img_human,params.input_size[1],params.input_size[2]);
        

        label_horse[{{},1}]=label_horse[{{},1}]/org_size_horse[3]*params.input_size[1];
        label_horse[{{},1}]=(label_horse[{{},1}]/params.input_size[1]*2)-1;
        label_horse[{{},2}]=label_horse[{{},2}]/org_size_horse[2]*params.input_size[2];
        label_horse[{{},2}]=(label_horse[{{},2}]/params.input_size[2]*2)-1;
        local temp = label_horse[{{},1}]:clone();
        label_horse[{{},1}]=label_horse[{{},2}]
        label_horse[{{},2}]=temp

        label_human[{{},1}]=label_human[{{},1}]/org_size_human[2]*params.input_size[1];
        label_human[{{},1}]=(label_human[{{},1}]/params.input_size[1]*2)-1;
        label_human[{{},2}]=label_human[{{},2}]/org_size_human[1]*params.input_size[2];
        label_human[{{},2}]=(label_human[{{},2}]/params.input_size[2]*2)-1;
        local temp = label_human[{{},1}]:clone();
        label_human[{{},1}]=label_human[{{},2}]
        label_human[{{},2}]=temp

        if (torch.max(label_horse)>=params.input_size[1]) then
            print ('PROBLEM horse');
            print (label_horse_org);
            print (label_horse);
            print (org_size_horse);
        end

        if (torch.max(label_human)>=params.input_size[1]) then
            print ('PROBLEM human');
            print (label_human_org);
            print (label_human);
            print (org_size_human);
        end
        
        -- flip or rotate
        if self.augmentation then
            local rand=math.random(2);
            
            if rand==1 then
                img_horse,label_horse = self:hFlipImAndLabel(img_horse,label_horse);
                _,label_human = self:hFlipImAndLabel(nil,label_human);
            end

            img_horse,label_horse,_,label_human=self:rotateImAndLabel(img_horse,label_horse,nil,label_human,params.angles);
        end

        -- subtract the mean
        for i=1,img_horse:size()[1] do
            img_horse[i]:csub(params.mean[i])
        end

        return img_horse,label_horse,label_human

        -- img_horse,label_horse,label_human=self:processImAndLabelNoIm(img_horse,img_path_human,label_horse,label_human,params)
    end

    function data:addTrainingDataNoIm(training_set_horse,training_set_human,batch_size,lines_horse,lines_human,start_idx_horse,params)
        
        local list_idx=start_idx_horse;
        local list_size=#lines_horse;
        -- print (list_size,lines_horse[1])

        local curr_idx=1;

        while curr_idx<= batch_size do
            -- count=count+1;
            -- add mod length here to make the batch size rotate
            -- print ('curr_idx',curr_idx,'list_idx ',list_idx);

            local img_path_horse=lines_horse[list_idx][1];
            local label_path_horse=lines_horse[list_idx][2];
            
            local img_path_human={lines_human[list_idx][2],lines_human[list_idx][3]};
            local label_path_human=lines_human[list_idx][1];
                        
            -- print (lines_horse[list_idx]);
            
            local status_img_horse,img_horse=pcall(image.load,img_path_horse);
            
            -- local status_img_human,img_human=pcall(image.load,img_path_human);
            
            
            if status_img_horse then
                -- print (284)
                -- print (img:size(),mask:size())
                local label_horse=npy4th.loadnpy(label_path_horse):double();
                local label_human=npy4th.loadnpy(label_path_human):double();

                if img_horse:size()[1]==1 then
                    img_horse= torch.cat(img_horse,img_horse,1):cat(img_horse,1)
                end
                
                img_horse,label_horse,label_human=self:processImAndLabelNoIm(img_horse,img_path_human,label_horse,label_human,params)

                training_set_horse.data[curr_idx]=img_horse:int();
                -- training_set_human.data[curr_idx]=img_human:int();
                
                training_set_horse.label[curr_idx]=label_horse;
                training_set_human.label[curr_idx]=label_human;
            else
                print ('PROBLEM READING INPUT');
                curr_idx=curr_idx-1;
            end
            
            list_idx=(list_idx%list_size)+1;
            curr_idx=curr_idx+1;
        end
        return list_idx;
    
    end
    
end

return data