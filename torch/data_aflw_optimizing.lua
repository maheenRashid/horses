
do  
    local data = torch.class('data')

    function data:__init(args)
        -- print (args);
        self.file_path=args.file_path;
        self.batch_size=args.batch_size;
        self.mean_file=args.mean_file;
        self.std_file=args.std_file;
        self.limit=args.limit;
        self.augmentation=args.augmentation;
        self.rotFix=args.rotFix;
        print ('self.augmentation',self.augmentation);

        
        self.start_idx_horse=1;
        self.params={input_size={40,40},
                    labels_shape={5,3},
                    angles={-10,-5,0,5,10}};

        if args.input_size then
            self.params.input_size=args.input_size
        end

        if args.imagenet_mean then
           self.params.imagenet_mean={122,117,104};
        end 
        
        self.mean_im=image.load(self.mean_file)*255;
        self.std_im=image.load(self.std_file)*255;



        self.training_set={};
        
        self.lines_horse=self:readDataFile(self.file_path);
        
        if self.augmentation then
            self.lines_horse =self:shuffleLines(self.lines_horse);
        end



        if self.limit~=nil then
            local lines_horse=self.lines_horse;
            self.lines_horse={};

            for i=1,self.limit do
                self.lines_horse[#self.lines_horse+1]=lines_horse[i];
            end
        end
        print (#self.lines_horse);
        self.images_all,self.images_index=self:initializeImages();
    end

    function data:initializeImages()
        local images_index={};
        local num_images=0;
        for idx_line,line_curr in pairs(self.lines_horse) do
            line_curr=line_curr[1];
            if not images_index[line_curr] then
                num_images=num_images+1;
                images_index[line_curr]=num_images;
            end
        end

        local images_all=torch.zeros(num_images,3,self.params.input_size[1],self.params.input_size[2]);
        local problem_idx={};
        local num_problem=0;
        for img_path,img_idx in pairs(images_index) do
            local status_img_horse,img_horse=pcall(image.load,img_path);
            if status_img_horse then
                if img_horse:size(2)~=self.params.input_size[1] or img_horse:size(3)~=self.params.input_size[2] then
                    img_horse=image.scale(img_horse,self.params.input_size[1],self.params.input_size[2]);
                end
                images_all[img_idx]=img_horse;
            else
                num_problem=num_problem+1;
                problem_idx[img_idx]=num_problem;
            end
        end

        -- print (problem_idx,table.getn{problem_idx});
        -- problem_idx[1]=1;
        -- problem_idx[5]=1;
        -- problem_idx[10]=1;
        -- num_problem=3;
        -- print (problem_idx,table.getn{problem_idx});
        print ('num_problem',num_problem,'num_images',num_images);

        if num_problem>0 then
            local images_all_old=images_all:clone();
            local images_index_old=images_index;
            -- local lines_horse_new={};
            images_index={};

            local idx_keep={};
            images_all=torch.zeros(num_images-num_problem,3,self.params.input_size[1],self.params.input_size[2]);

            local i=0;
            for img_path_curr,img_path_curr_index in pairs(images_index_old) do
                if not problem_idx[img_path_curr_index] then
                    i=i+1;

                    idx_keep[#idx_keep+1]=img_path_curr_index;
                    
                    images_index[img_path_curr]=i;
                    images_all[i]=images_all_old[img_path_curr_index];
                    
                    -- lines_horse_new[i]=self.lines_horse[img_path_curr_index];
                end
            end

            for img_path_curr,index_new in pairs(images_index) do
                local index_old=images_index_old[img_path_curr];
                local im_new=images_all[index_new];
                local im_old=images_all_old[index_old];
                -- print ('asserting',img_path_curr,index_new,index_old);
                -- assert (torch.eq(im_new,im_old));
            end
            -- print ('(i,images_all:size(),images_all_old:size())');
            -- print (i,images_all:size(),images_all_old:size());
            -- print ('(problem_idx)');
            -- print (problem_idx);
            -- print ('(images_index)');
            -- print (images_index);
            -- print ('(images_index_old)');
            -- print (images_index_old);
            -- print (self.lines_horse);
            
            -- self.lines_horse=lines_horse_new;
            
        end
        
        print ('mean_subbing');        
        local mean_im_all=self.mean_im:view(1,self.mean_im:size(1),self.mean_im:size(2),self.mean_im:size(3)):clone();
        mean_im_all=torch.repeatTensor(mean_im_all,images_all:size(1),1,1,1)
        local std_im_all=self.std_im:view(1,self.std_im:size(1),self.std_im:size(2),self.std_im:size(3)):clone();
        std_im_all=torch.repeatTensor(std_im_all,images_all:size(1),1,1,1)

        images_all:mul(255);
        if self.params.imagenet_mean then
            for i=1,images_all:size(2) do
                images_all[{{},i,{},{}}]:csub(self.params.imagenet_mean[i])
            end
        else
            images_all=torch.cdiv((images_all-mean_im_all),std_im_all);
        end
        print ('done');
        -- self.images_all=images_all:clone();
        -- self.images_index=images_index;
        return images_all,images_index;

    end



    function data:shuffleLines(lines)
        local x=lines;
        local len=#lines;

        local shuffle = torch.randperm(len)
        
        local lines2={};
        for idx=1,len do
            lines2[idx]=x[shuffle[idx]];
        end
        return lines2;
    end

    function data:getTrainingData()
        local start_idx_horse_before = self.start_idx_horse

        self.training_set.data=torch.zeros(self.batch_size,3,self.params.input_size[1]
            ,self.params.input_size[2]);
        self.training_set.label=torch.zeros(self.batch_size,self.params.labels_shape[1],self.params.labels_shape[2]);
        self.training_set.input={};
        
        -- t=os.clock()
        self.start_idx_horse=self:addTrainingData(self.training_set,self.batch_size,
            self.lines_horse,self.start_idx_horse,self.params,self.images_all,self.images_index)    
        -- print ('addTrainingData:',os.clock()-t);

        if self.start_idx_horse<start_idx_horse_before and self.augmentation then
            print ('shuffling data'..self.start_idx_horse..' '..start_idx_horse_before )
            self.lines_horse=self:shuffleLines(self.lines_horse);
        end
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
        
        return im,label;
    end

    function data:rotateImAndLabel(img_horse,label_horse,angles)
        
        local isValid = false;
        local img_horse_org=img_horse:clone();
        local label_horse_org=label_horse:clone();
        local iter=0;
        
        while not isValid do
            isValid = true;
            label_horse= label_horse_org:clone();

            local rand=math.random(#angles);
            local angle=math.rad(angles[rand]);
            -- print ('no rotfix');
            img_horse=image.rotate(img_horse_org,angle,"bilinear");

            if self.rotFix then
                local rot = torch.ones(img_horse_org:size());
                rot=image.rotate(rot,angle,"simple");
                img_horse[rot:eq(0)]=img_horse_org[rot:eq(0)];
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
                    if torch.all(label_horse[i]:ge(-1)) and torch.all(label_horse[i]:le(1)) then
                        isValid=true;
                    else
                        isValid=false;
                    end
                end

                if not isValid then
                    break;
                end

            end
            
            iter=iter+1;
            if iter==100 then
                print ('BREAKING rotation');
                label_horse=label_horse_org;
                img_horse=img_horse_org;
                break;
            end
        end

        return img_horse,label_horse
    end

    function data:processImAndLabel(img_horse,label_horse,params)
        
        -- img_horse:mul(255);
        
        local org_size_horse=img_horse:size();
        local label_horse_org=label_horse:clone();
        
        -- if img_horse:size(2)~=params.input_size[1] then 
        --     img_horse = image.scale(img_horse,params.input_size[1],params.input_size[2]);
        -- end
        label_horse[{{},1}]=label_horse[{{},1}]/org_size_horse[3]*params.input_size[1];
        label_horse[{{},1}]=(label_horse[{{},1}]/params.input_size[1]*2)-1;
        label_horse[{{},2}]=label_horse[{{},2}]/org_size_horse[2]*params.input_size[2];
        label_horse[{{},2}]=(label_horse[{{},2}]/params.input_size[2]*2)-1;
        
        local temp = label_horse[{{},1}]:clone();
        label_horse[{{},1}]=label_horse[{{},2}]
        label_horse[{{},2}]=temp

        if (torch.max(label_horse)>=params.input_size[1]) then
            print ('PROBLEM horse');
            print (label_horse_org);
            print (label_horse);
            print (org_size_horse);
        end

        -- if self.params.imagenet_mean then
        --     for i=1,img_horse:size()[1] do
        --         img_horse[i]:csub(params.imagenet_mean[i])
        --     end
        -- else
        --     img_horse=torch.cdiv((img_horse-self.mean_im),self.std_im);
        -- end
        
        -- flip or rotate
        if self.augmentation then
            local rand=math.random(2);
            if rand==1 then
                img_horse,label_horse = self:hFlipImAndLabel(img_horse,label_horse);
            end
            img_horse,label_horse=self:rotateImAndLabel(img_horse,label_horse,params.angles);
        end

        return img_horse,label_horse
    end

    function data:addTrainingData(training_set,batch_size,lines_horse,start_idx_horse,params,images_all,images_index)
        local list_idx=start_idx_horse;
        local list_size=#lines_horse;
        local curr_idx=1;
        -- local images_all=self.images_all:clone();
        while curr_idx<= batch_size do
            local img_path_horse=lines_horse[list_idx][1];
            local label_path_horse=lines_horse[list_idx][2];
            
            -- t=os.clock()
            -- local status_img_horse,img_horse=pcall(image.load,img_path_horse);

            -- print ('pcall:',os.clock()-t);

            if images_index[img_path_horse] then
                -- t=os.clock()
                local img_horse=images_all[images_index[img_path_horse]]:clone();
                local label_horse=npy4th.loadnpy(label_path_horse):double();
                -- print ('label_horse:',os.clock()-t);

                if img_horse:size()[1]==1 then
                    img_horse= torch.cat(img_horse,img_horse,1):cat(img_horse,1)
                end

                -- t=os.clock()
                img_horse,label_horse=self:processImAndLabel(img_horse,label_horse,params)
                -- print ('processImAndLabel:',os.clock()-t);
                
                -- t=os.clock()
                training_set.data[curr_idx]=img_horse:int():clone();
                training_set.label[curr_idx]=label_horse:clone();
                training_set.input[curr_idx]=img_path_horse;
                collectgarbage()
                -- print ('training_set populate:',os.clock()-t);
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