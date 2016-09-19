
do  
    local Visualize = torch.class('Visualize')

    function Visualize:__init()
        
    end

    function Visualize:plotPoint(im,point,pointSize,color_curr)
        if x>im:size(3) or y>im:size(2) then
            return im;
        end
        local point=torch.Tensor(point);
        local starts=torch.round(point-pointSize/2);
        local ends=torch.round(point+pointSize/2);

        for x_curr=math.max(1,starts[1]),math.min(im:size(3),ends[1]) do
            for y_curr=math.max(1,starts[2]),math.min(im:size(2),ends[2]) do
                for idx_rgb=1,3 do
                    im[idx_rgb][y_curr][x_curr]=color_curr[idx_rgb]/255;            
                end
            end
        end

        return im;

    end

    function Visualize:saveBatchImagesWithKeypoints(im_all,pts_all,file_info,mean,scale,colors,pointSize)
        if mean~=nil then
            for idx_rgb=1,3 do
                im_all[{{},idx_rgb,{},{}}]=im_all[{{},idx_rgb,{},{}}]+mean[idx_rgb];
            end
        end

        for idx_im=1,im_all:size(1) do
            local out_file= file_info[1]..idx_im..file_info[2]
            local im_new=self:drawKeyPoints(im_all[idx_im]:clone(),pts_all[idx_im],scale,colors,pointSize);
            image.save(out_file,im_new);
        end
    end
    

    function Visualize:drawKeyPoints(im,keypoints,scale,colors,pointSize)
        assert (#keypoints:size()==2);
        assert (keypoints:size(1)==2);
        assert (im:size(1)==3);
        
        if not pointSize then
            pointSize=math.max(torch.round(math.min(im:size(2),im:size(3))*0.02),1);
        end
        
        if not colors then
            colors={{255,0,0}};
        end

        if torch.max(im)>1 then
            im=im/255;
        end

        if scale~=nil then
            assert (#scale==2);
            keypoints=keypoints-scale[1];
            keypoints=torch.div(keypoints,scale[2]-scale[1]);
            keypoints[{1,{}}]=keypoints[{1,{}}]*im:size(2);
            keypoints[{2,{}}]=keypoints[{2,{}}]*im:size(3);
        end

        for label_idx=1,keypoints:size(2) do
            x=keypoints[2][label_idx];
            y=keypoints[1][label_idx];
            local color_curr=colors[math.min(#colors,label_idx)];
            im=self:plotPoint(im,{x,y},pointSize,color_curr);
        end

        return im;
    end

end    

return Visualize;