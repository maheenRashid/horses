do  
    local TPS_Helper = torch.class('TPS_Helper')

    function TPS_Helper:__init()
        
    end

    function TPS_Helper:getLoss(pred_output,gt_output,ind_mean)
		local loss=torch.pow(pred_output-gt_output,2);
		-- print (ind_mean);
		if ind_mean then
			loss=torch.mean(loss:view(loss:size(1),loss:size(2)*loss:size(3)*loss:size(4)),2);
		else
			loss=torch.mean(loss);
		end
		return loss;
    end

	function TPS_Helper:getTransformedLandMarkPoints(labels_all,out_grids,tableFlag)
		local t_pts_all;
		if tableFlag then
			t_pts_all={};
			for idx=1,#labels_all do
				t_pts_all[idx]=labels_all[idx]:clone();
			end
		else
			t_pts_all=labels_all:clone();
		end

		for i=1,out_grids:size(1) do
			local grid_curr=out_grids[i];
			local labels=labels_all[i];
			local end_label;
			
			if tableFlag then
				end_label=labels:size(2);
			else
				end_label=labels:size(1);
			end
			-- print ('labels',labels)

			for label_idx=1,end_label do

				local label_curr
				if tableFlag then
					label_curr=labels[{{1,2},label_idx}];
					-- print ('label_curr',label_idx,label_curr);
				else
					label_curr=labels[{label_idx,{1,2}}];
				end
				label_curr=label_curr:view(1,1,2);
				label_curr=torch.repeatTensor(label_curr,grid_curr:size(1),grid_curr:size(2),1);
				
				local dist=torch.sum(torch.pow(grid_curr-label_curr,2),3);
				dist=dist:view(dist:size(1),dist:size(2));
				local idx=torch.find(dist:eq(torch.min(dist)),1)[1]
				local row = math.ceil(idx/dist:size(2));
				local col = idx%dist:size(2);
				if col==0 then
					col=dist:size(2);
				end
				
				if tableFlag then
					t_pts_all[i][1][label_idx]=row;
					t_pts_all[i][2][label_idx]=col;
					-- print (row,col,t_pts_all[i]);
				else
					t_pts_all[i][label_idx][1]=row;
					t_pts_all[i][label_idx][2]=col;
				end
				
			end
		end
		return t_pts_all;
	end


end    

return TPS_Helper;