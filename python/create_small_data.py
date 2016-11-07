import util;
import os;
import visualize;
import numpy as np;
import time;


dir_server='/home/SSD3/maheen-data';
click_str='http://vision1.idav.ucdavis.edu:1000';
    


def getNumNeighbors(file_horse):
    lines=util.readLinesFromFile(file_horse);
    lines=np.array(lines);
    uni_lines=np.unique(lines);
    counts=np.zeros(uni_lines.shape);
    for idx_uni_curr,uni_curr in enumerate(uni_lines):
        if idx_uni_curr%500==0:
            print idx_uni_curr;
        counts[idx_uni_curr]=np.sum(lines==uni_curr);
    return uni_lines,counts;

def writeSmallDatasetFile(out_file_pre,horse_data,num_neighbor,
                          num_data,in_file_horse,in_file_face,in_file_face_noIm,post_tags=None):
    if post_tags is None:
        post_tags=['_horse.txt','_face.txt','_face_noIm.txt'];
        
    in_files=[in_file_horse,in_file_face,in_file_face_noIm];
    
    data_org=util.readLinesFromFile(in_file_horse);
    data_org=np.array(data_org);
    idx_keep_all=[];
    print horse_data.shape
    horse_data=horse_data[:num_data];
    for horse_curr in horse_data:
        idx_curr=np.where(data_org==horse_curr)[0];
        idx_curr=np.sort(idx_curr)
        idx_keep=idx_curr[:num_neighbor];
        idx_keep_all=idx_keep_all+list(idx_keep);
#         print num_data,idx_keep
        
    idx_keep_all=np.array(idx_keep_all);
    print idx_keep_all.shape
    files_to_return=[];
    for idx_in_file,in_file in enumerate(in_files):
        out_file_curr=out_file_pre+post_tags[idx_in_file];
        if idx_in_file==0:
            data_keep=data_org[idx_keep_all];
        else:
            data_curr=util.readLinesFromFile(in_file);
            data_curr=np.array(data_curr);
            data_keep=data_curr[idx_keep_all];
        util.writeFile(out_file_curr,data_keep);
        files_to_return.append(out_file_curr);
    
    return files_to_return;
        
def main():
    dir_server='/home/SSD3/maheen-data';
    click_str='http://vision1.idav.ucdavis.edu:1000';
    
    out_dir_meta_data='/home/SSD3/maheen-data/horse_project/data_check';
    dir_neighbors='/home/SSD3/maheen-data/horse_project/neighbor_data';
    matches_file=os.path.join(dir_neighbors,'horse_trainImageList_2_data_100_neigbors.txt');
    
    out_dir_debug=os.path.join(dir_neighbors,'debug');
    out_dir_breakdowns=os.path.join(dir_neighbors,'small_datasets');
    file_pre='matches';
    util.mkdir(out_dir_breakdowns);
    util.mkdir(out_dir_debug);
    
    out_file_counts=os.path.join(out_dir_debug,'counts.npz');
    out_file_dist=os.path.join(out_dir_debug,'counts_dist.png');
    
    out_dir_meta_horse = os.path.join(out_dir_meta_data,'horse');
    out_dir_meta_face = os.path.join(out_dir_meta_data,'aflw');
    
    num_neighbors=100;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_allKP.txt');
    out_file_face_noIm=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_allKP_noIm.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_train_allKP.txt');
    
    old_horse_file='/home/SSD3/maheen-data/horse_project/data_check/horse/matches_5_train_allKP_minLoss_clean_full.txt';
    old_data=util.readLinesFromFile(old_horse_file);
    old_data=np.array(old_data);
    old_data=np.unique(old_data);

#     new_data,counts=getNumNeighbors(out_file_horse);
#     np.savez(out_file_counts,new_data=new_data,counts=counts);
    
    data=np.load(out_file_counts);
    print data.files;
    new_data=data['new_data'];
    counts=data['counts'];
    num_data=range(500,len(old_data),500);
    num_data[-1]=len(old_data);
    
    num_neighbors=range(5,25,5);
    np.random.shuffle(old_data);
    for num_neighbor in num_neighbors:
        for num_data_curr in num_data:
            file_curr=file_pre+'_'+str(num_neighbor)+'_'+str(num_data_curr);
            out_file_pre=os.path.join(out_dir_breakdowns,file_curr);
            files=writeSmallDatasetFile(out_file_pre,old_data,num_neighbor,num_data_curr,out_file_horse,\
                                        out_file_face,out_file_face_noIm);
            for file_curr in files:
                print file_curr;
    


    
def writeMinLossFile(out_file_pre,post_tags,minloss_post,old_horse_file,old_human_file,old_human_file_noIm):
    new_files=[out_file_pre+post_tag_curr for post_tag_curr in post_tags];
    
    old_data=util.readLinesFromFile(old_horse_file);
    old_data=np.array(old_data);
    
    new_data=util.readLinesFromFile(new_files[0]);
    new_data=np.array(new_data);
    new_data_uni=np.unique(new_data);
    bin_keep=np.in1d(old_data,new_data_uni);
#     print bin_keep.shape,sum(bin_keep);
    old_files=[old_horse_file,old_human_file,old_human_file_noIm];
    new_files_write=[file_curr[:file_curr.rindex('.')]+minloss_post for file_curr in new_files];
    for old_file_curr,new_file_curr,new_file_org in zip(old_files,new_files_write,new_files):
        data_curr=util.readLinesFromFile(old_file_curr);
        data_curr=np.array(data_curr);
        data_keep=data_curr[bin_keep];
        print old_file_curr,new_file_curr,len(data_keep);
        print data_keep[0];
        new_file_org_data=util.readLinesFromFile(new_file_org);
        new_file_org_data=np.array(new_file_org_data);
        bin_check=np.in1d(data_keep,new_file_org_data);
        print sum(bin_check),data_keep.shape[0];
        assert sum(bin_check)==data_keep.shape[0];
#         util.writeFile(new_file_curr,data_keep);
    
def getCommandFaceTest(path_to_th,out_dir,file_curr,batch_size=100):
    command=['th',path_to_th];
    command=command+['-val_data_path',file_curr];
    command=command+['-outDir',out_dir];
    command=command+['-batchSize',str(batch_size)];
    amount_data=len(util.readLinesFromFile(file_curr));
    num_iterations=amount_data/batch_size;
    if amount_data%batch_size!=0:
        num_iterations=num_iterations+1;
    command=command+['-iterations',str(num_iterations)];
    command=' '.join(command);
    return command;

def getCommandFaceTrain(path_to_th,out_dir,file_curr):
    command=['th',path_to_th];
    command=command+['-data_path',file_curr];
    command=command+['-outDir',out_dir];
    command=' '.join(command);
    return command;

def getCommandFull2LossTrain(path_to_th,out_dir,horse_data_path,human_data_path):
    command=['th',path_to_th];
    command=command+['-outDir',out_dir];
    command=command+['-horse_data_path',horse_data_path];
    command=command+['-human_data_path',human_data_path];
    command=' '.join(command);
    return command;


def writeMinLossFileLossData(out_file_pre,post_tags,minloss_post,loss_file):
    new_files=[out_file_pre+post_tag_curr for post_tag_curr in post_tags];
    horse_data=util.readLinesFromFile(new_files[0]);
    horse_data=np.array(horse_data);
    horse_data_uni=np.unique(horse_data);
    face_data=util.readLinesFromFile(new_files[1]);
    face_data_noIm=util.readLinesFromFile(new_files[2]);
    assert len(face_data)==len(face_data_noIm);
    
    loss_all=np.load(loss_file);
    loss_all=loss_all[:len(face_data)];
    assert loss_all.shape[0]==len(face_data);
    
    new_data=[[],[],[]];
    for idx_curr,horse_curr in enumerate(horse_data_uni):
        idx_rel=np.where(horse_data==horse_curr)[0];
        loss_rel=loss_all[idx_rel];
        min_idx=np.argmin(loss_rel);
        min_idx_big=idx_rel[min_idx];
        assert loss_rel[min_idx]==loss_all[min_idx_big];
        new_data[0].append(horse_curr);
        new_data[1].append(face_data[min_idx_big]);
        new_data[2].append(face_data_noIm[min_idx_big]);
  
    new_files_out=[new_file_curr[:new_file_curr.rindex('.')]+minloss_post for new_file_curr in new_files];
    for new_file_to_write,data_to_write in zip(new_files_out,new_data):
        print new_file_to_write,len(data_to_write);
        util.writeFile(new_file_to_write,data_to_write);

def getFilePres():
    num_neighbors=range(5,10,5);
    num_data=range(500,3500,500);
    num_data=num_data+[3531]
    post_tags='_horse_minloss';
    file_pre='matches';
    file_pres=[file_pre+'_'+str(num_neighbors[0])+'_'+str(num_data_curr)+post_tags for num_data_curr in num_data];
    return file_pres,num_data

def getLogFileLoss(log_file):
    data_curr=util.readLinesFromFile(log_file);
    data_curr=data_curr[-1];
    data_curr=data_curr.split(' ')[-1].strip('"');
    data_curr=float(data_curr);
    return data_curr;

def getMinLoss(dir_meta,loss_dir_pre,loss_dir_posts,log_file):
    loss_all=[];
    for loss_dir_post in loss_dir_posts:
        file_curr=os.path.join(dir_meta,loss_dir_pre+loss_dir_post,log_file);
        if os.path.exists(file_curr):
            loss=getLogFileLoss(file_curr);
        else:
            loss=float('inf');
        loss_all.append(loss);
        
    loss_all=np.array(loss_all);
    min_idx=np.argmin(loss_all);
    min_loss=loss_all[min_idx];
    return min_loss,loss_dir_posts[min_idx];


def script_getMinLoss():
    dir_metas=['/home/SSD3/maheen-data/horse_project/full_system_small_data',
               '/home/SSD3/maheen-data/horse_project/face_baselines_small_data'];
    file_pres,num_data=getFilePres();
    loss_dir_pre='test_images_';
    loss_dir_posts=[str(num_curr) for num_curr in range(1680,9000,1680)];
    log_file='log_test.txt';
#     num_data=
    losses_all=[];
    losses_all_end=[];
    min_loss_iter_all=[];
    for dir_meta in dir_metas:
        loss_curr=[];
        loss_end=[];
        min_loss_iter=[];
        for file_pre in file_pres:
            min_loss,min_loss_post=getMinLoss(os.path.join(dir_meta,file_pre),loss_dir_pre,loss_dir_posts,log_file);
#             print loss_dir_posts
            loss_end_curr,min_loss_post_end=getMinLoss(os.path.join(dir_meta,file_pre),loss_dir_pre,['8400'],log_file);
            loss_curr.append(min_loss);
            loss_end.append(loss_end);
            min_loss_iter.append(min_loss_post);
            print min_loss,min_loss_post,loss_end_curr,min_loss_post_end;

        losses_all.append(loss_curr);
        losses_all_end.append(loss_end)
        min_loss_iter_all.append(min_loss_iter);
        
    out_file=os.path.join(dir_metas[0],'comparison_best.png');
    print len(file_pres),len(losses_all[0]),len(losses_all[1]),len(num_data)
    xAndYs=[(num_data,losses_all[0]),(num_data,losses_all[1]),\
           (num_data,losses_all_end[0]),(num_data,losses_all_end[1])];
    legend_entries=['Ours Best','Baseline Best','Ours 8400','Baseline 8400'];
    visualize.plotSimple(xAndYs,out_file,title='',xlabel='Training Data',\
                         ylabel='Average Euclidean Distance',legend_entries=legend_entries);
    print out_file.replace(dir_server,click_str)
    
    
print 'hello'
script_getMinLoss();
print 'hello post'