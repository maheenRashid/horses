import util;
import os;
import numpy as np;
import scipy.misc;
import multiprocessing;
import visualize;
import Image;


def script_saveTrainTxt():
    face_data='/home/SSD3/maheen-data/face_data';
    horse_data='/home/SSD3/maheen-data/horse_data';
    new_face_data = '/home/SSD3/maheen-data/aflw_data';
    txt_file='data_list.txt';

    for data_type,ext in [(horse_data,None),(new_face_data,'.jpg'),(face_data,None)]:
        im_path_meta=os.path.join(data_type,'im');
        npy_path_meta=os.path.join(data_type,'npy');
        out_file_train=os.path.join(data_type,'train.txt');
        npy_file=os.path.join(npy_path_meta,txt_file);
        
        makeTrainDataFile(npy_file,im_path_meta,npy_path_meta,out_file_train,ext)
    
def saveAflwBBoxIm():
    path_txt='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    # path_pre='/home/laoreja/finetune-deep-landmark/dataset'
    path_im,bbox,anno_points=parseAnnoFile(path_txt);
    print len(path_im);
    print len(set(path_im));


    out_dir_im = '/home/SSD3/maheen-data/aflw_data/im';
    out_dir_npy = '/home/SSD3/maheen-data/aflw_data/npy';
    
    util.mkdir(out_dir_im);
    util.mkdir(out_dir_npy);

    args = [];
    args_bbox_npy=[];

    for idx,path_im_curr,bbox_curr,key_pts in zip(range(len(path_im)),path_im,bbox,anno_points):    
        path_curr,file_name=os.path.split(path_im_curr);
        
        path_curr=path_curr.split('/');
        path_pre_curr=path_curr[-1];
        
        file_just_name=file_name[:file_name.rindex('.')];
        file_name=file_just_name+'_'+str(idx);

        out_dir_curr=os.path.join(out_dir_im,path_pre_curr);
        out_dir_npy_curr=os.path.join(out_dir_npy,path_pre_curr);

        util.mkdir(out_dir_curr);
        util.mkdir(out_dir_npy_curr);

        out_file=os.path.join(out_dir_curr,file_name+'.jpg');
        if not os.path.exists(out_file):
            args.append((path_im_curr,out_file,bbox_curr,idx));

        out_file_npy=os.path.join(out_dir_npy_curr,file_name+'.npy');
        # args_bbox_npy.append((bbox_curr,key_pts,out_file_npy,idx));
        if not os.path.exists(out_file_npy):
            args_bbox_npy.append((bbox_curr,key_pts,out_file_npy,idx));

    print len(args);
    print len(args_bbox_npy);
    # print args;
    # p=multiprocessing.Pool(multiprocessing.cpu_count());
    # p.map(saveBBoxIm,args);
    # p.map(saveBBoxNpy,args_bbox_npy);

def saveHorseBBoxIm():
    # path_txt='/home/laoreja/finetune-deep-landmark/dataset/train/trainImageList.txt';
    path_txt='/home/laoreja/finetune-deep-landmark/dataset/train/trainImageList_2.txt';
    # path_pre='/home/laoreja/finetune-deep-landmark/dataset'
    path_im,bbox,anno_points=parseAnnoFile(path_txt);

    out_dir_im = '/home/SSD3/maheen-data/horse_data/im';
    out_dir_npy = '/home/SSD3/maheen-data/horse_data/npy';
    args = [];
    args_bbox_npy=[];
    
    for idx,path_im_curr,bbox_curr,key_pts in zip(range(len(path_im)),path_im,bbox,anno_points):    
        path_curr,file_name=os.path.split(path_im_curr);
        path_curr=path_curr.split('/');
        if path_curr[-1]=='gxy':
            path_pre_curr=path_curr[-2];
        else:
            path_pre_curr=path_curr[-1];
        
        out_dir_curr=os.path.join(out_dir_im,path_pre_curr);
        out_dir_npy_curr=os.path.join(out_dir_npy,path_pre_curr);

        util.mkdir(out_dir_curr);
        util.mkdir(out_dir_npy_curr);

        out_file=os.path.join(out_dir_curr,file_name);
        args.append((path_im_curr,out_file,bbox_curr,idx));

        out_file_npy=os.path.join(out_dir_npy_curr,file_name[:file_name.rindex('.')]+'.npy');
        args_bbox_npy.append((bbox_curr,key_pts,out_file_npy,idx));
        
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(saveBBoxIm,args);
    p.map(saveBBoxNpy,args_bbox_npy);

def saveFaceBBoxIm():
    path_txt='/home/laoreja/deep-landmark-master/dataset/train/trainImageList.txt';
    path_pre='/home/laoreja/deep-landmark-master/dataset/train'
    path_im,bbox,anno_points=parseAnnoFile(path_txt,path_pre,face=True);

    out_dir_im = '/home/SSD3/maheen-data/face_data/im';
    out_dir_npy = '/home/SSD3/maheen-data/face_data/npy';
    args = [];
    args_bbox_npy=[];

    for idx,path_im_curr,bbox_curr,key_pts in zip(range(len(path_im)),path_im,bbox,anno_points):    
        path_curr,file_name=os.path.split(path_im_curr);
        path_curr=path_curr.split('/');
        path_pre_curr=path_curr[-1];
        
        out_dir_curr=os.path.join(out_dir_im,path_pre_curr);
        out_dir_npy_curr=os.path.join(out_dir_npy,path_pre_curr);

        util.mkdir(out_dir_curr);
        util.mkdir(out_dir_npy_curr);

        out_file=os.path.join(out_dir_curr,file_name);
        args.append((path_im_curr,out_file,bbox_curr,idx));

        out_file_npy=os.path.join(out_dir_npy_curr,file_name[:file_name.rindex('.')]+'.npy');
        args_bbox_npy.append((bbox_curr,key_pts,out_file_npy,idx));
        # break;

    print len(args_bbox_npy)
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    # p.map(saveBBoxIm,args);
    p.map(saveBBoxNpy,args_bbox_npy);


def saveBBoxNpy((bbox,key_pts,out_file,idx)):
    print idx;
    min_pts=np.array([bbox[0],bbox[2]])
    key_pts=np.array(key_pts);
    # print key_pts
    if key_pts.shape[1]>2:
        for idx in range(key_pts.shape[0]):
            if key_pts[idx,2]>0:
                key_pts[idx,:2]=key_pts[idx,:2]-min_pts;
    else:
        key_pts=key_pts-min_pts
        key_pts=np.hstack((key_pts,np.ones((key_pts.shape[0],1))))
    # print key_pts
    np.save(out_file,key_pts);

def saveDataTxtFiles():
    # horse_data='/home/SSD3/maheen-data/face_data/npy';
    horse_data='/home/SSD3/maheen-data/horse_data/npy';
    # horse_data = '/home/SSD3/maheen-data/aflw_data/npy';

    folders=[os.path.join(horse_data,folder_curr) for folder_curr in os.listdir(horse_data) if os.path.isdir(os.path.join(horse_data,folder_curr))];
    file_list=[];
    to_del=[];
    for folder_curr in folders:
        file_list_curr = util.getFilesInFolder(folder_curr,'.npy');
        
        if len(file_list_curr)==0:
            to_del.append(folder_curr);

        file_list.extend(file_list_curr);
    
    for folder_curr in to_del:
        shutil.rmtree(folder_curr);

    out_file=os.path.join(horse_data,'data_list.txt');
    util.writeFile(out_file,file_list);

def addDim((idx,in_file,out_file)):
    print idx;
    data=np.load(in_file);
    data=np.hstack((data,np.ones((data.shape[0],1))));
    np.save(out_file,data);

def addDimFaceData():
    face_data_dir='/home/SSD3/maheen-data/face_data/npy'
    out_dir='/home/SSD3/maheen-data/face_data/npy_dimAdd'
    util.mkdir(out_dir);

    face_data_file=os.path.join(face_data_dir,'data_list.txt');
    in_files=util.readLinesFromFile(face_data_file);

    args=[];
    for idx,in_file in enumerate(in_files):
        out_file=in_file.replace(face_data_dir,out_dir);
        
        if os.path.exists(out_file):
            continue;
        
        folder_curr=out_file[:out_file.rindex('/')];
        util.mkdir(folder_curr);
        args.append((idx,in_file,out_file))

    print len(args);
    # for arg in args:
    #   addDim(arg);
    #   break;
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(addDim,args);
    # rename /npy to npy_dimMissing. rename /npy_dimAdd to /npy

def main():


    


    return
    # saveHorseBBoxIm()
    # saveDataTxtFiles()

    # return
    
    matches='/home/laoreja/data/knn_res_new/5_points_list.txt';
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';

    face_data_list_file='/home/SSD3/maheen-data/aflw_data/npy/data_list.txt';
    path_pre_replace=['/home/SSD3/laoreja-data/aflwd/aflw/data/flickr','/home/SSD3/maheen-data/aflw_data/npy']
    face_data_list=util.readLinesFromFile(face_data_list_file);

    new_match_file='/home/SSD3/maheen-data/aflw_data/match_5.txt';

    face_data=util.readLinesFromFile(face_data_file);
    face_data=[' '.join(line_curr.split(' ')[:5]) for line_curr in face_data];
    
    matches_list=util.readLinesFromFile(matches);
    matches_split=[match_curr.split(' ') for match_curr in matches_list];
    horse_list=[match_split[0] for match_split in matches_split];

    matches_new=[];
    missing_files=[];
    for match_split in matches_split:
        match_split_new=[match_split[0]];
        continue_flag=False;
        for matches_idx in range(5):
            start_idx=(matches_idx*5)+1;
            end_idx=start_idx+5;
            match_curr=match_split[start_idx:end_idx];
            match_curr=' '.join(match_curr);
            
            if match_curr in face_data:
                idx_curr=face_data.index(match_curr)    
            else:
                missing_files.append(match_curr);
                continue_flag=True;
                break;
            file_match_curr=match_curr.split(' ')[0];
            file_match_curr=file_match_curr.replace(path_pre_replace[0],path_pre_replace[1]);
            file_match_curr=file_match_curr[:file_match_curr.rindex('.')]+'_'+str(idx_curr)+'.npy';
            # print file_match_curr;
            
            match_split_new.append(file_match_curr);
            assert os.path.exists(file_match_curr);
        if continue_flag:
            continue;
        matches_new.append(' '.join(match_split_new));

    print len(matches_new),len(matches_split),len(matches_split)-len(matches_new);
    print len(missing_files);
    util.writeFile(new_match_file,matches_new);




        # print idx_curr,face_data[idx_curr],match_curr;





    # saveDataTxtFiles()
    # saveAflwBBoxIm()

    # saveHorseBBoxIm()
    
    # out_dir_scratch = '/home/SSD3/maheen-data/scratch';
    # dir_meta='/home/SSD3/maheen-data/face_data';
    # im_dir_meta=os.path.join(dir_meta,'im');
    # npy_dir_meta=os.path.join(dir_meta,'npy');
    # file_pre='lfw_5590/Aaron_Eckhart_0001'
    # file_im=os.path.join(im_dir_meta,file_pre+'.jpg');
    # file_npy=os.path.join(npy_dir_meta,file_pre+'.npy');
    # out_file=os.path.join(out_dir_scratch,'bbox_check.png');
    # visualize.plotImageAndAnno(file_im,out_file,np.load(file_npy));



    return
    path_txt='/home/laoreja/deep-landmark-master/dataset/train/trainImageList.txt';
    path_pre='/home/laoreja/deep-landmark-master/dataset/train'
    path_im,bbox,anno_points=parseAnnoFile(path_txt,path_pre,face=True);

    out_dir_scratch = '/home/SSD3/maheen-data/scratch';
    out_dir_bbox_im='/home/SSD3/maheen-data/face_data/im';
    util.mkdir(out_dir_scratch);

    for path_im_curr,bbox_curr,anno_curr in zip(path_im,bbox,anno_points):

        path_im_curr=os.path.join(out_dir_bbox_im,path_im_curr.split('/')[-2],path_im_curr.split('/')[-1]);
        
        im=scipy.misc.imread(path_im_curr);
        im=im[bbox_curr[2]:bbox_curr[3],bbox_curr[0]:bbox_curr[1]];
        min_pts=np.array([bbox_curr[0],bbox_curr[2]])
        anno_curr=np.array(anno_curr);
        print min_pts
        print anno_curr
        anno_curr=anno_curr-min_pts

        print anno_curr;
        out_file=os.path.join(out_dir_scratch,'check_bbox.png');
        # anno_curr=[im.shape[1]/2,im.shape[0]/2];

        visualize.plotImageAndAnno(path_im_curr,out_file,anno_curr);
        break;
        



if __name__=='__main__':
    main();