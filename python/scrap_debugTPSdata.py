import util
import os;
import visualize;
import cv2;
import numpy as np;
import multiprocessing;
from data import *;
def makeCulpritFile():
    
    
    out_dir='/home/SSD3/maheen-data/temp/debug_problem_batch';
    file_human='/home/SSD3/maheen-data/horse_project/aflw/matches_5_train_fiveKP_noIm.txt';
    file_horse='/home/SSD3/maheen-data/horse_project/horse_resize/matches_5_train_fiveKP.txt';
    new_file_human=file_human[:file_human.rindex('.')]+'_debug.txt';
    new_file_horse=file_horse[:file_horse.rindex('.')]+'_debug.txt'
    batch_no=3;
    batch_size=64;

    data_horse=util.readLinesFromFile(file_horse);
    data_human=util.readLinesFromFile(file_human);
    
    assert len(data_horse)==len(data_human);
    print (len(data_horse)/batch_size)
    # for batch_no in range(71,72):
    batch_no=71;
    line_idx=(batch_size*(batch_no-1))%len(data_horse);

    print ('____');
    print (batch_no);
    print (line_idx);
    print data_horse[line_idx];
    print data_human[line_idx];
    data_horse_rel=data_horse[line_idx:line_idx+batch_size];
    data_human_rel=data_human[line_idx:line_idx+batch_size];
    assert len(data_horse_rel)==batch_size;
    assert len(data_human_rel)==batch_size;

    util.writeFile(new_file_horse,data_horse_rel);
    util.writeFile(new_file_human,data_human_rel);
    print new_file_human;
    print new_file_horse;


def saveImWithAnno((idx,im_path,npy_path,out_path)):
    print idx;
    im=cv2.imread(im_path,1);
    # im=cv2.res tiize(im,(224,224));
    label=np.load(npy_path).astype(np.int);
    x=label[:,0];
    y=label[:,1];
    color=(0,0,255);
    colors=[(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,255,255)]

    for label_idx in range(len(x)):
        if label[label_idx,2]>0:
            cv2.circle(im,(x[label_idx],y[label_idx]),5,colors[label_idx],-1);
    cv2.imwrite(out_path,im);

def local_script_makeBboxPairFiles(params):
    path_txt = params['path_txt']
    path_pre = params['path_pre']
    type_data = params['type_data']
    out_dir_meta = params['out_dir_meta']
    out_dir_im = params['out_dir_im']
    out_dir_npy = params['out_dir_npy']
    out_file_list_npy = params['out_file_list_npy']
    out_file_list_im = params['out_file_list_im']
    out_file_pairs = params['out_file_pairs']
    overwrite = params['overwrite']

    util.mkdir(out_dir_im);
    util.mkdir(out_dir_npy);

    if type_data=='face':
        path_im,bbox,anno_points=parseAnnoFile(path_txt,path_pre,face=True);
    else:
        path_im,bbox,anno_points=parseAnnoFile(path_txt,path_pre,face=False);

    args = [];
    args_bbox_npy=[];
    data_pairs=[];
    for idx,path_im_curr,bbox_curr,key_pts in zip(range(len(path_im)),path_im,bbox,anno_points):    
        path_curr,file_name=os.path.split(path_im_curr);
        file_name=file_name[:file_name.rindex('.')];
        path_curr=path_curr.split('/');

        if type_data=='horse':
            if path_curr[-1]=='gxy':
                path_pre_curr=path_curr[-2];
            else:
                path_pre_curr=path_curr[-1];
        else:
            path_pre_curr=path_curr[-1];

        if type_data=='aflw':
            file_name=file_name+'_'+str(idx);

        out_dir_curr=os.path.join(out_dir_im,path_pre_curr);
        out_dir_npy_curr=os.path.join(out_dir_npy,path_pre_curr);

        util.mkdir(out_dir_curr);
        util.mkdir(out_dir_npy_curr);

        # out_file=os.path.join(out_dir_curr,file_name);
        out_file=os.path.join(out_dir_curr,file_name+'.jpg');
        out_file_npy=os.path.join(out_dir_npy_curr,file_name+'.npy');
        data_pairs.append((out_file,out_file_npy));

        if not os.path.exists(out_file) or overwrite:
            args.append((path_im_curr,out_file,bbox_curr,idx));
        if not os.path.exists(out_file_npy) or overwrite:
            args_bbox_npy.append((bbox_curr,key_pts,out_file_npy,idx));

    print args_bbox_npy
    print args
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(saveBBoxIm,args);
    p.map(saveBBoxNpy,args_bbox_npy);

    data_list_npy=[arg_curr[2] for arg_curr in args_bbox_npy];
    data_list_im=[arg_curr[1] for arg_curr in args];
    util.writeFile(out_file_list_npy,data_list_npy);
    util.writeFile(out_file_list_im,data_list_im);

    data_pairs=[pair[0]+' '+pair[1] for pair in data_pairs];
    util.writeFile(out_file_pairs,data_pairs);



def main():

    old_file='/home/laoreja/data/knn_res_new/knn_5_points_train_list.txt'
    new_file='/home/maheenrashid/Downloads/knn_5_points_train_list_clean.txt'
    match_str='n02374451_4338.JPEG'
    lines=util.readLinesFromFile(old_file);
    lines_to_keep=[];
    for line in lines:
        if match_str not in line:
            lines_to_keep.append(line);
    assert len(lines_to_keep)==len(lines)-1;
    util.writeFile(new_file,lines_to_keep);

    return
    file_curr='/home/laoreja/finetune-deep-landmark/dataset/train/trainImageList_2.txt';
    out_file='/home/maheenrashid/Downloads/trainImageList_2_clean.txt';
    lines=util.readLinesFromFile(file_curr);
    lines_to_keep=[];
    for line in lines:
        if line=='/home/laoreja/data/horse-images/annotation/imagenet_n02374451/gxy/n02374451_4338.JPEG 156 169 79 99 161 88 1 43 46 1 167 95 1 164 95 1 43 56 1':
            print 'found!'
        else:
            lines_to_keep.append(line);

    print len(lines_to_keep),len(lines);
    assert len(lines_to_keep)+1==len(lines);
    util.writeFile(out_file,lines_to_keep);

    return
    horse_file='/home/SSD3/maheen-data/horse_project/horse/matches_5_val_fiveKP.txt';
    human_file='/home/SSD3/maheen-data/horse_project/aflw/matches_5_val_fiveKP_noIm.txt'
    horse_data=util.readLinesFromFile(horse_file);
    human_data=util.readLinesFromFile(human_file);

    # horse_data=[horse_data[41]];
    # human_data=[human_data[41]];
    # print horse_data[0];
    
    horse_im=[line_curr.split(' ')[0] for line_curr in horse_data];
    human_im=[line_curr.split(' ')[0].replace('/npy/','/im/').replace('.npy','.jpg') for line_curr in human_data];

    horse_npy=[line_curr.split(' ')[1] for line_curr in horse_data];
    human_npy=[line_curr.split(' ')[0] for line_curr in human_data];

    problem_cases=[];
    for horse_npy_curr in horse_npy:
        labels=np.load(horse_npy_curr);
        if np.any(labels<0):
            problem_cases.append(horse_npy_curr);

    print len(problem_cases),len(set(problem_cases));





    return

    dir_server='/home/SSD3/maheen-data';
    out_dir_debug=os.path.join(dir_server,'temp','debug_problem_batch/rerun');
    
    im_file='/home/laoreja/data/horse-images/annotation/imagenet_n02374451/gxy/n02374451_4338.JPEG';
    npy_file='/home/SSD3/maheen-data/temp/debug_problem_batch/rerun/npy/imagenet_n02374451/n02374451_4338.npy';

    out_file=os.path.join(out_dir_debug,'check.png');
    saveImWithAnno((1,im_file,npy_file,out_file))


    # arg=([156, 169, 79, 99], [[161, 88, 1], [43, 46, 1], [167, 95, 1], [164, 95, 1], [43, 56, 1]], '/home/SSD3/maheen-data/temp/debug_problem_batch/rerun/npy/imagenet_n02374451/n02374451_4338.npy', 0);
    # # print np.load(arg[2]);
    # saveBBoxNpy(arg);
    # # print np.load(arg[2]);

    return
    dir_server='/home/SSD3/maheen-data';
    out_dir_debug=os.path.join(dir_server,'temp','debug_problem_batch/rerun');
    util.mkdir(out_dir_debug);
    params_dict={};
    params_dict['path_txt'] = '/home/SSD3/maheen-data/temp/debug_problem_batch/train_dummy.txt'
    # '/home/laoreja/finetune-deep-landmark/dataset/train/trainImageList_2.txt';
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'horse';
    params_dict['out_dir_meta'] = out_dir_debug
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['overwrite'] = True;
    local_script_makeBboxPairFiles(params_dict)

    return
    npy_file='/home/SSD3/maheen-data/horse_project/horse/npy/imagenet_n02374451/n02374451_4338.npy';
    labels=np.load(npy_file);
    print labels;

    return
    dir_server='/home/SSD3/maheen-data';
    out_dir_debug=os.path.join(dir_server,'temp','debug_problem_batch');
    util.mkdir(out_dir_debug);
    
    out_horse_im_dir=os.path.join(out_dir_debug,'horse_im');
    out_human_im_dir=os.path.join(out_dir_debug,'human_im');
    util.mkdir(out_horse_im_dir);util.mkdir(out_human_im_dir);


    horse_file='/home/SSD3/maheen-data/horse_project/horse_resize/matches_5_train_fiveKP_debug.txt';
    human_file='/home/SSD3/maheen-data/horse_project/aflw/matches_5_train_fiveKP_noIm_debug.txt'
    horse_data=util.readLinesFromFile(horse_file);
    human_data=util.readLinesFromFile(human_file);
    horse_data=[horse_data[41]];
    human_data=[human_data[41]];
    print horse_data[0];
    
    horse_im=[line_curr.split(' ')[0] for line_curr in horse_data];
    human_im=[line_curr.split(' ')[0].replace('/npy/','/im/').replace('.npy','.jpg') for line_curr in human_data];

    horse_npy=[line_curr.split(' ')[1] for line_curr in horse_data];
    human_npy=[line_curr.split(' ')[0] for line_curr in human_data];

    args=[];
    for idx,horse_im_curr in enumerate(horse_im):
        args.append((idx,horse_im_curr,horse_npy[idx],os.path.join(out_horse_im_dir,str(idx)+'.jpg')));
    for idx,horse_im_curr in enumerate(human_im):
        args.append((idx,horse_im_curr,human_npy[idx],os.path.join(out_human_im_dir,str(idx)+'.jpg')));

    # saveImWithAnno(args[-1]);
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(saveImWithAnno,args);

    out_file_html=os.path.join(out_dir_debug,'viz_matches.html');
    img_paths=[];
    captions=[];

    for idx in range(len(horse_im)):
        horse_im_curr=os.path.join(out_horse_im_dir,str(idx)+'.jpg');
        horse_im_curr=util.getRelPath(horse_im_curr,dir_server);
        human_im_curr=os.path.join(out_human_im_dir,str(idx)+'.jpg');
        human_im_curr=util.getRelPath(human_im_curr,dir_server);
        img_paths.append([horse_im_curr,human_im_curr]);
        captions.append(['horse '+str(idx),'human']);

    # for idx,horse_im_curr in enumerate(horse_im):
    #   human_im_curr=util.getRelPath(human_im[idx],dir_server);
    #   horse_im_curr=util.getRelPath(horse_im_curr,dir_server);
    #   img_paths.append([horse_im_curr,human_im_curr]);
    #   captions.append(['horse','human']);

    visualize.writeHTML(out_file_html,img_paths,captions,224,224);






if __name__=='__main__':
    main();


