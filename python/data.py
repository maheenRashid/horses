import cv2
import util;
import visualize;
import os;
import shutil;
import numpy as np;
import glob;
import random;
import multiprocessing;
import scipy.misc;
import Image;
import matplotlib.pyplot as plt;
import cPickle as pickle;
from collections import namedtuple
import h5py;

def createParams(type_Experiment):
    if type_Experiment == 'makeBboxPairFiles':
        list_params=['path_txt',
                'path_pre',
                'type_data',
                'out_dir_meta',
                'out_dir_im',
                'out_dir_npy',
                'out_file_list_npy',
                'out_file_list_im',
                'out_file_pairs',
                'overwrite']
        params = namedtuple('Params_makeBboxPairFiles',list_params);
    else:
        params=None;

    return params;

def parseAnnoFile(path_txt,path_pre=None,face=False):
    face_data=util.readLinesFromFile(path_txt);
    
    path_im=[];
    bbox=[]
    anno_points=[];

    for line_curr in face_data:
        line_split=line_curr.split(' ');
        pts=[float(str_curr) for str_curr in line_split[1:]];
        pts=[int(str_curr) for str_curr in pts]
        if path_pre is not None:
            path_im.append(os.path.join(path_pre,line_split[0].replace('\\','/')));
        else:
            path_im.append(line_split[0]);

        bbox.append(pts[:4]);

        if face:
            increment=2;
        else:
            increment=3;
        
        anno_points_curr=[pts[start:start+increment] for start in range(4,len(pts),increment)];
        anno_points.append(anno_points_curr);

    return path_im,bbox,anno_points

def script_makeBboxPairFiles(params):
    path_txt = params.path_txt
    path_pre = params.path_pre
    type_data = params.type_data
    out_dir_meta = params.out_dir_meta
    out_dir_im = params.out_dir_im
    out_dir_npy = params.out_dir_npy
    out_file_list_npy = params.out_file_list_npy
    out_file_list_im = params.out_file_list_im
    out_file_pairs = params.out_file_pairs
    overwrite = params.overwrite

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

        
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(saveBBoxIm,args);
    # p.map(saveBBoxNpy,args_bbox_npy);

    data_list_npy=[arg_curr[2] for arg_curr in args_bbox_npy];
    data_list_im=[arg_curr[1] for arg_curr in args];
    util.writeFile(out_file_list_npy,data_list_npy);
    util.writeFile(out_file_list_im,data_list_im);

    data_pairs=[pair[0]+' '+pair[1] for pair in data_pairs];
    util.writeFile(out_file_pairs,data_pairs);

def saveBBoxIm((im_file,out_file,bbox,idx)):
    # print idx;
    im=scipy.misc.imread(im_file);
    # im=Image.open(im_file);
    # im=np.asarray(im);
    try:
        if len(im.shape)<3:
            im=im[bbox[2]:bbox[3],bbox[0]:bbox[1]];
        else:
            im=im[bbox[2]:bbox[3],bbox[0]:bbox[1],:];
        scipy.misc.imsave(out_file,im);
    except:
        print idx,im_file
        # return im_file;

def saveBBoxNpy((bbox,key_pts,out_file,idx)):
    # print idx;
    min_pts=np.array([bbox[0],bbox[2]])
    key_pts=np.array(key_pts);
    if key_pts.shape[1]>2:
        for idx in range(key_pts.shape[0]):
            if key_pts[idx,2]>0:
                key_pts[idx,:2]=key_pts[idx,:2]-min_pts;
    else:
        key_pts=key_pts-min_pts
        key_pts=np.hstack((key_pts,np.ones((key_pts.shape[0],1))))
    # print key_pts
    np.save(out_file,key_pts);

def dump_script_makeBBoxPairFiles():
    # horse
    params_dict={};
    params_dict['path_txt'] ='/home/maheenrashid/Downloads/trainImageList_2_clean.txt'
    # '/home/laoreja/finetune-deep-landmark/dataset/train/trainImageList_2.txt';
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'horse';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/horse'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['overwrite'] = False;

    # face
    params_dict={};
    params_dict['path_txt'] = '/home/laoreja/deep-landmark-master/dataset/train/trainImageList.txt';
    params_dict['path_pre'] = '/home/laoreja/deep-landmark-master/dataset/train';
    params_dict['type_data'] = 'face';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/face'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['overwrite'] = False
    
    # aflw
    params_dict={};
    params_dict['path_txt'] = '/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'aflw';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/aflw'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['overwrite'] = False

    params=createParams('makeBboxPairFiles');
    params=params(**params_dict);

    script_makeBboxPairFiles(params)

    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_meta,'params.p'),'wb'));

def makeMatchFile(num_neighbors,matches_file,face_data_file,out_dir_meta_horse,out_dir_meta_face,out_file_horse,out_file_face,out_dir_meta_face_old=None):

    face_data=util.readLinesFromFile(face_data_file);
    face_data=[' '.join(line_curr.split(' ')[:num_neighbors]) for line_curr in face_data];
    
    matches_list=util.readLinesFromFile(matches_file);
    matches_split=[match_curr.split(' ') for match_curr in matches_list];
    horse_list=[match_split[0] for match_split in matches_split];

    
    match_data=[];

    missing_files=[];
    for match_split in matches_split:
        match_split_new=[match_split[0]];

        horse_path,horse_file_name=os.path.split(match_split[0]);
        horse_file_name=horse_file_name[:horse_file_name.rindex('.')];
        horse_path=horse_path.split('/');
        if horse_path[-1]=='gxy':
            horse_path=horse_path[-2];
        else:
            horse_path=horse_path[-1];

        horse_file_out=os.path.join(out_dir_meta_horse[0],horse_path,horse_file_name+'.jpg');
        horse_file_npy_out=os.path.join(out_dir_meta_horse[1],horse_path,horse_file_name+'.npy');

        continue_flag=False;
        for matches_idx in range(num_neighbors):
            start_idx=(matches_idx*num_neighbors)+1;
            end_idx=start_idx+num_neighbors;
            match_curr=match_split[start_idx:end_idx];
            match_curr=' '.join(match_curr);
            
            if match_curr in face_data:
                idx_curr=face_data.index(match_curr)    
            elif ('lfw_5590/' in match_curr) or ('net_7876/' in match_curr):
                # print ('valid',match_curr)
                idx_curr=-1;
            else:
                print ('invalid',match_curr);
                missing_files.append((horse_file_out,horse_file_npy_out,match_curr));
                continue;
            
            file_match_curr=match_curr.split(' ')[0];

            path_curr,file_curr=os.path.split(file_match_curr);
            path_curr=path_curr.split('/')[-1];
            file_curr=file_curr[:file_curr.rindex('.')];
            if idx_curr>=0:
                file_curr=file_curr+'_'+str(idx_curr);
                file_match_curr=os.path.join(out_dir_meta_face[0],path_curr,file_curr+'.jpg');
                file_match_npy_curr=os.path.join(out_dir_meta_face[1],path_curr,file_curr+'.npy');
            else:
                file_match_curr=os.path.join(out_dir_meta_face_old[0],path_curr,file_curr+'.jpg');
                file_match_npy_curr=os.path.join(out_dir_meta_face_old[1],path_curr,file_curr+'.npy');

            match_data.append([horse_file_out,horse_file_npy_out,file_match_curr,file_match_npy_curr]);

    valid_matches=[];
    not_exist=[];
    for match_curr in match_data:
        keep=True;
        for idx,file_curr in enumerate(match_curr):
            if not os.path.exists(file_curr):
                if idx>0:
                    print 'not exist',match_curr,file_curr;
                not_exist.append(file_curr);
                keep=False;
                break;
        if keep:
            valid_matches.append((match_curr[0]+' '+match_curr[1],match_curr[2]+' '+match_curr[3]));

    not_exist=set(not_exist);
    print len(not_exist);
    print len(match_data),len(valid_matches);
    util.writeFile(out_file_horse,[data_curr[0] for data_curr in valid_matches]);
    util.writeFile(out_file_face,[data_curr[1] for data_curr in valid_matches]);
    util.modifyHumanFile(out_file_face,out_file_face_noIm);

    return not_exist;
    
def modifyHumanFileMultiProc((idx,im_file,npy_file)):
    print idx;
    im=scipy.misc.imread(im_file);
    im_size=im.shape;
    line_curr=npy_file+' '+str(im.shape[0])+' '+str(im.shape[1]);
    return line_curr;

def modifyHumanFile(orig_file,new_file):
    data=util.readLinesFromFile(orig_file);
    data=[tuple([idx]+data_curr.split(' ')) for idx,data_curr in enumerate(data)];
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    new_lines=p.map(modifyHumanFileMultiProc,data);
    # new_lines=[];
    # for idx,(im_file,npy_file) in enumerate(data):
    #     print idx,len(data);
    #     im=scipy.misc.imread(im_file);
    #     im_size=im.shape;
    #     line_curr=npy_file+' '+str(im.shape[0])+' '+str(im.shape[1]);
    #     new_lines.append(line_curr);
    print len(new_lines);
    print new_lines[0];
    util.writeFile(new_file,new_lines);


def resizeImAndNpy224((idx,im_file,npy_file,out_im_file,out_npy_file)):
    print idx;
    dir_out_im=os.path.split(out_im_file)[0]
    dir_out_npy=os.path.split(out_npy_file)[0]

    if not os.path.exists(dir_out_im):
        os.makedirs(dir_out_im);
    
    if not os.path.exists(dir_out_npy):
        os.makedirs(dir_out_npy);

    im = cv2.imread(im_file, 1)
    rows, cols = im.shape[:2]
    im = cv2.resize(im, (224, 224))

    cv2.imwrite(out_im_file, im)

    labels = np.load(npy_file).astype(np.float32)
    labels[:,0] = labels[:, 0] * 1.0 / cols * 224.0
    labels[:,1] = labels[:, 1] * 1.0 / rows * 224.0
    np.save(out_npy_file, labels)
    

def script_resizeImAndNpy224():
    pass;



def comparativeViz(out_file_html,folders,num_ims,file_posts,dir_server):
    img_files=[];
    caption_files=[];
    for idx_im in num_ims:
        img_files_curr=[];
        caption_files_curr=[];
        for folder_curr in folders:
            pass;

def comparativeLossViz(img_dirs,file_post,loss_post,range_batches,range_images,out_file_html,dir_server,img_caption_pre=None):
    img_files_all=[];
    captions_all=[];
    if img_caption_pre is not None:
        assert len(img_caption_pre)==len(img_dirs);

    for batch_num in range_batches:
    # range(1,num_batches+1):
        for im_num in range_images:
            for idx_img_dir,img_dir in enumerate(img_dirs):
                loss_all=np.load(os.path.join(img_dir,str(batch_num)+loss_post));
                if im_num>loss_all.shape[0]:
                    continue;

                loss_curr=loss_all[im_num-1,0];
                loss_str="{:10.4f}".format(loss_curr);
                files_curr=[os.path.join(img_dir,str(batch_num)+'_'+str(im_num)+file_post_curr) for file_post_curr in file_post];
                files_curr=[util.getRelPath(file_curr,dir_server) for file_curr in files_curr];
                captions_curr=[os.path.split(file_curr)[1]+' '+loss_str for file_curr in files_curr];
                if img_caption_pre is not None:
                    captions_curr=[img_caption_pre[idx_img_dir]+' '+caption_curr for caption_curr in captions_curr];
                img_files_all.append(files_curr);
                captions_all.append(captions_curr);

    visualize.writeHTML(out_file_html,img_files_all,captions_all,224,224);

    

def generateTPSTestCommands(path_to_th,batch_size,iterations,model_pre,model_post,range_models,out_dir_meta):

    commands_all=[];

    for model_num in range_models:
        path_to_model=model_pre+str(model_num)+model_post;
        just_model_name=os.path.split(path_to_model)[1];
        just_model_name=just_model_name[:just_model_name.rindex('.')];
        out_dir_curr=os.path.join(out_dir_meta,just_model_name);
        if not os.path.exists(out_dir_curr):
            os.makedirs(out_dir_curr);

        command=[];
        command.append('th');
        command.append(path_to_th);
        command.extend(['-model',path_to_model]);
        command.extend(['-outDir',out_dir_curr]);
        command.extend(['-iterations',str(iterations)]);
        command=' '.join(command);
        commands_all.append(command);

    return commands_all;


def findProblemNPYMP(file_curr):
    data=np.load(file_curr);
    data_rel=data[data[:,2]>0,:2];
    if np.any(data_rel)>224:
        return file_curr;
    else:
        return None;

# def findProblemNPY(npy_files):



def main():

    # data='/home/SSD3/maheen-data/horse_project/data_resize/horse/matches_5_train_allKP.txt'
    # # /home/SSD3/maheen-data/horse_project/data_resize/horse/matches_5_train_allKP.txt
    # to_search=\
    # ['/home/SSD3/maheen-data/horse_project/data_check/horse/im/horses_pascal_selected/2009_004662.jpg /home/SSD3/maheen-data/horse_project/data_check/horse/npy/horses_pascal_selected/2009_004662.npy',
    # '/home/SSD3/maheen-data/horse_project/data_check/horse/im/imagenet_n02374451/n02374451_11539.jpg /home/SSD3/maheen-data/horse_project/data_check/horse/npy/imagenet_n02374451/n02374451_11539.npy',
    # '/home/SSD3/maheen-data/horse_project/data_check/horse/im/imagenet_n02374451/n02374451_16786.jpg /home/SSD3/maheen-data/horse_project/data_check/horse/npy/imagenet_n02374451/n02374451_16786.npy',
    # '/home/SSD3/maheen-data/horse_project/data_check/horse/im/imagenet_n02374451/n02374451_4338.jpg /home/SSD3/maheen-data/horse_project/data_check/horse/npy/imagenet_n02374451/n02374451_4338.npy']
    # data=util.readLinesFromFile(data);
    # print data[0];

    # to_search=[file_curr.replace('data_check','data_resize') for file_curr in to_search];
    # idx_lines=[data.index(line_curr) for line_curr in to_search if line_curr in data];
    # print idx_lines;
    # for idx_line_curr in idx_lines:
    #     print 'batch_no',(idx_line_curr)/64

    # # npy_files=[file_curr[file_curr.index(' ')+1:] for file_curr in data];
    # # print npy_files[0];
    # # print len(npy_files);
    # # p=multiprocessing.Pool(multiprocessing.cpu_count());
    # # problem_files=p.map(findProblemNPYMP,npy_files);
    # # problem_files=[file_curr for file_curr in problem_files if file_curr is not None];
    # # print (len(problem_files));




    # return


    # data='/home/laoreja/new-deep-landmark/train/vanilla/aflw_224/aflw_vanilla_val_224.txt';
    # data='/home/laoreja/new-deep-landmark/train/vanilla/aflw_224/aflw_vanilla_train_224_weight.txt';
    # data=util.readLinesFromFile(data);
    # print data;
    # total=0;
    # for h5_file_curr in data:
    #     with h5py.File(h5_file_curr,'r') as hf:
    #         print('List of arrays in this file: ', hf.keys())
    #         data = hf.get('confidence')
    #         np_data = np.array(data)
    #         total=total+np_data.shape[0];
    #         print('Shape of the array dataset_1: ', np_data.shape)
    # print total;


    # return
    # horse_path='/home/SSD3/maheen-data/horse_project/data_resize/horse/matches_5_train_allKP.txt'
    # human_path_noIm='/home/SSD3/maheen-data/horse_project/data_resize/aflw/matches_5_train_allKP_noIm.txt'
    # human_path='/home/SSD3/maheen-data/horse_project/data_resize/aflw/matches_5_train_allKP.txt'
    # paths=[horse_path,human_path_noIm,human_path];
    # out_files=[file_curr[:file_curr.rindex('.')]+'_dummy.txt' for file_curr in paths];
    # for file_curr,out_file_curr in zip(paths,out_files):
    #     data_curr=util.readLinesFromFile(file_curr);
    #     data_curr=data_curr[0:50:5];
    #     # print data_curr;
    #     print len(data_curr);
    #     util.writeFile(out_file_curr,data_curr);
    #     print out_file_curr;




    # return
    # im_path= "/home/SSD3/maheen-data/horse_project/data_resize/horse/im/_04_Aug16_png/horse+head12.jpg"
    #   # 2 : "/home/SSD3/maheen-data/horse_project/data_resize/horse/npy/_04_Aug16_png/horse+head12.npy"
    # # "/home/SSD3/maheen-data/horse_project/data_resize/aflw/im/0/image67102_20650.jpg"
    # np_path="/home/SSD3/maheen-data/horse_project/data_resize/horse/npy/_04_Aug16_png/horse+head12.npy"
    # # "/home/SSD3/maheen-data/horse_project/data_resize/aflw/npy/0/image67102_20650.npy"

    # # im=scipy.misc.read(im_path);
    # im=cv2.imread(im_path);

    # labels=np.load(np_path);
    # print labels
    # for i in xrange(labels.shape[0]):
    #     cv2.circle(im, (labels[i][0], labels[i][1]), 2, (0,0,255), -1)
    # cv2.imwrite('/home/SSD3/maheen-data/temp/check.png', im)





    # return


    # path_to_th='/home/maheenrashid/Downloads/horses/torch/test_tps_cl.th';
    # iterations=10;
    # out_dir_models='/home/SSD3/maheen-data/horse_human_fiveKP_tps_adam'
    # model_pre=os.path.join(out_dir_models,'intermediate','model_all_');
    # model_post='.dat';
    # range_models=range(450,4500+1,450);
    # out_dir_meta=os.path.join(out_dir_models,'test_overtime');
    # batch_size=60;

    # # commands=generateTPSTestCommands(path_to_th,batch_size,iterations,model_pre,model_post,range_models,out_dir_meta)
    # # print len(commands);
    # # print commands[0];

    # # out_file_commands=os.path.join(out_dir_meta+'.sh');
    # # util.writeFile(out_file_commands,commands);

    # dir_server='/home/SSD3/maheen-data';
    # range_batches=range(1,10);
    # # batch_size=60;
    # range_images=range(1,61,5);
    # img_dir_meta='/home/SSD3/maheen-data/horse_human_fiveKP_tps_adam/test_overtime'
    # img_dir=[os.path.join(img_dir_meta,'model_all_'+str(range_model_curr)) for range_model_curr in range_models]
    # out_file_html='/home/SSD3/maheen-data/horse_human_fiveKP_tps_adam/test_viz.html'
    # file_post=['_horse.jpg','_human.jpg','_gtwarp.jpg','_predwarp.jpg']
    # loss_post='_loss.npy';
    # out_file_html=img_dir_meta+'.html';
    # img_caption_pre=[str(model_num) for model_num in range_models];
    # comparativeLossViz(img_dir,file_post,loss_post,range_batches,range_images,out_file_html,dir_server,img_caption_pre)


    # return
    dir_server='/home/SSD3/maheen-data';
    range_batches=range(1,9);
    # batch_size=60;
    range_images=range(1,129,5);
    img_dir=['/home/SSD3/maheen-data/horse_human_fiveKP_tps_adam/test_viz/']
    # out_file_html='/home/SSD3/maheen-data/horse_human_fiveKP_tps_adam/test_viz.html'
    
    img_dir=['/home/SSD3/maheen-data/horse_project/tps_train_allKP_adam/test_viz']
    out_file_html='/home/SSD3/maheen-data/horse_project/tps_train_allKP_adam/test_viz.html'
    
    file_post=['_horse.jpg','_human.jpg','_gtwarp.jpg','_predwarp.jpg']
    loss_post='_loss.npy';
    comparativeLossViz(img_dir,file_post,loss_post,range_batches,range_images,out_file_html,dir_server)


    return
    img_files=[];
    caption_files=[];
    out_dir='/home/SSD3/maheen-data/training_kp_withWarp_test_debug_tps_adam'
    out_dir='/home/SSD3/maheen-data/testing_5_kp_withWarp_fixed_adam_debug';
    out_dir='/home/SSD3/maheen-data/training_5_kp_withWarp_fixed_adam__1e-05/test';
    dir_server='/home/SSD3/maheen-data';
    out_file_html=os.path.join(out_dir,'viz.html');

    for i in range(1,94):
        im_file=os.path.join(out_dir,str(i)+'_org.jpg');
        warp_file=os.path.join(out_dir,str(i)+'_warp.jpg');
        im_file_small=os.path.join(out_dir,str(i)+'_small_org.jpg');
        warp_file_small=os.path.join(out_dir,str(i)+'_small_warp.jpg');
        im_file=util.getRelPath(im_file,dir_server);
        warp_file=util.getRelPath(warp_file,dir_server);

        im_file_small=util.getRelPath(im_file_small,dir_server);
        warp_file_small=util.getRelPath(warp_file_small,dir_server);
        
        img_files.append([im_file,warp_file]);
        # ,im_file_small,warp_file_small]);
        caption_files.append([str(i)+' org',str(i)+' warp']);
        # ,'small_org','small_warp']);

    visualize.writeHTML(out_file_html,img_files,caption_files,224,224);

    return
    out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
    num_neighbors=5;
    out_file_human=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_fiveKP.txt');
    out_file_human_new=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_fiveKP_noIm.txt');
    modifyHumanFile(out_file_human,out_file_human_new)

    # out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
    out_file_human=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
    out_file_human_new=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP_noIm.txt');
    modifyHumanFile(out_file_human,out_file_human_new)

    return
        # matches_file='/home/maheenrashid/Downloads/knn_5_points_train_list_clean.txt'
        # face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
        # # face_data_file_old='/home/laoreja/deep-landmark-master/dataset/train/trainImageList.txt';
        # face_data_list_file='/home/SSD3/maheen-data/aflw_data/npy/data_list.txt';

        # out_dir_meta_horse='/home/SSD3/maheen-data/horse_project/horse';
        # out_dir_meta_horse_list=[os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
        # out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
        # out_dir_meta_face_list=[os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
        
        # out_dir_meta_face_old='/home/SSD3/maheen-data/horse_project/face';
        # out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
        
        # num_neighbors=5;
        # out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
        # out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');

        # missing_files=makeMatchFile(num_neighbors,matches_file,face_data_file,out_dir_meta_horse_list,out_dir_meta_face_list,out_file_horse,out_file_face,out_dir_meta_face_old_list)
        





        # return



        # out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
        # num_neighbors=5;
        # out_file_human=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_fiveKP.txt');
        # out_file_human_new=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_fiveKP_noIm.txt');
        # # modifyHumanFile(out_file_human,out_file_human_new)

        # # out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
        # out_file_human=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
        # out_file_human_new=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP_noIm.txt');
        # # modifyHumanFile(out_file_human,out_file_human_new)
        # print out_file_human_new;


        # return
    # img_files=[];
    # caption_files=[];
    # out_dir='/home/SSD3/maheen-data/training_kp_withWarp_test_final'
    # dir_server='/home/SSD3/maheen-data';
    # out_file_html=os.path.join(out_dir,'viz.html');

    # for i in range(1,94):
    #     im_file=os.path.join(out_dir,str(i)+'.jpg');
    #     warp_file=os.path.join(out_dir,str(i)+'_warp.jpg');
    #     im_file=util.getRelPath(im_file,dir_server);
    #     warp_file=util.getRelPath(warp_file,dir_server);
    #     img_files.append([im_file,warp_file]);
    #     caption_files.append(['org','warp']);

    # visualize.writeHTML(out_file_html,img_files,caption_files,224,224);

    # return

    file_horse='/home/SSD3/maheen-data/horse_project/horse/matches_5_train_fiveKP.txt';
    out_file_horse='/home/SSD3/maheen-data/horse_project/horse_resize/matches_5_train_fiveKP.txt';

    lines=util.readLinesFromFile(file_horse);
    print len(lines);
    
    lines=list(set(lines));
    
    print len(lines);

    lines=[line_curr.split(' ') for line_curr in lines];
    
    im_files=[line_curr[0] for line_curr in lines];
    npy_files=[line_curr[1] for line_curr in lines];

    out_dir_meta_old='/home/SSD3/maheen-data/horse_project/horse/';
    out_dir_meta_new='/home/SSD3/maheen-data/horse_project/horse_resize/';
    replace_paths=[out_dir_meta_old,out_dir_meta_new];

    args=[];
    for idx in range(len(im_files)):
        im_file=im_files[idx];
        npy_file=npy_files[idx];
        out_im_file=im_file.replace(replace_paths[0],replace_paths[1]);    
        out_npy_file=npy_file.replace(replace_paths[0],replace_paths[1]);
        args.append((idx,im_file,npy_file,out_im_file,out_npy_file));

    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(resizeImAndNpy224,args);

    out_dir_meta_old='/home/SSD3/maheen-data/horse_project/horse/';
    out_dir_meta_new='/home/SSD3/maheen-data/horse_project/horse_resize/';
    replace_paths=[out_dir_meta_old,out_dir_meta_new];
    lines=util.readLinesFromFile(file_horse);
    lines_new=[line.replace(replace_paths[0],replace_paths[1]) for line in lines];
    util.writeFile(out_file_horse,lines_new);

    lines=util.readLinesFromFile(out_file_horse);
    print (len(lines))
    im_file=lines[90].split(' ')[0];
    im=cv2.imread(im_file,1);

    labels=np.load(lines[90].split(' ')[1]);

    for i in xrange(labels.shape[0]):
        cv2.circle(im, (labels[i][0], labels[i][1]), 2, (0,0,255), -1)
    cv2.imwrite('/home/SSD3/maheen-data/temp/check.png', im)



    return

    dir_out='/home/SSD3/maheen-data/temp/horse_human/viz_transform_aflw_val';

    visualize.writeHTMLForFolder(dir_out);

    return
    out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
    num_neighbors=5;
    out_file_human=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_fiveKP.txt');
    out_file_human_new=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_fiveKP_noIm.txt');
    modifyHumanFile(out_file_human,out_file_human_new)

    # out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
    out_file_human=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
    out_file_human_new=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP_noIm.txt');
    modifyHumanFile(out_file_human,out_file_human_new)

    return
    matches_file='/home/laoreja/data/knn_res_new/knn_5_points_val_list.txt'
    
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    # face_data_file_old='/home/laoreja/deep-landmark-master/dataset/train/trainImageList.txt';
    face_data_list_file='/home/SSD3/maheen-data/aflw_data/npy/data_list.txt';

    out_dir_meta_horse='/home/SSD3/maheen-data/horse_project/horse';
    out_dir_meta_horse_list=[os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
    out_dir_meta_face_list=[os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    
    out_dir_meta_face_old='/home/SSD3/maheen-data/horse_project/face';
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_fiveKP.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_val_fiveKP.txt');

    missing_files=makeMatchFile(num_neighbors,matches_file,face_data_file,out_dir_meta_horse_list,out_dir_meta_face_list,out_file_horse,out_file_face,out_dir_meta_face_old_list)

    return
    matches_file='/home/laoreja/data/knn_res_new/knn_5_points_train_list.txt'
    matches_file='/home/maheenrashid/Downloads/knn_5_points_train_list_clean.txt'
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    # face_data_file_old='/home/laoreja/deep-landmark-master/dataset/train/trainImageList.txt';
    face_data_list_file='/home/SSD3/maheen-data/aflw_data/npy/data_list.txt';

    out_dir_meta_horse='/home/SSD3/maheen-data/horse_project/horse';
    out_dir_meta_horse_list=[os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
    out_dir_meta_face_list=[os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    
    out_dir_meta_face_old='/home/SSD3/maheen-data/horse_project/face';
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');

    missing_files=makeMatchFile(num_neighbors,matches_file,face_data_file,out_dir_meta_horse_list,out_dir_meta_face_list,out_file_horse,out_file_face,out_dir_meta_face_old_list)
    
    return
    out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'.txt');
    out_file_face_new=os.path.join(out_dir_meta_face,'matches_noIm_'+str(num_neighbors)+'.txt');
    # modifyHumanFile(out_file_face,out_file_face_new);

    # old_data=util.readLinesFromFile(out_file_face);
    # old_data=[line_curr.split(' ')[1] for line_curr in old_data];
    # new_data=util.readLinesFromFile(out_file_face_new);
    # new_data=[line_curr.split(' ')[0] for line_curr in new_data];
    # assert len(old_data)==len(new_data);
    # for i,old_line in enumerate(old_data):
    #     print i;
    #     assert old_line==new_data[i];

    return
    matches_file='/home/laoreja/data/knn_res_new/5_points_list.txt';

    matches_file='/home/laoreja/data/knn_res_new/knn_train_list.txt'
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    face_data_list_file='/home/SSD3/maheen-data/aflw_data/npy/data_list.txt';
    out_dir_meta_horse='/home/SSD3/maheen-data/horse_project/horse';
    out_dir_meta_horse_list=[os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face='/home/SSD3/maheen-data/horse_project/aflw';
    out_dir_meta_face_list=[os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'.txt');

    makeMatchFile(num_neighbors,matches_file,face_data_file,out_dir_meta_horse_list,out_dir_meta_face_list,out_file_horse,out_file_face)


    return
    # script_saveTrainTxt()
    # dir_viz='/home/SSD3/maheen-data/temp/horse_human/viz_transform_aflw';
    # visualize.writeHTMLForFolder(dir_viz,'.jpg');

    return
    out_dir_meta='/home/SSD3/maheen-data'
    face_dir='aflw_data';
    horse_dir='horse_data';
    num_neighbors=5;
    
    path_replace_horse=['/home/laoreja/data/horse-images/annotation',os.path.join(out_dir_meta,horse_dir,'im')];
    path_replace_face=['/npy/','/im/'];
    new_match_file = os.path.join(out_dir_meta,face_dir,'match_'+str(num_neighbors)+'.txt');
    out_face_train_file = os.path.join(out_dir_meta,face_dir,'match_'+str(num_neighbors)+'_train.txt');
    out_horse_train_file = os.path.join(out_dir_meta,horse_dir,'match_'+str(num_neighbors)+'_train.txt');
    horse_txt_file = os.path.join(out_dir_meta,horse_dir,'train.txt');
    face_txt_file = os.path.join(out_dir_meta,face_dir,'train.txt');

    horse_train=util.readLinesFromFile(horse_txt_file);
    horse_train_just_beginning=[horse_curr.split(' ')[0] for horse_curr in horse_train];
    horse_train_just_beginning=[horse_curr[:horse_curr.rindex('.')] for horse_curr in horse_train_just_beginning];
    print horse_train_just_beginning[0];
    face_train=util.readLinesFromFile(face_txt_file);
    face_train_just_beginning=[face_curr.split(' ')[0] for face_curr in face_train];
    face_train_just_beginning=[face_curr[:face_curr.rindex('.')] for face_curr in face_train_just_beginning];


    print len(horse_train);
    print horse_train[0];
    print len(face_train);
    print face_train[0];
    # return
    matches=util.readLinesFromFile(new_match_file);
    print (len(matches));
    matches=[match_curr.split(' ') for match_curr in matches];

    horse_matches = [];
    face_matches = [];

    for match_curr in matches:
        assert len(match_curr)==num_neighbors+1;
        horse_curr=match_curr[0];

        horse_curr_path,horse_name=os.path.split(horse_curr);

        if horse_curr_path[-3:]=='gxy':
            horse_curr_path=horse_curr_path[:-3];

        horse_curr_path=horse_curr_path.replace(path_replace_horse[0],path_replace_horse[1]);

        horse_curr=os.path.join(horse_curr_path,horse_name[:horse_name.rindex('.')])
        if horse_curr in horse_train_just_beginning:
            horse_match=horse_train[horse_train_just_beginning.index(horse_curr)]
        else:
            # print horse_curr
            # print match_curr[0];
            # raw_input();
            continue;

        for face_curr in match_curr[1:]:
            face_curr=face_curr[:face_curr.rindex('.')];
            face_curr=face_curr.replace(path_replace_face[0],path_replace_face[1]);
            face_match=face_train[face_train_just_beginning.index(face_curr)];
            horse_matches.append(horse_match);
            face_matches.append(face_match);

        # print match_curr;
        # print match_curr[0];
        # for idx,i in enumerate(match_curr[1:]):
        #   print idx,face_matches[idx],i,horse_matches[idx]
    assert len(face_matches)==len(horse_matches);
    print len(face_matches);
    util.writeFile(out_face_train_file,face_matches);
    util.writeFile(out_horse_train_file,horse_matches);



    return
    # face_dir='/home/SSD3/maheen-data/face_data';
    # train_txt=os.path.join(face_dir,'train.txt');
    # files=util.readLinesFromFile(train_txt);
    # files=[file_curr.split(' ') for file_curr in files];
    # [im_files,npy_files]=zip(*files);
    # for idx,npy_file in enumerate(npy_files):
    #   print idx,len(npy_files);
    #   assert os.path.exists(npy_file);
    #   assert np.load(npy_file).shape[1]==3;

    # print len(im_files);
    # print (im_files[0]);

    # print len(npy_files);
    # print (npy_files[0]);
    dir_viz='/home/SSD3/maheen-data/temp/horse_human/viz_transform';
    visualize.writeHTMLForFolder(dir_viz,'.jpg');

    return
    horse_data='/home/SSD3/maheen-data/horse_data';
    new_face_data = '/home/SSD3/maheen-data/face_data';
    old_txt='train.txt';
    num_to_keep=10;
    new_txt='train_'+str(num_to_keep)+'.txt';
    for data_type in [horse_data,new_face_data]:
        lines_new=util.readLinesFromFile(os.path.join(data_type,old_txt));
        random.shuffle(lines_new);
        lines_new=lines_new[:num_to_keep];
        file_new=os.path.join(data_type,new_txt)
        util.writeFile(file_new,lines_new);
        print len(lines_new),file_new;
    

    return
        


if __name__=='__main__':
    main();