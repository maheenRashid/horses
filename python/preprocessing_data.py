import util;
import numpy as np;
import os;
import cv2;
import cPickle as pickle;
from collections import namedtuple
import multiprocessing;

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
                'overwrite',
                'resize',
                'out_file_problem']
        params = namedtuple('Params_makeBboxPairFiles',list_params);
    elif type_Experiment=='makeMatchFile':
        list_params=['num_neighbors',
                    'matches_file',
                    'face_data_file',
                    'out_dir_meta_horse',
                    'out_dir_meta_face',
                    'out_file_horse',
                    'out_file_face',
                    'out_file_face_noIm',
                    'out_dir_meta_face_old',
                    'resize',
                    'threshold'];
        params = namedtuple('Params_makeMatchFile',list_params);
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

def saveBBoxImAndNpyResize((im_file,bbox,key_pts,out_file,out_file_npy,resize_size,idx)):
    # resize_size is rows cols
    print idx;
    problem=False;
    im = cv2.imread(im_file, 1)
    rows, cols = im.shape[:2]
    im=im[bbox[2]:bbox[3],bbox[0]:bbox[1],:];
    if resize_size is not None:
        im = cv2.resize(im, (resize_size[1],resize_size[0]))
    
    key_pts=getBBoxNpyResize(bbox,key_pts,resize_size)
    # ,cols,rows);
    if key_pts is None or np.any(key_pts[key_pts[:,2]>0,:2]<0):
        problem=True;

    if problem:
        print key_pts;
        return (out_file,out_file_npy,im_file);
    else:
        cv2.imwrite(out_file, im)
        np.save(out_file_npy,key_pts);      

def getBBoxNpyResize(bbox,key_pts,resize_size):
# ,cols,rows):
    min_pts=np.array([bbox[0],bbox[2]])
    key_pts=np.array(key_pts);
    if key_pts.shape[1]>2:
        for idx in range(key_pts.shape[0]):
            if key_pts[idx,2]>0:
                key_pts[idx,:2]=key_pts[idx,:2]-min_pts;
    else:
        key_pts=key_pts-min_pts
        key_pts=np.hstack((key_pts,np.ones((key_pts.shape[0],1))))
    
    key_pts= key_pts.astype(np.float32);
    
    cols=bbox[3]-bbox[2];
    rows=bbox[1]-bbox[0];
    if np.any(key_pts[:,0]>rows) or np.any(key_pts[:,1]>cols):
        # print key_pts,cols,rows
        # print 'PROBLEM';
        return None;

    if resize_size is not None :
        key_pts[:,0] = key_pts[:, 0] * 1.0 / rows * resize_size[1];
        key_pts[:,1] = key_pts[:, 1] * 1.0 / cols * resize_size[0];
    
    return key_pts;

def modifyHumanFileMultiProc((idx,im_file,npy_file)):
    print idx;
    im=scipy.misc.imread(im_file);
    im_size=im.shape;
    line_curr=npy_file+' '+str(im.shape[0])+' '+str(im.shape[1]);
    return line_curr;

def modifyHumanFile(orig_file,new_file,resize):
    data=util.readLinesFromFile(orig_file);
    data=[tuple([idx]+data_curr.split(' ')) for idx,data_curr in enumerate(data)];
    if resize is None:
        p=multiprocessing.Pool(multiprocessing.cpu_count());
        new_lines=p.map(modifyHumanFileMultiProc,data);
    else:
        new_lines=[data_curr[2]+' '+str(resize[0])+' '+str(resize[1]) for data_curr in data];
    util.writeFile(new_file,new_lines);

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
    resize_size=params.resize;
    out_file_problem=params.out_file_problem;

    util.mkdir(out_dir_im);
    util.mkdir(out_dir_npy);

    if type_data=='face':
        path_im,bbox,anno_points=parseAnnoFile(path_txt,path_pre,face=True);
    else:
        path_im,bbox,anno_points=parseAnnoFile(path_txt,path_pre,face=False);

    args = [];
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

        out_file=os.path.join(out_dir_curr,file_name+'.jpg');
        out_file_npy=os.path.join(out_dir_npy_curr,file_name+'.npy');
        data_pairs.append((out_file,out_file_npy));

        if not os.path.exists(out_file) or overwrite:
            args.append((path_im_curr,bbox_curr,key_pts,out_file,out_file_npy,resize_size,idx));

    # print len(args),len(path_im);
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    
    problem_cases=p.map(saveBBoxImAndNpyResize,args);
    problem_cases=[pair_curr for pair_curr in problem_cases if pair_curr is not None];
    problem_pairs=[(pair_curr[0],pair_curr[1]) for pair_curr in problem_cases];
    for p in problem_pairs:
        print p[0]+' '+p[1];

    data_pairs=[pair_curr for pair_curr in data_pairs if pair_curr not in problem_pairs];
    data_list_npy=[pair_curr[1] for pair_curr in data_pairs];
    data_list_im=[pair_curr[0] for pair_curr in data_pairs];

    util.writeFile(out_file_list_npy,data_list_npy);
    util.writeFile(out_file_list_im,data_list_im);

    data_pairs=[pair[0]+' '+pair[1] for pair in data_pairs];
    util.writeFile(out_file_pairs,data_pairs);

    problem_input=[pair_curr[2] for pair_curr in problem_cases];
    util.writeFile(out_file_problem,problem_input);

def dump_script_makeBBoxPairFiles():
    # horse
    params_dict={};
    params_dict['path_txt'] ='/home/laoreja/new-deep-landmark/dataset/train/trainImageList_2.txt'
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'horse';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_resize/horse'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem.txt');
    params_dict['overwrite'] = True;
    params_dict['resize']=(224,224);

    # face
    params_dict={};
    params_dict['path_txt'] = '/home/laoreja/deep-landmark-master/dataset/train/trainImageList.txt';
    params_dict['path_pre'] = '/home/laoreja/deep-landmark-master/dataset/train';
    params_dict['type_data'] = 'face';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_resize/face'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem.txt');
    params_dict['overwrite'] = True;
    params_dict['resize']=(224,224);


    # aflw
    params_dict={};
    params_dict['path_txt'] = '/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'aflw';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_resize/aflw'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem.txt');
    params_dict['overwrite'] = False;
    params_dict['resize']=(224,224);

    params=createParams('makeBboxPairFiles');
    params=params(**params_dict);
    script_makeBboxPairFiles(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_meta,'params.p'),'wb'));

    #val    
    params_dict={};
    params_dict['path_txt'] ='/home/laoreja/new-deep-landmark/dataset/train/valImageList_2.txt'
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'horse';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_resize/horse'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list_val.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list_val.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs_val.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem_val.txt');
    params_dict['overwrite'] = False;
    params_dict['resize']=(224,224);

    params=createParams('makeBboxPairFiles');
    params=params(**params_dict);
    script_makeBboxPairFiles(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_meta,'params_val.p'),'wb'));

def makeMatchFile(params):
    num_neighbors = params.num_neighbors;
    matches_file = params.matches_file;
    face_data_file = params.face_data_file;
    out_dir_meta_horse = params.out_dir_meta_horse;
    out_dir_meta_face = params.out_dir_meta_face;
    out_file_horse = params.out_file_horse;
    out_file_face = params.out_file_face;
    out_file_face_noIm = params.out_file_face_noIm;
    out_dir_meta_face_old = params.out_dir_meta_face_old;
    resize = params.resize;
    threshold = params.threshold;

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

    print missing_files
    valid_matches=[];
    not_exist=[];
    too_linear=[];
    for match_curr in match_data:
        keep=True;
        for idx,file_curr in enumerate(match_curr):
            if not os.path.exists(file_curr):
                not_exist.append(file_curr);
                keep=False;
                break;
        if keep:
            keep=isValidPair((match_curr[1],match_curr[3],threshold));
            if not keep:
                too_linear.append(match_curr);

        if keep:
            valid_matches.append((match_curr[0]+' '+match_curr[1],match_curr[2]+' '+match_curr[3]));
        
    not_exist=set(not_exist);
    print 'not existing files',len(not_exist);
    print 'too linear',len(too_linear);
    print 'total matches',len(match_data),'total valid',len(valid_matches);
    util.writeFile(out_file_horse,[data_curr[0] for data_curr in valid_matches]);
    util.writeFile(out_file_face,[data_curr[1] for data_curr in valid_matches]);
    modifyHumanFile(out_file_face,out_file_face_noIm,resize);

    return not_exist;

def dump_makeMatchFile():
    # five kp
    out_dir_meta_data='/home/SSD3/maheen-data/horse_project/data_resize';
    matches_file='/home/maheenrashid/Downloads/knn_5_points_train_list_clean.txt'
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    face_data_list_file=os.path.join(out_dir_meta_data,'aflw','npy','data_list.txt');

    out_dir_meta_horse = os.path.join(out_dir_meta_data,'horse');
    out_dir_meta_horse_list = [os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face = os.path.join(out_dir_meta_data,'aflw');
    out_dir_meta_face_list = [os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    
    out_dir_meta_face_old=os.path.join(out_dir_meta_data,'face');
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
    out_file_face_noIm=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP_noIm.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
    resize=(224,224)

    params_dict={};
    params_dict['num_neighbors'] = num_neighbors;
    params_dict['matches_file'] = matches_file;
    params_dict['face_data_file'] = face_data_file;
    params_dict['out_dir_meta_horse'] = out_dir_meta_horse_list;
    params_dict['out_dir_meta_face'] = out_dir_meta_face_list;
    params_dict['out_file_horse'] = out_file_horse;
    params_dict['out_file_face'] = out_file_face;
    params_dict['out_file_face_noIm'] = out_file_face_noIm;
    params_dict['out_dir_meta_face_old'] = out_dir_meta_face_old_list;
    params_dict['resize'] = resize;

    # all
    out_dir_meta_data='/home/SSD3/maheen-data/horse_project/data_resize';
    matches_file='/home/laoreja/data/knn_res_new/knn_train_list.txt';
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    face_data_list_file=os.path.join(out_dir_meta_data,'aflw','npy','data_list.txt');
    out_dir_meta_horse = os.path.join(out_dir_meta_data,'horse');
    out_dir_meta_horse_list = [os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face = os.path.join(out_dir_meta_data,'aflw');
    out_dir_meta_face_list = [os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    out_dir_meta_face_old=os.path.join(out_dir_meta_data,'face');
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_allKP.txt');
    out_file_face_noIm=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_allKP_noIm.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_train_allKP.txt');
    resize=(224,224)

    params_dict={};
    params_dict['num_neighbors'] = num_neighbors;
    params_dict['matches_file'] = matches_file;
    params_dict['face_data_file'] = face_data_file;
    params_dict['out_dir_meta_horse'] = out_dir_meta_horse_list;
    params_dict['out_dir_meta_face'] = out_dir_meta_face_list;
    params_dict['out_file_horse'] = out_file_horse;
    params_dict['out_file_face'] = out_file_face;
    params_dict['out_file_face_noIm'] = out_file_face_noIm;
    params_dict['out_dir_meta_face_old'] = out_dir_meta_face_old_list;
    params_dict['resize'] = resize;

    # val
    out_dir_meta_data='/home/SSD3/maheen-data/horse_project/data_resize';
    matches_file='/home/SSD3/maheen-data/temp/knn_all_points_val_list.txt';
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    face_data_list_file=os.path.join(out_dir_meta_data,'aflw','npy','data_list.txt');
    out_dir_meta_horse = os.path.join(out_dir_meta_data,'horse');
    out_dir_meta_horse_list = [os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face = os.path.join(out_dir_meta_data,'aflw');
    out_dir_meta_face_list = [os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    out_dir_meta_face_old=os.path.join(out_dir_meta_data,'face');
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_allKP.txt');
    out_file_face_noIm=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_allKP_noIm.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_val_allKP.txt');
    resize=(224,224)

    params_dict={};
    params_dict['num_neighbors'] = num_neighbors;
    params_dict['matches_file'] = matches_file;
    params_dict['face_data_file'] = face_data_file;
    params_dict['out_dir_meta_horse'] = out_dir_meta_horse_list;
    params_dict['out_dir_meta_face'] = out_dir_meta_face_list;
    params_dict['out_file_horse'] = out_file_horse;
    params_dict['out_file_face'] = out_file_face;
    params_dict['out_file_face_noIm'] = out_file_face_noIm;
    params_dict['out_dir_meta_face_old'] = out_dir_meta_face_old_list;
    params_dict['resize'] = resize;

    params=createParams('makeMatchFile');
    params=params(**params_dict);

    makeMatchFile(params);

    pickle.dump(params._asdict(),open(os.path.join(out_dir_meta_data,'Params_makeMatchFile.p'),'wb'));

def isValidPair((horse_file,human_file,threshold)):
    valid_horse,valid_human=getValidPoints(horse_file,human_file);
    isValid=True;
    if valid_horse.shape[0]<3:
        isValid=False;
    elif (valid_horse.shape[0]==3):
        distances=getAllPointLineDistancesThreePts(valid_human[:,:2]);
        if min(distances)<=threshold:
            isValid=False;
    return isValid;

def getPointLineDistance(line_pt_1,line_pt_2,pt):
    assert line_pt_1.size==2;
    assert line_pt_2.size==2;
    assert pt.size==2;
    diffs=line_pt_2-line_pt_1;
    epsilon=1e-8;
    if np.sum(np.abs(diffs))<epsilon:
        print 'naner!';
        distance=float('nan')
    elif np.abs(diffs[0])<epsilon:
        distance=np.abs(pt[0]-line_pt_1[0]);
    elif np.abs(diffs[1])<epsilon:
        distance=np.abs(pt[1]-line_pt_1[1]);
    else:
        deno=np.sqrt(np.sum(np.power(diffs,2)));
        numo=(diffs[0]*pt[1])-(diffs[1]*pt[0])+(line_pt_1[0]*line_pt_2[1])-(line_pt_1[1]*line_pt_2[0]);
        numo=np.abs(numo);
        distance=numo/deno;
    
    return distance;

def getAllPointLineDistancesThreePts(pts):
    assert pts.shape[0]==3;
    distances=[];
    for i in range(pts.shape[0]):
        line_pt_1=pts[i,:2];
        line_pt_2=pts[(i+1)%3,:2];
        pt=pts[(i+2)%3,:2];
        distance=getPointLineDistance(line_pt_1,line_pt_2,pt);
        distances.append(distance);
    return distances;

def getValidPoints(horse_file,human_file):
    horse_pts=np.load(horse_file);
    human_pts=np.load(human_file);
    check_pts=np.logical_and(horse_pts[:,2]>0,human_pts[:,2]>0);
    valid_horse=horse_pts[check_pts,:];
    valid_human=human_pts[check_pts,:];
    assert valid_horse.shape[0]==valid_human.shape[0];
    assert np.all(valid_horse[:,2]>0);
    assert np.all(valid_human[:,2]>0)

    return valid_horse,valid_human;

def main():
    print 'hello from preprocessing_data';
    return
    params_dict={};
    params_dict['path_txt'] = '/home/laoreja/new-deep-landmark/dataset/train/aflw_valImageList.txt';
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'aflw';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_check/aflw'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list_val.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list_val.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs_val.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem_val.txt');
    params_dict['overwrite'] = False;
    params_dict['resize']=(224,224);

    params=createParams('makeBboxPairFiles');
    params=params(**params_dict);
    script_makeBboxPairFiles(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_meta,'params.p'),'wb'));


    return
    # five kp
    out_dir_meta_data='/home/SSD3/maheen-data/horse_project/data_check';
    matches_file='/home/maheenrashid/Downloads/knn_5_points_train_list_clean.txt'
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    face_data_list_file=os.path.join(out_dir_meta_data,'aflw','npy','data_list.txt');

    out_dir_meta_horse = os.path.join(out_dir_meta_data,'horse');
    out_dir_meta_horse_list = [os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face = os.path.join(out_dir_meta_data,'aflw');
    out_dir_meta_face_list = [os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    
    out_dir_meta_face_old=os.path.join(out_dir_meta_data,'face');
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
    out_file_face_noIm=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_fiveKP_noIm.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_train_fiveKP.txt');
    resize=(224,224)

    params_dict={};
    params_dict['num_neighbors'] = num_neighbors;
    params_dict['matches_file'] = matches_file;
    params_dict['face_data_file'] = face_data_file;
    params_dict['out_dir_meta_horse'] = out_dir_meta_horse_list;
    params_dict['out_dir_meta_face'] = out_dir_meta_face_list;
    params_dict['out_file_horse'] = out_file_horse;
    params_dict['out_file_face'] = out_file_face;
    params_dict['out_file_face_noIm'] = out_file_face_noIm;
    params_dict['out_dir_meta_face_old'] = out_dir_meta_face_old_list;
    params_dict['resize'] = resize;
    params_dict['threshold']=11.2;

    params=createParams('makeMatchFile');
    params=params(**params_dict);

    makeMatchFile(params);

    pickle.dump(params._asdict(),open(os.path.join(out_dir_meta_data,'Params_makeMatchFile.p'),'wb'));

    # all
    out_dir_meta_data='/home/SSD3/maheen-data/horse_project/data_check';
    matches_file='/home/laoreja/data/knn_res_new/knn_train_list.txt';
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    face_data_list_file=os.path.join(out_dir_meta_data,'aflw','npy','data_list.txt');
    out_dir_meta_horse = os.path.join(out_dir_meta_data,'horse');
    out_dir_meta_horse_list = [os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face = os.path.join(out_dir_meta_data,'aflw');
    out_dir_meta_face_list = [os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    out_dir_meta_face_old=os.path.join(out_dir_meta_data,'face');
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_allKP.txt');
    out_file_face_noIm=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_train_allKP_noIm.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_train_allKP.txt');
    resize=(224,224)

    params_dict={};
    params_dict['num_neighbors'] = num_neighbors;
    params_dict['matches_file'] = matches_file;
    params_dict['face_data_file'] = face_data_file;
    params_dict['out_dir_meta_horse'] = out_dir_meta_horse_list;
    params_dict['out_dir_meta_face'] = out_dir_meta_face_list;
    params_dict['out_file_horse'] = out_file_horse;
    params_dict['out_file_face'] = out_file_face;
    params_dict['out_file_face_noIm'] = out_file_face_noIm;
    params_dict['out_dir_meta_face_old'] = out_dir_meta_face_old_list;
    params_dict['resize'] = resize;
    params_dict['threshold']=11.2;

    params=createParams('makeMatchFile');
    params=params(**params_dict);

    makeMatchFile(params);

    pickle.dump(params._asdict(),open(os.path.join(out_dir_meta_data,'Params_makeMatchFile.p'),'wb'));


    # val
    out_dir_meta_data='/home/SSD3/maheen-data/horse_project/data_check';
    matches_file='/home/SSD3/maheen-data/temp/knn_all_points_val_list.txt';
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    face_data_list_file=os.path.join(out_dir_meta_data,'aflw','npy','data_list.txt');
    out_dir_meta_horse = os.path.join(out_dir_meta_data,'horse');
    out_dir_meta_horse_list = [os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face = os.path.join(out_dir_meta_data,'aflw');
    out_dir_meta_face_list = [os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    out_dir_meta_face_old=os.path.join(out_dir_meta_data,'face');
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_allKP.txt');
    out_file_face_noIm=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_allKP_noIm.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_val_allKP.txt');
    resize=(224,224)

    params_dict={};
    params_dict['num_neighbors'] = num_neighbors;
    params_dict['matches_file'] = matches_file;
    params_dict['face_data_file'] = face_data_file;
    params_dict['out_dir_meta_horse'] = out_dir_meta_horse_list;
    params_dict['out_dir_meta_face'] = out_dir_meta_face_list;
    params_dict['out_file_horse'] = out_file_horse;
    params_dict['out_file_face'] = out_file_face;
    params_dict['out_file_face_noIm'] = out_file_face_noIm;
    params_dict['out_dir_meta_face_old'] = out_dir_meta_face_old_list;
    params_dict['resize'] = resize;
    params_dict['threshold']=11.2;

    params=createParams('makeMatchFile');
    params=params(**params_dict);

    makeMatchFile(params);

    pickle.dump(params._asdict(),open(os.path.join(out_dir_meta_data,'Params_makeMatchFile.p'),'wb'));


    return

    # horse
    params_dict={};
    params_dict['path_txt'] ='/home/laoreja/new-deep-landmark/dataset/train/trainImageList_2.txt'
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'horse';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_check/horse'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem.txt');
    params_dict['overwrite'] = False;
    params_dict['resize']=(224,224);

    params=createParams('makeBboxPairFiles');
    params=params(**params_dict);
    script_makeBboxPairFiles(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_meta,'params.p'),'wb'));

    # face
    params_dict={};
    params_dict['path_txt'] = '/home/laoreja/deep-landmark-master/dataset/train/trainImageList.txt';
    params_dict['path_pre'] = '/home/laoreja/deep-landmark-master/dataset/train';
    params_dict['type_data'] = 'face';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_check/face'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem.txt');
    params_dict['overwrite'] = False;
    params_dict['resize']=(224,224);

    params=createParams('makeBboxPairFiles');
    params=params(**params_dict);
    script_makeBboxPairFiles(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_meta,'params.p'),'wb'));

    # aflw
    params_dict={};
    params_dict['path_txt'] = '/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'aflw';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_check/aflw'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem.txt');
    params_dict['overwrite'] = False;
    params_dict['resize']=(224,224);

    params=createParams('makeBboxPairFiles');
    params=params(**params_dict);
    script_makeBboxPairFiles(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_meta,'params.p'),'wb'));

    #val    
    params_dict={};
    params_dict['path_txt'] ='/home/laoreja/new-deep-landmark/dataset/train/valImageList_2.txt'
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'horse';
    params_dict['out_dir_meta'] = '/home/SSD3/maheen-data/horse_project/data_check/horse'
    util.mkdir(params_dict['out_dir_meta']);
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list_val.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list_val.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs_val.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem_val.txt');
    params_dict['overwrite'] = False;
    params_dict['resize']=(224,224);

    params=createParams('makeBboxPairFiles');
    params=params(**params_dict);
    script_makeBboxPairFiles(params)
    pickle.dump(params._asdict(),open(os.path.join(params.out_dir_meta,'params_val.p'),'wb'));




    return
    # all
    out_dir_meta_data='/home/SSD3/maheen-data/horse_project/data_resize';
    matches_file='/home/SSD3/maheen-data/temp/knn_all_points_val_list.txt';
    face_data_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_trainImageList.txt';
    face_data_list_file=os.path.join(out_dir_meta_data,'aflw','npy','data_list.txt');
    out_dir_meta_horse = os.path.join(out_dir_meta_data,'horse');
    out_dir_meta_horse_list = [os.path.join(out_dir_meta_horse,'im'),os.path.join(out_dir_meta_horse,'npy')];
    out_dir_meta_face = os.path.join(out_dir_meta_data,'aflw');
    out_dir_meta_face_list = [os.path.join(out_dir_meta_face,'im'),os.path.join(out_dir_meta_face,'npy')];
    out_dir_meta_face_old=os.path.join(out_dir_meta_data,'face');
    out_dir_meta_face_old_list=[os.path.join(out_dir_meta_face_old,'im'),os.path.join(out_dir_meta_face_old,'npy')];
    num_neighbors=5;
    out_file_face=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_allKP.txt');
    out_file_face_noIm=os.path.join(out_dir_meta_face,'matches_'+str(num_neighbors)+'_val_allKP_noIm.txt');
    out_file_horse=os.path.join(out_dir_meta_horse,'matches_'+str(num_neighbors)+'_val_allKP.txt');
    resize=(224,224)

    params_dict={};
    params_dict['num_neighbors'] = num_neighbors;
    params_dict['matches_file'] = matches_file;
    params_dict['face_data_file'] = face_data_file;
    params_dict['out_dir_meta_horse'] = out_dir_meta_horse_list;
    params_dict['out_dir_meta_face'] = out_dir_meta_face_list;
    params_dict['out_file_horse'] = out_file_horse;
    params_dict['out_file_face'] = out_file_face;
    params_dict['out_file_face_noIm'] = out_file_face_noIm;
    params_dict['out_dir_meta_face_old'] = out_dir_meta_face_old_list;
    params_dict['resize'] = resize;

    params=createParams('makeMatchFile');
    params=params(**params_dict);

    makeMatchFile(params);

    pickle.dump(params._asdict(),open(os.path.join(out_dir_meta_data,'Params_makeMatchFile.p'),'wb'));

    return
    train_new='/home/laoreja/new-deep-landmark/dataset/train/trainImageList_2.txt';
    train_new='/home/laoreja/data/knn_res_new/knn_train_list.txt'
    val_old='/home/laoreja/new-deep-landmark/dataset/train/valImageList_2.txt';
    train_new='/home/SSD3/maheen-data/temp/knn_all_points_val_list.txt';
    print train_new
    # /home/laoreja/data/knn_res_new/knn_train_list.txt
    
    


    # dump_script_makeBBoxPairFiles()
    # dump_makeMatchFile();



    # train_new='/home/laoreja/new-deep-landmark/dataset/train/trainImageList_2.txt'
    # val_old='/home/laoreja/new-deep-landmark/dataset/train/valImageList_2.txt';

    # train_new='/home/laoreja/new-deep-landmark/dataset/train/horse_final_trainImageList.txt'
    # val_old='/home/laoreja/new-deep-landmark/dataset/train/horse_final_valImageList.txt';


    train_new=util.readLinesFromFile(train_new);
    val_old=util.readLinesFromFile(val_old);
    train_new=[line_curr.split(' ')[0] for line_curr in train_new];
    val_old=[line_curr.split(' ')[0] for line_curr in val_old];
    print len(val_old),len(train_new)
    print val_old[0],train_new[0];
    overlap=[file_curr for file_curr in train_new if file_curr in val_old];
    print len(overlap);




if __name__=='__main__':
    main();