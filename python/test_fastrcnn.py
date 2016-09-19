import numpy as np;
import matplotlib
import numpy as np;
matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import os;
import scipy.misc;
import util;
import visualize;

def saveDets(im_file,class_name,dets_file,out_file,thresh):
    dets=np.load(dets_file);
    # print np.min(dets[:,-1]),np.max(dets[:,-1]),dets.shape

    inds = np.where(dets[:, -1] >= thresh)[0]
    # print len(inds)
    if len(inds) == 0:
        print 'PROBLEM ',im_file,np.min(dets[:,-1]),np.max(dets[:,-1]),dets.shape
    #     return
    im=scipy.misc.imread(im_file);
    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(im, aspect='equal')
    ax=plt.gca();
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if out_file is not None:
        plt.savefig(out_file);
        plt.close('all');


def script_saveDetectedImages():
    out_file='/disk2/aprilExperiments/horses/list_of_frames.txt';
    out_dir_dets='/disk2/aprilExperiments/horses/frames_with_detections';
    post_pend='_horse_detections.npy';
    post_pend_im='_horse_detections.png';
    class_name='horse'
    thresh=0.5;

    frames=util.readLinesFromFile(out_file);
    # frames=['/disk2/aprilExperiments/horses/mediaFromPPT_frames/media7_00014.jpg']
    for idx_frame_curr,frame_curr in enumerate(frames):
        if idx_frame_curr%100==0:
            print idx_frame_curr,len(frames);

        frame_name=frame_curr[frame_curr.rindex('/')+1:];
        frame_name=frame_name[:frame_name.rindex('.')];
        video_name=frame_name[:frame_name.index('_')];
        dets_file=os.path.join(out_dir_dets,video_name,frame_name+post_pend)
        out_file_im=dets_file.replace('.npy','.png');
        saveDets(frame_curr,class_name,dets_file,out_file_im,thresh)

def main():
    
    out_file_html='/disk2/aprilExperiments/horses/frames_with_detections/visualize.html';
    out_dir_meta='/disk2/aprilExperiments/horses/frames_with_detections';
    img_paths=[];
    captions=[];
    rel_path=['/disk2','../../../..'];
    for dir_curr in os.listdir(out_dir_meta):
        dir_curr=os.path.join(out_dir_meta,dir_curr);
        if os.path.isdir(dir_curr):
            print dir_curr
            jpegs=[os.path.join(dir_curr,file_curr) for file_curr in os.listdir(dir_curr) if file_curr.endswith('.png')];
            jpegs=[file_curr.replace(rel_path[0],rel_path[1]) for file_curr in jpegs];
            # print jpegs[:10];
            jpegs.sort();
            # print jpegs[:10];
            captions_curr=['']*len(jpegs);
            print captions_curr;
            img_paths.append(jpegs);
            captions.append(captions_curr);
            # raw_input();
    visualize.writeHTML(out_file_html,img_paths,captions,height=100,width=100);



    return
    dirs_meta=['/disk2/aprilExperiments/horses/mediaFromPPT_frames','/disk2/aprilExperiments/horses/ResearchSpring2016_frames'];
    out_file='/disk2/aprilExperiments/horses/list_of_frames.txt'
    im_list=[];
    for dir_curr in dirs_meta:
        list_curr=[os.path.join(dir_curr,im_curr) for im_curr in os.listdir(dir_curr) if im_curr.endswith('.jpg')];
        im_list=im_list+list_curr;
    util.writeFile(out_file,im_list);



    return
    in_file = '/disk2/aprilExperiments/horses/list_of_frames.txt';
    out_dir_meta = '/disk2/aprilExperiments/horses/frames_with_detections/';
    util.mkdir(out_dir_meta);

    with open(in_file,'rb') as f:
        im_names = f.readlines();
    im_names = [line.strip('\n') for line in im_names];

    for im_name in im_names:
        vid_name=im_name[im_name.rindex('/')+1:im_name.rindex('_')]
        out_dir_curr=os.path.join(out_dir_meta,vid_name);
        if not os.path.exists(out_dir_curr):
            os.mkdir(out_dir_curr);


    return
    out_dir='/disk2/temp/horses';
    arr_file=os.path.join(out_dir,'Outside4_00011_horse_detections.npy');
    im_file ='/disk2/aprilExperiments/horses/ResearchSpring2016_frames/Outside4_00011.jpg';
    arr=np.load(arr_file);
    
    out_file=arr_file[:-4]+'.png'
    saveDets(im_file,'horse',arr,out_file,0.8)
    # plt.imshow(im);
    # plt.savefig();
    print 'done';


if __name__=='__main__':
    main();