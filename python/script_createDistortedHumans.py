import util;
import visualize;
import os;
import preprocessing_data
import scipy.misc;
import cv2;

# from IPython.display import Image

val_list_file='/home/laoreja/new-deep-landmark/dataset/train/aflw_valImageList.txt';
out_dir='/home/SSD3/maheen-data/temp'
dir_server='/home/SSD3/maheen-data'
val_list=util.readLinesFromFile(val_list_file);

path_im,bbox,anno_points = preprocessing_data.parseAnnoFile(val_list_file);

print len(path_im),len(bbox),len(anno_points);
print path_im[0],bbox[0],anno_points[0]

print len(val_list);

# plt.ion();
for idx,(im_path,bbox_curr,anno_points_curr) in enumerate(zip(path_im,bbox,anno_points)):
    print im_path;
    out_file_curr=os.path.join(out_dir,str(idx)+'.jpg');
    im=cv2.imread(im_path);
    cv2.rectangle(im, (bbox_curr[0],bbox_curr[2]), (bbox_curr[1],bbox_curr[3]),(255,0,0));

    cv2.imwrite(out_file_curr,im);
    print out_file_curr.replace(dir_server,'http://vision1.idav.ucdavis.edu:1000');
    if idx==5:
    	break;
    # break;