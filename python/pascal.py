import untangle;
import os;
import util;
import scipy.misc;
import matplotlib.pyplot as plt;
import multiprocessing;
import shutil;

def writeClassTextFile(train_val_txt,path_to_im,out_file):
	lines=util.readLinesFromFile(train_val_txt);
	pos_im=[];
	lines_split=[line.split(' ',1) for line in lines];
	for idx,line_split in enumerate(lines_split):
		num=int(line_split[1]);


	pos_im=[line_split[0] for line_split in lines_split if int(line_split[1])>=0];
	ims=[os.path.join(path_to_im,pos_im_curr+'.jpg') for pos_im_curr in pos_im];
	
	util.writeFile(out_file,ims);


def saveBBoxImage(out_file,path_to_anno,out_dir_im,class_name='horse'):
	files = util.readLinesFromFile(out_file);
	just_names = util.getFileNames(files,ext=False);
	annotations=[os.path.join(path_to_anno,just_name+'.xml') for just_name in just_names];
	
	print len(annotations)
	for im_file,anno,just_name in zip(files,annotations,just_names):
		
		out_file_pre = os.path.join(out_dir_im,just_name+'_');
		obj = untangle.parse(anno);

		for idx_object_curr,object_curr in enumerate(obj.annotation.object):
			if object_curr.name.cdata == class_name:
				out_file=out_file_pre+str(idx_object_curr)+'.jpg';
				# if os.path.exists(out_file):
				# 	continue;


				bnd_box=[object_curr.bndbox.xmin.cdata,object_curr.bndbox.ymin.cdata,object_curr.bndbox.xmax.cdata,object_curr.bndbox.ymax.cdata];
				bnd_box=[int(coord) for coord in bnd_box];

				# print bnd_box;

				im=scipy.misc.imread(im_file);
				if len(im.shape)<3:
					crop=im[bnd_box[1]:bnd_box[3],bnd_box[0]:bnd_box[2]];
				else:
					crop=im[bnd_box[1]:bnd_box[3],bnd_box[0]:bnd_box[2],:];

				print out_file;
				scipy.misc.imsave(out_file,crop);

def copyfile_wrapper((in_file,out_file,counter)):
	print counter;
	shutil.copyfile(in_file,out_file);

def main():
	# print 'hello';
	train_val_txt='/Users/maheenrashid/Dropbox (Personal)/Davis_docs/Research/VOCdevkit 2/VOC2012/ImageSets/Main/horse_trainval.txt';
	path_to_im='/Users/maheenrashid/Dropbox (Personal)/Davis_docs/Research/VOCdevkit 2/VOC2012/JPEGImages';
	path_to_anno='/Users/maheenrashid/Dropbox (Personal)/Davis_docs/Research/VOCdevkit 2/VOC2012/Annotations';

	out_dir='../pascal';
	util.mkdir(out_dir);
	
	out_file=os.path.join(out_dir,'horse.txt');

	out_dir_im='../pascal/just_horse_im';
	util.mkdir(out_dir_im);

	# saveBBoxImage(out_file,path_to_anno,out_dir_im)
	im_files=util.getFilesInFolder(out_dir_im,ext='.jpg');
	file_names=util.getFileNames(im_files,ext=True);
	batch_size=20;

	batch_idx=util.getIdxRange(len(file_names),batch_size);
	print len(batch_idx);
	args=[];
	counter=0;
	for idx_batch_start,batch_start in enumerate(batch_idx[:-1]):
		batch_end = batch_idx[idx_batch_start+1];
		im_files_rel=im_files[batch_start:batch_end];
		file_names_rel=file_names[batch_start:batch_end];
		out_dir_curr=os.path.join(out_dir_im,str(idx_batch_start));
		util.mkdir(out_dir_curr);
		for file_name,im_file_curr in zip(file_names_rel,im_files_rel):
			out_file=os.path.join(out_dir_curr,file_name);
			if not os.path.exists(out_file):
				args.append((im_file_curr,out_file,counter));
				counter+=1;

	p=multiprocessing.Pool(multiprocessing.cpu_count());
	print len(args);
	p.map(copyfile_wrapper,args);

			





	

	


if __name__=='__main__':
	main();




