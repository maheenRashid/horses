import util;
import numpy as np;
import visualize;
# import sklearn;
import os;
import time;
from sklearn.cluster import KMeans

def getDataAndLabels(data_path,batchSize,numBatches,fc6_dir):
    lines=util.readLinesFromFile(data_path);
    out_data=np.zeros((len(lines),4096));
    start_curr=0;
    for i in range(numBatches):
        file_curr=os.path.join(fc6_dir,str(i+1)+'.npy');
        fc6_curr=np.load(file_curr);
        end_curr=min(start_curr+fc6_curr.shape[0],len(lines));
        len_curr=end_curr-start_curr
        out_data[start_curr:end_curr,:]=fc6_curr[:len_curr];
        start_curr=end_curr;
    return out_data,lines;

def normalizeData(data):
    norm_data=np.linalg.norm(data,axis=1);
    norm_data=np.expand_dims(norm_data,axis=1);
    norm_data=np.tile(norm_data,(1,data.shape[1]));
    data=data/norm_data;
    return data;

def getClusterIdx(data,num_clusters):
#     norm_data=np.linalg.norm(data,axis=1);
#     norm_data=np.expand_dims(norm_data,axis=1);
#     norm_data=np.tile(norm_data,(1,data.shape[1]));
#     data=data/norm_data;
    data=normalizeData(data)
    kmeaner=KMeans(n_clusters=num_clusters,n_jobs=12);
    cluster_idx=kmeaner.fit_predict(data);
    return cluster_idx;

def saveClusterIdx(data_path,batchSize,numBatches,fc6_dir,num_clusters,out_file_clusters):
    data,labels = getDataAndLabels(data_path,batchSize,numBatches,fc6_dir);
    cluster_idx = getClusterIdx(data,num_clusters);
    print out_file_clusters,cluster_idx.shape
    np.save(out_file_clusters,cluster_idx);

def makeClusterHTML(out_file_html,labels,num_cols,size_im,dir_server):
    ims=[];
    captions=[];
    start_idx=0;
    while start_idx<len(labels):
        row_curr=[];
        caption_curr=[];
        if start_idx+num_cols>len(labels):
            num_cols_real=len(labels)-start_idx;
        else:
            num_cols_real=num_cols;
        for col_no in range(num_cols_real):
            idx_curr=start_idx+col_no;
            label_curr=labels[idx_curr];
            row_curr.append(util.getRelPath(label_curr,dir_server));
            caption_curr.append('');
        ims.append(row_curr);
        captions.append(caption_curr);
        start_idx=start_idx+num_cols_real;
    visualize.writeHTML(out_file_html,ims,captions,size_im,size_im);
    print out_file_html.replace(dir_server,'http://vision1.idav.ucdavis.edu:1000')

def script_makeClusterHTML(labels,clusters,out_dir):
    labels=np.array(labels);
    clusters_uni=list(set(clusters));
    labels_clusters=[];
    for cluster_idx in clusters_uni:
        labels_rel=labels[clusters==cluster_idx];
        print len(labels_rel);
        labels_clusters.append(labels_rel);
        out_file_html=os.path.join(out_dir,str(cluster_idx)+'.html');
        num_cols=40;
        size_im=20;
        makeClusterHTML(out_file_html,labels_rel,num_cols,size_im,dir_server);

    
print 'hello';
data_path='/home/SSD3/maheen-data/horse_project/data_check/horse/pairs_all.txt';
dir_server='/home/SSD3/maheen-data';
out_dir='/home/SSD3/maheen-data/horse_project/sanityCheckHorse'
batchSize=128;
numBatches=30;
fc6_dir=os.path.join(out_dir,'fc6');
num_clusters=8;
out_file_clusters=os.path.join(fc6_dir,'clusters.npy');
# saveClusterIdx(data_path,batchSize,numBatches,fc6_dir,num_clusters,out_file_clusters);

# labels=util.readLinesFromFile(data_path);
# labels=[line_curr[:line_curr.index(' ')] for line_curr in labels];
clusters=np.load(out_file_clusters);
# script_makeClusterHTML(labels,clusters,out_dir)

data,labels=getDataAndLabels(data_path,batchSize,numBatches,fc6_dir);
data=normalizeData(data);
clusters_uni=list(set(clusters));
for cluster_idx in clusters_uni:
    data_rel=data[clusters==cluster_idx,:];
    labels_rel=data[clusters==cluster_idx];
    pairwise=np.dot(data_rel,data_rel.T);
    # print np.min(pairwise),np.max(pairwise);
    np.fill_diagonal(pairwise,0);
    problems=np.where(pairwise>0.98);
    # print problems;
    # print pairwise.shape;
    print problems[0].size,np.min(pairwise),np.max(pairwise);
    # break;



    



    
