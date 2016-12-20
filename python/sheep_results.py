import util;
import os;
import numpy as np;
import visualize;
dir_server='/home/SSD3/maheen-data';
click_str='http://vision1.idav.ucdavis.edu:1000';

def parseAnnoStr(annos):
    annos=[int(num) for num in annos];
    annos=np.array(annos);
    annos=np.reshape(annos,(len(annos)/2,2));
    return annos;

def readGTFile(file_curr):
    lines=util.readLinesFromFile(file_curr);
    im_paths=[];
    ims_size=[];
    annos_all=[];
    for line_curr in lines:
        line_split=line_curr.rsplit(None,14);
        im_paths.append(line_split[0]);
        
        im_size=line_split[1:1+4];
        im_size=[int(num) for num in im_size];
        im_size=[im_size[2]-im_size[0],im_size[3]-im_size[1]];
        ims_size.append(im_size);
        
        annos=line_split[1+4:];
        annos=parseAnnoStr(annos);
        annos_all.append(annos);
        
    return im_paths,ims_size,annos_all;
        
def readPredFile(pred_file):
    lines=util.readLinesFromFile(pred_file);
    annos_all=[];
    for line_curr in lines:
        annos_str=line_curr.split();
        annos=parseAnnoStr(annos_str);
        annos_all.append(annos);
    return annos_all;

def getDiffs(annos_gt,annos_pred):
    diffs_all=[];
    for anno_gt,anno_pred in zip(annos_gt,annos_pred):
        diffs=np.power(anno_gt-anno_pred,2);
        diffs=np.sum(diffs,1);
        diffs=np.power(diffs,0.5);
        diffs_all.append(diffs);
    return diffs_all;

def getErrorPercentageImSize(im_sizes,diffs_all):
    errors_all=[];
    for im_size,diffs in zip(im_sizes,diffs_all):
        assert (im_size[0]==im_size[1]);
        errors=-1*np.ones(diffs.shape);
        errors[diffs>0]=diffs[diffs>0]/im_size[0];
        errors_all.append(errors);
    return errors_all;

def us_getFilePres(gt_file,out_dir_us,post_us,num_iter,batch_us):
    files_gt=[];
    files_pred=[];
    im_paths=util.readLinesFromFile(gt_file);
    im_paths=[im_path[:im_path.index(' ')] for im_path in im_paths];
    num_gt=len(im_paths);
    count=0;
    for batch_num in range(num_iter):
        for im_num in range(batch_us):
            file_pre=str(batch_num+1)+'_'+str(im_num+1);
            file_gt=file_pre+post_us[0];
            file_pred=file_pre+post_us[1];
            files_gt.append(os.path.join(out_dir_us,file_gt));
            files_pred.append(os.path.join(out_dir_us,file_pred));
#             print file_pred;
#             print file_gt
            
    files_gt=files_gt[:num_gt];
    files_pred=files_pred[:num_gt];
    return im_paths,files_gt,files_pred;

def us_getPredGTPairs(gt_pt_files,pred_pt_files):
    annos_gt=[];
    annos_pred=[];
    for gt_file,pred_file in zip(gt_pt_files,pred_pt_files):
        gt_pts=np.load(gt_file);
        pred_pts=np.load(pred_file);
        bin_keep=gt_pts[:,2]>0
        gt_pts=gt_pts[bin_keep,:2];
        pred_pts=pred_pts[bin_keep,:2];
        annos_gt.append(gt_pts);
        annos_pred.append(pred_pts);
        
    return annos_gt,annos_pred;

def us_getDiffs(gt_pt_files,pred_pt_files):
    diffs_all=[];
    for gt_file,pred_file in zip(gt_pt_files,pred_pt_files):
        gt_pts=np.load(gt_file);
        pred_pts=np.load(pred_file);
        bin_keep=gt_pts[:,2]>0
        diffs_curr=-1*np.ones((gt_pts.shape[0],));
        gt_pts=gt_pts[bin_keep,:2];
        pred_pts=pred_pts[bin_keep,:2];
        diffs=np.power(gt_pts-pred_pts,2);
        diffs=np.sum(diffs,1);
        diffs=np.power(diffs,0.5);
        diffs_curr[bin_keep]=diffs;
        diffs_all.append(diffs_curr);
    return diffs_all;


def us_getErrorsAll(gt_file,out_dir_us,post_us,num_iter,batch_size):
    im_paths,gt_pt_files,pred_pt_files=us_getFilePres(gt_file,out_dir_us,post_us,num_iter,batch_size);
    
#     annos_gt,annos_pred=us_getPredGTPairs(gt_pt_files,pred_pt_files);
#     diffs_all=getDiffs(annos_gt,annos_pred);
    diffs_all=us_getDiffs(gt_pt_files,pred_pt_files);
    im_sizes=[[2,2]]*len(diffs_all)
    errors_all=getErrorPercentageImSize(im_sizes,diffs_all);
    return errors_all;

def them_getErrorsAll(gt_file,pred_file):
    im_paths,im_sizes,annos_gt=readGTFile(gt_file);
    annos_pred=readPredFile(pred_file);
    diffs_all=getDiffs(annos_gt,annos_pred);
    errors_all=getErrorPercentageImSize(im_sizes,diffs_all);
    return errors_all;

def plotComparisonCurve(errors_all,out_file,labels):
    vals=[];
    for err in errors_all:
        err=np.array(err);
#         avg=np.mean(err,1);
        bin_keep=err>=0;
        err[err<0]=0;
        div=np.sum(bin_keep,1);
        sum_val=np.sum(err,1).astype(np.float);
        avg=sum_val/div;
        
        avg=np.sort(avg);
        vals.append(avg);
        
    xAndYs=[(range(len(val_curr)),val_curr) for val_curr in vals];
    xlabel='Sorted Image Number';
    ylabel='BBox Normalized Error';
    visualize.plotSimple(xAndYs,out_file,xlabel=xlabel,ylabel=ylabel,legend_entries=labels);
    
#     plotSimple(xAndYs,out_file,title='',xlabel='',ylabel='',legend_entries=None,loc=0,outside=False,logscale=False)
        
def plotComparisonKpAvgError(errors_all,out_file,ticks,labels,xlabel=None,ylabel=None,colors=None,ylim=None,title=''):
    vals={};
    for err,label_curr in zip(errors_all,labels):
        err=np.array(err);
        bin_keep=err>=0;
        err[err<0]=0;
        div=np.sum(bin_keep,0);
        sum_val=np.sum(err,0).astype(np.float);
        avg=sum_val/div;
#         avg=np.mean(err,0);
        
        
        vals[label_curr]=avg;

    if colors is None:
        colors=['b','g'];
    if xlabel is None:
        xlabel='Keypoint';
        
    if ylabel is None:
        ylabel='BBox Normalized Error';
        
    visualize.plotGroupBar(out_file,vals,ticks,labels,colors,xlabel=xlabel,ylabel=ylabel,\
                           width=1.0/len(vals),ylim=ylim,title=title);
#     ,title='',width=0.25,ylim=None)


def getErrRates(err,thresh=0.1):
#     vals=[];
#     total_errRate=[];
#     for err in zip(errors_all):
    err=np.array(err);
#     print err.shape;
    sum_errs=np.sum(err>thresh,0).astype(np.float);

    total_errs=np.sum(err>=0,0);
#     print total_errs.shape;
#     print sum_errs.shape;
    err_rate=sum_errs/total_errs*100.0;
#         print err_rate;
    sum_errs_tot=np.sum(sum_errs);
    total_errs_tot=np.sum(total_errs);
    err_rate_tot=sum_errs_tot/total_errs_tot*100.0;
#         total_errRate.append(err_rate_tot);
#         vals.append(err_rate);
    return err_rate,err_rate_tot;

def plotComparisonKpError(errors_all,out_file,ticks,labels,xlabel=None,ylabel=None,colors=None,thresh=0.1,\
                          title='',ylim=None):
    vals={};
    for err,label_curr in zip(errors_all,labels):
        err=np.array(err);
        sum_errs=np.sum(err>thresh,0).astype(np.float);
        total_errs=np.sum(err>=0,0);
#         print total_errs;
#         print sum_errs;
        err_rate=sum_errs/total_errs*100.0;
#         print err_rate;
        vals[label_curr]=err_rate;

    if colors is None:
        colors=['b','g'];
        
    if xlabel is None:
        xlabel='Keypoint';
        
    if ylabel is None:
        ylabel='Failure Rate %';
        
    visualize.plotGroupBar(out_file,vals,ticks,labels,colors,xlabel=xlabel,ylabel=ylabel,\
                           width=1.0/len(vals),title=title,ylim=ylim);

#     pass;

def writeJustTestScript(out_file_sh,val_data_path,iterations,batch_size,model_out_tups,face):    
    file_th='/home/maheenrashid/Downloads/horses/torch/justTest.th';
    commands_all=[];
    for model_path_curr,out_dir_curr in model_out_tups:
        command_curr=['th',file_th];
        command_curr=command_curr+['-val_data_path',val_data_path];
        command_curr=command_curr+['-iterations',str(iterations)];
        command_curr=command_curr+['-batchSize',str(batch_size)];
        command_curr=command_curr+['-full_model_path',model_path_curr];
        command_curr=command_curr+['-outDirTest',out_dir_curr];
        if face:
            command_curr=command_curr+['-face'];
        command_curr=' '.join(command_curr);
#         print command_curr;
        commands_all.append(command_curr);
    
    util.writeFile(out_file_sh,commands_all);
    print len(commands_all);
    print out_file_sh;


def sheepPeopleComparisonScript():
    dir_sheep_results=os.path.join(dir_server,'horse_project/sheep_baseline_results');
    dir_input_data='/home/SSD3/maheen-data/horse_project/files_for_sheepCode'
    util.mkdir(dir_sheep_results);
    
    us_test=['sheep_test_us_sheep_minloss.txt','horse_test_us_horse_minloss.txt'];
    us_test=[os.path.join(dir_input_data,file_curr) for file_curr in us_test]
    out_us=[os.path.join(dir_sheep_results,'sheep_us/test_images'),\
    os.path.join(dir_sheep_results,'horse_us/test_images')];
    batch_size=50;
    num_iter=2;
    post_us=['_gt_pts.npy','_pred_pts.npy']
    im_size=2;
    
    them_test=['sheep_test.txt','horse_test.txt'];
    them_test=[os.path.join(dir_input_data,file_curr) for file_curr in them_test];
    out_them=[file_curr[:file_curr.rindex('.')]+'_TIF_result.txt' for file_curr in them_test];
    
    out_file_curve_pre=os.path.join(dir_sheep_results,'curve_comparison');
#                                 .png');
    out_file_curve_3_pre=os.path.join(dir_sheep_results,'curve_comparison_just3');
#                                   .png');
    
    out_file_kp_avg_pre=os.path.join(dir_sheep_results,'avg_kp_comparison');
    out_file_kp_err_pre=os.path.join(dir_sheep_results,'failure_kp_comparison')
    
    labels=['Ours','TIF'];
#              Sheep'],['Ours Horse','TIF Horse']];
    out_file_post_tags=[ 'Sheep','Horse']
    ticks=['LE','RE','N','LM','RM'];
    
    errors_all=[];
    
    for gt_file_them,pred_file_them,gt_file_us,out_dir_us,out_file_post_tag_curr in \
            zip(them_test,out_them,us_test,out_us,out_file_post_tags):
        us_errors_all=us_getErrorsAll(gt_file_us,out_dir_us,post_us,num_iter,batch_size)
        them_errors_all=them_getErrorsAll(gt_file_them,pred_file_them);

        errors_all=[us_errors_all,them_errors_all];
        
        labels_curr=[label_curr+'\n'+out_file_post_tag_curr for label_curr in labels];
        # if out_file_post_tag_curr=='Sheep':
        #     labels_curr[0]=labels_curr[0]+' ';
        
        out_file_curve=out_file_curve_pre+'_'+out_file_post_tag_curr+'.pdf';
        plotComparisonCurve(errors_all,out_file_curve,labels_curr);
        out_file_curve=out_file_curve_pre+'_'+out_file_post_tag_curr+'.png';
        plotComparisonCurve(errors_all,out_file_curve,labels_curr);
        print out_file_curve.replace(dir_server,click_str);
        
        errors_3=[np.array(err)[:,:2] for err in errors_all];
        # labels_curr=labels;
        # title=out_file_post_tag_curr;
#         out_file_curve_3=out_file_curve_3_pre+'_'+out_file_post_tag_curr+'.pdf';
#         plotComparisonCurve(errors_3,out_file_curve_3,labels_curr);
#         out_file_curve_3=out_file_curve_3_pre+'_'+out_file_post_tag_curr+'.png';
#         plotComparisonCurve(errors_3,out_file_curve_3,labels_curr);
#         print out_file_curve_3.replace(dir_server,click_str);
        ylim=None
        title=''
        out_file_kp_avg=out_file_kp_avg_pre+'_'+out_file_post_tag_curr+'.pdf';
        plotComparisonKpAvgError(errors_all,out_file_kp_avg,ticks,labels_curr,ylim=ylim);
        out_file_kp_avg=out_file_kp_avg_pre+'_'+out_file_post_tag_curr+'.png';
        plotComparisonKpAvgError(errors_all,out_file_kp_avg,ticks,labels_curr,title=title,ylim=ylim);

        print out_file_kp_avg.replace(dir_server,click_str);
        ylim=None;
        out_file_kp_err=out_file_kp_err_pre+'_'+out_file_post_tag_curr+'.pdf';
        plotComparisonKpError(errors_all,out_file_kp_err,ticks,labels_curr,ylim=ylim);
        out_file_kp_err=out_file_kp_err_pre+'_'+out_file_post_tag_curr+'.png';
        plotComparisonKpError(errors_all,out_file_kp_err,ticks,labels_curr,title=title,ylim=ylim);

        print out_file_kp_err.replace(dir_server,click_str);



def ourComparisonScript():        
    val_data_path='/home/SSD3/maheen-data/horse_project/data_check/horse/matches_5_val_allKP_minLoss_clean.txt';
    dir_figures='/home/SSD3/maheen-data/horse_project/cvpr_figs';
    iterations=2;
    batch_size=100;
    post_us=['_gt_pts.npy','_pred_pts.npy']
    ticks=['LE','RE','N','LM','RM'];
    
    out_dir_us=os.path.join(dir_figures,'full_system_2loss_tps/test_images');
    out_dir_us_affine=os.path.join(dir_figures,'full_system_2loss_affine/test_images');
    out_dir_bl_ft=os.path.join(dir_figures,'baseline_finetune/test_images');
    out_dir_bl_tps=os.path.join(dir_figures,'baseline_tps/test_images');
    out_dir_bl_affine=os.path.join(dir_figures,'baseline_affine/test_images');
    
    out_dir_us_sheep=os.path.join(dir_figures,'full_system_2loss_tps_sheep/test_images');
    out_dir_bl_ft_sheep=os.path.join(dir_figures,'baseline_finetune_sheep/test_images');
    out_dir_bl_tps_sheep=os.path.join(dir_figures,'baseline_tps_sheep/test_images');


    # dirs_test_all=[out_dir_us,out_dir_bl_ft,out_dir_bl_tps];
    # labels=['Ours','BL Finetune','BL TPS'];
    # out_file_curve=os.path.join(dir_figures,'curve_comparison_us_ft_tps.pdf');
    # out_file_kp_err=os.path.join(dir_figures,'failure_kp_comparison_us_ft_tps.pdf');
    # out_file_kp_avg=os.path.join(dir_figures,'avg_kp_comparison_us_ft_tps.pdf');
    # ylim=[0,0.09];
    # colors=['b','g','r'];
    
#     dirs_test_all=[out_dir_us,out_dir_us_affine,out_dir_bl_ft,out_dir_bl_tps,out_dir_bl_affine];
#     labels=['Ours TPS','Ours Affine','BL Finetune','BL TPS','BL Affine'];
#     out_file_curve=os.path.join(dir_figures,'curve_comparison_us_ft_tps_affine.pdf');
#     out_file_kp_err=os.path.join(dir_figures,'failure_kp_comparison_us_ft_tps_affine.pdf');
#     out_file_kp_avg=os.path.join(dir_figures,'avg_kp_comparison_us_ft_tps_affine.pdf');
#     ylim=[0,0.11]
#     colors=['b','c','g','r','y'];
    
    iterations=2;
    batch_size=50;

    dirs_test_all=[out_dir_us_sheep,out_dir_bl_ft_sheep,out_dir_bl_tps_sheep];
    labels=['Ours','BL Finetune','BL TPS'];
    out_file_curve=os.path.join(dir_figures,'curve_comparison_us_ft_tps_sheep.pdf');
    out_file_kp_err=os.path.join(dir_figures,'failure_kp_comparison_us_ft_tps_sheep.pdf');
    out_file_kp_avg=os.path.join(dir_figures,'avg_kp_comparison_us_ft_tps_sheep.pdf');
    ylim=[0,0.06];
    colors=['b','g','r'];
    
#     dirs_test_all=[out_dir_us_sheep,out_dir_bl_ft_sheep]
#     labels=['Ours','BL Finetune'];
#     out_file_curve=os.path.join(dir_figures,'curve_comparison_us_ft_sheep.pdf');
#     out_file_kp_err=os.path.join(dir_figures,'failure_kp_comparison_us_ft_sheep.pdf');
#     out_file_kp_avg=os.path.join(dir_figures,'avg_kp_comparison_us_ft_sheep.pdf');
#     ylim=None;
#     colors=['b','g'];

    
    errors_all=[];
    for out_dir_test in dirs_test_all:
        im_paths,gt_pt_files,pred_pt_files=us_getFilePres(val_data_path,out_dir_test,post_us,iterations,batch_size);
        errors_curr=us_getErrorsAll(val_data_path,out_dir_test,post_us,iterations,batch_size);
        errors_all.append(errors_curr)

    plotComparisonCurve(errors_all,out_file_curve,labels);
    print out_file_curve.replace(dir_server,click_str);

    plotComparisonKpError(errors_all,out_file_kp_err,ticks,labels,colors=colors,ylim=[0,12]);
    print out_file_kp_err.replace(dir_server,click_str);
    
    # plotComparisonKpAvgError(errors_all,out_file_kp_avg,ticks,labels,colors=colors,ylim=ylim);
    # print out_file_kp_avg.replace(dir_server,click_str);



sheepPeopleComparisonScript()
# ourComparisonScript()