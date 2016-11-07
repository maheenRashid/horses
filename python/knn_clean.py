import math
import json
import os;

N_lms = 5
lms = ['no', 'le', 're', 'rm', 'lm']

def calc_distance(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def calc_angle(l_c1, l_c2, l_12):
	return (l_c1**2 + l_c2**2 - l_12**2) / (2*l_c1*l_c2)


# !ATTENTION lms in .txt are already in right order   

def get_pattern(exist_list):
    pattern = ''
    for i in xrange(len(lms)):
        pattern += 'T' if exist_list[i] else 'F'
    return pattern

def generate_json(list_path, json_path,exist_pattern_file):
    ignore = 0
    faces_per_img = {}
    
    with open(list_path) as fd:
        contents = fd.readlines()
        print 'contents len: ', len(contents)

    parts = [line.rsplit(' ',19) for line in contents]
    image_info = {}

    exist_pattern = {}

    for li in parts:
        im = {}
        im['path'] = li[0]
        if im['path'] not in faces_per_img:
            faces_per_img[im['path']] = 0
        else:
            faces_per_img[im['path']] += 1
            
        im_id = im['path']+'_'+str(faces_per_img[im['path']])
        # print li;
        im['bbox'] = [int(ele) if '.' not in ele else float(ele) for ele in li[1:5]]
        im['le'] = [float(ele) for ele in li[5:7]]
        im['re'] = [float(ele) for ele in li[8:10]]
        im['no'] = [float(ele) for ele in li[11:13]]
        im['lm'] = [float(ele) for ele in li[14:16]]
        im['rm'] = [float(ele) for ele in li[17:19]]

        im['existence'] = []
        im['existence'].append(int(float(li[13])) == 1)
        im['existence'].append(int(float(li[7])) == 1)
        im['existence'].append(int(float(li[10])) == 1)
        im['existence'].append(int(float(li[19])) == 1)    
        im['existence'].append(int(float(li[16])) == 1)
       
        im['pattern'] = get_pattern(im['existence'])
        if im['pattern'] not in exist_pattern:
            exist_pattern[im['pattern']] = [im_id]
        else:
            exist_pattern[im['pattern']].append(im_id)

        for i in xrange(N_lms):
            for j in xrange(i+1, N_lms):
                if im['existence'][i] and im['existence'][j]:
                    im[lms[i]+lms[j]] = calc_distance(im[lms[i]], im[lms[j]])

        if im['existence'][0] == False or (im['existence'][1] == False and im['existence'][2] == False):
            ignore += 1
            continue
        else:
            im['angles'] = {}
            for i in xrange(1, N_lms):
                j = i % 4 + 1

                if im['existence'][i] and im['existence'][j]:
                    i, j = sorted([i, j])
                    im['angles'][lms[i]+'n'+lms[j]] = calc_angle(im['no'+lms[i]], im['no'+lms[j]], im[lms[i]+lms[j]])
                else:
                    pass
                
            if im['pattern'].startswith('TTT'):
                im['mid_eye'] = (0.5 * (im['le'][0]+im['re'][0]), 0.5 * (im['le'][1]+im['re'][1]))
                im['comp_angle'] = math.atan2(im['no'][1] - im['mid_eye'][1], im['no'][0] - im['mid_eye'][0])
            elif im['pattern'].startswith('TTF') or im['pattern'].startswith('TFT'):
                exist_eye = im['le'] if im['pattern'][1] == 'T' else im['re']
                im['comp_angle'] = math.atan2(im['no'][1] - exist_eye[1], im['no'][0] - exist_eye[0])

                

        image_info[im_id] = im

    image_info_file = open(json_path, "w")
    json.dump(image_info, image_info_file)
    image_info_file.close() 
    print 'ignore num: ', ignore

    # with open(file_name,'wb') as f:
    #     for string in list_to_write:
    #         f.write(string+'\n');

    with open(exist_pattern_file,'w') as f:
        json.dump(exist_pattern,f);


    return exist_pattern    


def findKNN(horse_image_info, human_image_info, horse_group, human_group, K,  list_filename):
    
    file_names = []
    captions = []

    horse_angle_lt_zero = set()
    human_angle_lt_zero = set()
    

    for horse_id in horse_group:
        horse_im = horse_image_info[horse_id]

        file_names_row = [horse_im['path']]
        captions_row = [os.path.split(horse_id)[-1]]
        dist = []

        for human_id in human_group:
            human_im = human_image_info[human_id]
            tmp_dist = math.fabs(human_im['comp_angle'] - horse_im['comp_angle'])
            dist.append((human_id, tmp_dist))

            if human_im['comp_angle'] < 0:
                human_angle_lt_zero.add(human_id)
            if horse_im['comp_angle'] < 0:
                horse_angle_lt_zero.add(horse_id)


        dist = sorted(dist, key=lambda d:d[1])[:K]

        for item in dist:
            tmp_human = human_image_info[item[0]]
            tmp_img_name = os.path.split(tmp_human['path'])[-1]
            bbox = tmp_human['bbox']
#             img = cv2.imread(tmp_human['path'])
#             cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), (0,0,255), thickness = (img.shape[0]/400+1))
#             tmp_path = os.path.join(tmp_dir, tmp_img_name)
#             cv2.imwrite(tmp_path, img)

#             file_names_row.append(tmp_path)
            file_names_row.append(tmp_human['path'])
            file_names_row.extend([str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])])
            captions_row.append(tmp_img_name)

        file_names.append(file_names_row)
        captions.append(captions_row)

    # process path
    # if not writeBbox:
    #     SIZE = 200
    #     idxes = np.random.choice(range(len(file_names)), size=SIZE)
    #     file_names_sample = [file_names[i] for i in range(len(file_names)) if i in idxes ]
    #     captions_sample = [captions[i] for i in range(len(captions)) if i in idxes ]


    #     for i in xrange(len(file_names_sample)):
    #         for j in xrange(len(file_names_sample[i])):
    #             file_names_sample[i][j] = file_names_sample[i][j].replace('/home/SSD3/laoreja-data', '..')
    #             file_names_sample[i][j] = file_names_sample[i][j].replace('/home/laoreja/data', '..')

    # %cd ~/data/
    # from writeHTML import writeHTML
    # %cd ~/data/knn_res_new
    # if not writeBbox:
    #     writeHTML(html_filename, file_names_sample, captions_sample, height=200, width=200) 
    
   
        # write list
    contents = []
    for line in file_names:
        line = ' '.join(line)
        line += '\n'
        contents.append(line)
        
    with open(list_filename, 'w') as fd:
        fd.writelines(contents)
            
    # %cd ~/new-deep-landmark/
    return file_names, horse_angle_lt_zero, human_angle_lt_zero

def get_three_class(exist_groups,combine=False):
    TTT = []
    TFT = []
    TTF = []
    for pattern, li in exist_groups.items():
        if pattern.startswith('TTT'):
            TTT.extend(li)
        elif pattern.startswith('TFT'):
            TFT.extend(li)
        elif pattern.startswith('TTF'):
            TTF.extend(li)
    if combine:
        return TTT+TFT+TTF
    else:
        return TTT, TFT, TTF

def getFileNames(in_file,out_dir,file_pre_to_use):
    file_pre=file_pre_to_use+os.path.split(in_file)[1];
    file_pre=file_pre[:file_pre.rindex('.')];

    out_file_json=os.path.join(out_dir,file_pre+'_data.json');
    out_file_exist=os.path.join(out_dir,file_pre+'_existPattern.json');
    return out_file_json,out_file_exist;

def script_saveJSONs(in_file,out_dir,file_pre_to_use):
    out_file_json,out_file_exist=getFileNames(in_file,out_dir,file_pre_to_use);
    generate_json(in_file,out_file_json,out_file_exist);


def script_saveKNNs(horse_json_file,human_json_file,horse_exist_file,human_exist_file,k,out_file_neighbors):
    horse_json=json.load(open(horse_json_file,'r'));
    human_json=json.load(open(human_json_file,'r'));
    horse_groups=get_three_class(json.load(open(horse_exist_file,'r')),True);
    human_groups=get_three_class(json.load(open(human_exist_file,'r')),True);
    print len(horse_groups),len(human_groups);
    findKNN(horse_json,human_json,horse_groups,human_groups,k,out_file_neighbors);

    #horse_groups, human_groups, 5, '/home/SSD3/maheen-data/temp/all_points_val.html', '/home/SSD3/maheen-data/temp/knn_all_points_val_list.txt', False)

def main():
    # '/home/laoreja/new-deep-landmark/dataset/train/valImageList_2.txt'

    # out_dir='/home/SSD3/maheen-data/horse_project/neighbor_data';
    # k=100;
    # in_file='/home/laoreja/finetune-deep-landmark/dataset/train/aflw_cvpr_trainImageList.txt';
    # file_pre_to_use='human_';
    # human_json_file,human_exist_file=getFileNames(in_file,out_dir,file_pre_to_use);
    # file_pre_to_use='horse_';
    # in_file_dir='/home/laoreja/new-deep-landmark/dataset/train';
    # files=['valImageList_2.txt','testImageList_2.txt','trainImageList_2.txt'];



    out_dir='/home/SSD3/maheen-data/horse_project/neighbor_data';
    k=5;
    in_file='/home/laoreja/finetune-deep-landmark/dataset/train/aflw_cvpr_trainImageList.txt';
    file_pre_to_use='human_';

    human_json_file,human_exist_file=getFileNames(in_file,out_dir,file_pre_to_use);

    file_pre_to_use='sheep_';
    in_file_dir='/home/SSD3/maheen-data/horse_project/sheep_data'
    # '/home/laoreja/data/sheep';
    files=['trainImageList.txt','testImageList.txt'];
    # 'valImageList_2.txt','testImageList_2.txt','trainImageList_2.txt'];
    for file_curr in files:
        script_saveJSONs(os.path.join(in_file_dir,file_curr),out_dir,file_pre_to_use);
    

    in_files=[os.path.join(in_file_dir,file_curr) for file_curr in files];
    for in_file in in_files:    
        horse_json_file,horse_exist_file=getFileNames(in_file,out_dir,file_pre_to_use);
        out_file_neighbors=horse_json_file[:horse_json_file.rindex('.')]+'_'+str(k)+'_neigbors.txt';
        print out_file_neighbors;
        script_saveKNNs(horse_json_file,human_json_file,horse_exist_file,human_exist_file,k,out_file_neighbors);



if __name__=='__main__':
    main();