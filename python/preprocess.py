import os;
import util;
def main():
	dir_vids='/disk2/aprilExperiments/horses/mediaFromPPT';
	dir_frames='/disk2/aprilExperiments/horses/mediaFromPPT_frames';
	out_file_commands='/disk2/aprilExperiments/horses/extract_frames.txt';
	util.mkdir(dir_frames);

	command_template='ffmpeg -i VIDEONAME -vf fps=1 OUTPRE%05d.jpg';
	vids=[os.path.join(dir_vids,file_curr) for file_curr in os.listdir(dir_vids) if file_curr.endswith('.mp4')];
	out_pres=[os.path.join(dir_frames,file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')]+'_') for file_curr in vids];
	commands=[];
	for vid,out_pre in zip(vids,out_pres):
		command_curr=command_template.replace('VIDEONAME',vid);
		command_curr=command_curr.replace('OUTPRE',out_pre);
		commands.append(command_curr);

	util.writeFile(out_file_commands,commands);



if __name__=='__main__':
	main();