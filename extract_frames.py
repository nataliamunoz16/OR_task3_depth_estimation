import os
import sys
from PIL import Image

def extract(src, dst):
	print("Extracting frames...")
	current_dir = os.path.dirname(os.path.abspath(__file__))
	ffmpeg_exe = os.path.join(current_dir, 'ffmpeg.exe')
	cmd = f'{ffmpeg_exe} -hide_banner -loglevel panic -r 30 '
	status = os.system(cmd)
	if status != 0: 
		print("FFmpeg command failed! Check the path and file.")
        
	cmd += '-i ' + src +' '
	cmd += '-r 30 '
	cmd += dst + '/%4d.png'
	os.system(cmd)
	
def merge(rgb, segm, dst):
	print("Merging frames...")
	for fname in os.listdir(rgb):
		frgb = os.path.join(rgb, fname)
		fsegm = os.path.join(segm, fname)
		fdst = os.path.join(dst, fname)
		Image.merge("RGBA", (*Image.open(frgb).split(), Image.open(fsegm))).save(fdst)
	
def clean(folder):
	print("Removing tmp files...")
	for f in os.listdir(folder): os.remove(os.path.join(folder,f))
	os.rmdir(folder)
	
# Path to data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, 'cloth3d++_subset')

# Parse args
if len(sys.argv) > 1:
	samples = [sys.argv[1]]
else:
	samples = os.listdir(SRC)

for sample in samples:
	# Extract RGB data
	src = os.path.join(SRC, sample, sample + '.mkv')
	assert os.path.isfile(src), 'Specified sample does not exist'
	rgb = os.path.join(SRC, sample, 'rgb')
	if not os.path.isdir(rgb): os.mkdir(rgb)
	else:
		print("Already extracted, skipping...")
		continue
	extract(src, rgb)

	# Extract alpha channel
	src = src.replace('.mkv','_segm.mkv')
	assert os.path.isfile(src), 'Specified sample does not exist'
	segm = rgb.replace('rgb','segm')
	if not os.path.isdir(segm): os.mkdir(segm)
	else:
		print("Already extracted, skipping...")
		continue
	extract(src, segm)

	# Merge RGB and Alpha
	dst = rgb.replace('rgb', 'frames')
	if not os.path.isdir(dst): os.mkdir(dst)
	else:
		print("Already extracted, skipping...")
		continue
	merge(rgb, segm, dst)
	
	# Remove TMP folder ('rgb' and 'segm')
	clean(rgb)
	clean(segm)
	