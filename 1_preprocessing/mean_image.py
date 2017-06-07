"""
Calculate Mean Image or RGB Mean for given images
"""

import argparse
import sys
import os  
import cv2
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser(prog='compute_mean_image.py')
subparsers = parser.add_subparsers(dest="command")
bgr_parser = subparsers.add_parser("bgr", help="generate three channels mean [compute_mean_image.py' -bgr -h]")
img_parser = subparsers.add_parser("img", help="generate mean image [compute_mean_image.py' -img -h]")

bgr_parser.add_argument('directory', nargs = '*', help='Input Dataset Directory', type=str)
img_parser.add_argument('directory', nargs = '*', help='Input Dataset Directory', type=str)

bgr_parser.add_argument('--input-width', help='Width of input images', default = 224, type=int, metavar='')
bgr_parser.add_argument('--input-height', help='Height of input images', default = 224, type=int, metavar='')

img_parser.add_argument('--input-width', help='Width of input images', default = 227, type=int, metavar='')
img_parser.add_argument('--input-height', help='Height of input images', default = 227, type=int, metavar='')

if len(sys.argv[1:])==0:
	sys.argv.append("-h")
args = parser.parse_args()

def rgb_mean(imgdir, height, width):
	img_b = np.zeros((height, width))
	img_g = np.zeros((height, width))
	img_r = np.zeros((height, width))
	n_images = 0
	for root, dirs, filenames in os.walk(imgdir):
		for image in filenames:
			img = cv2.imread(imgdir+"/"+image)
			img_r += img[:,:,2]
			img_g += img[:,:,1]
			img_b += img[:,:,0]
			n_images += 1
			if n_images % 1000==0:
				print("{}: Done 1000 images".format(datetime.now()))
	return np.mean(img_b/n_images), np.mean(img_g/n_images), np.mean(img_r/n_images)

def img_mean(imgdir, height, width):
	mean_img = np.zeros((1, 3, height, width))
	n_images = 0
	for root, dirs, filenames in os.walk(imgdir):
		for image in filenames:
			img = cv2.imread(imgdir+"/"+image)
			mean_img[0][0] += img[:,:,0]
			mean_img[0][1] += img[:,:,1]
			mean_img[0][2] += img[:,:,2]
			n_images += 1
			if n_images % 1000==0:
				print("{}: Done 1000 images".format(datetime.now()))
	return mean_img/n_images

if __name__ == "__main__":
	for imgdir in args.directory:
		if not os.path.exists(imgdir):
			sys.stderr.write("Invalid directory: "+imgdir+"\n")
			exit(-1)
	if len(args.directory) == 0:
		sys.stderr.write("No directory found!\n")
		exit(-1)

	print("\n"+"-"*36)
	n_dirs = len(args.directory)
	if args.command == "bgr":
		print("{}: Start calculating bgr mean...".format(datetime.now()))
		img_b, img_g, img_r = 0, 0, 0
		for imgdir in args.directory:
			imgdir_b, imgdir_g, imgdir_r = rgb_mean(imgdir, args.input_width, args.input_height)
			img_b += imgdir_b
			img_g += imgdir_g
			img_r += imgdir_r
		outfile = open("bgr.txt", "w")
		outfile.write(str(img_b/n_dirs)+" "+str(img_g/n_dirs)+ " "+str(img_r/n_dirs))
		print("{}: Done calculating bgr mean, saved in bgr.txt".format(datetime.now()))
		print("Result:", img_b/n_dirs, img_g/n_dirs, img_r/n_dirs)
		outfile.close()

	elif args.command == "img":
		print("{}: Start calculating mean image...".format(datetime.now()))
		mean_img = np.zeros((1, 3, args.input_height, args.input_width))
		for imgdir in args.directory:
			mean_imgdir = img_mean(imgdir, args.input_width, args.input_height)
			mean_img += mean_imgdir
		mean_img /= n_dirs
		np.save("mean_img.npy", mean_img)
		print(mean_img)
		print("{}: Done calculating mean image, saved as mean_img.npy".format(datetime.now()))

	print("-"*36)