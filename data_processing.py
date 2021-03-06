import os
import subprocess
import sys
import json
import cv2
import numpy as np
import time

def get_label():
	combined_image_path = sys.argv[2]
	json_file_output_path = sys.argv[3]
	emotion_api_path = sys.argv[4]

	if not os.path.exists(json_file_output_path):
		os.mkdir(json_file_output_path)

	print(' getting label from MS emotion api...')
	for image_name in os.listdir(combined_image_path):
		number = image_name.split('.')[0]
		p = subprocess.Popen([emotion_api_path, combined_image_path+image_name, json_file_output_path + str(number)+'.json'])
		p.wait()
	print(' finished')
	time.sleep(1)

def label_check(N_of_tra_img, N_of_tra_lab, N_of_val_img, N_of_val_lab):
	if N_of_tra_img != N_of_tra_lab or N_of_val_img != N_of_val_lab:
		print([N_of_tra_img, N_of_tra_lab, N_of_val_img, N_of_val_lab])
		print('pls check the data')
		return False
	else:
		return True



def labeling(filenames, label, val_filenames, val_label):
	print(' labeling training data...')
	labels = []
	val_labels = []
	combined_image_path = sys.argv[2]
	json_data_path = sys.argv[3]
	
	train_image_path = os.getcwd() + '/train_image_dir/'
	if not os.path.exists(train_image_path):
		os.mkdir(train_image_path)
	counter = 0
	for file_name in os.listdir(json_data_path):
		number = file_name.split('.')[0]
		image = cv2.imread(combined_image_path + number + '.jpg')
		json_data = open(json_data_path + file_name).read()
		data = json.loads(json_data)
		for i in range(len(data['arr'])):
			faceRectangle = data['arr'][i]['faceRectangle']
			faceAttribute = data['arr'][i]['faceAttributes']
			(x, y, w, h) = (faceRectangle['left'], faceRectangle['top'], faceRectangle['width'], faceRectangle['height'])
			emotion = faceAttribute['emotion']
			label_array = []
			for key in sorted(emotion.keys()):
				label_array.append(emotion[key])
			roi_cropped = image[y:y + h, x: x + w]
			cv2.imwrite(train_image_path + str(counter) + '.jpg', roi_cropped)
			label.append(label_array)
			counter += 1
	for i in range(counter):
		if i < int(0.8*counter):
			filenames.append(train_image_path + str(i) + '.jpg')
		else:
			val_filenames.append(train_image_path + str(i) + '.jpg')
	for index in range(len(label)):
		if index < len(filenames):
			labels.append(label[index])
		else:
			val_labels.append(label[index])
	check = label_check(len(filenames), len(labels), len(val_filenames), len(val_labels))




	if check:
		print(' finished')
		time.sleep(1)
		os.system('cls')
	return filenames, labels, val_filenames, val_labels
			

