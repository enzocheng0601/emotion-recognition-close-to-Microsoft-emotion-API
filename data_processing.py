import os
import subprocess
import sys
import json
import cv2
import numpy as np

def get_label():
	combined_image_path = sys.argv[2]
	json_file_output_path = sys.argv[3]
	emotion_api_path = sys.argv[4]

	if not os.path.exists(json_file_output_path):
		os.mkdir(json_file_output_path)

	print('getting label from MS emotion api...')
	for image_name in os.listdir(combined_image_path):
		number = image_name.split('.')[0]
		p = subprocess.Popen([emotion_api_path, combined_image_path+image_name, json_file_output_path + str(number)+'.json'])
		p.wait()
	sys.stdout.write("\033[K")


def labeling(filenames, label):
	print('labeling...')
	combined_image_path = sys.argv[2]
	json_data_path = sys.argv[3]
	train_image_path = os.getcwd() + '/train_image_dir/'
	if not os.path.exists(train_image_path):
		os.mkdir(train_image_path)
	counter = 1
	for file_name in os.listdir(json_data_path):
		number = file_name.split('.')[0]
		image = cv2.imread(combined_image_path + number + '.jpg')
		json_data = open(json_data_path + file_name).read()
		data = json.loads(json_data)
		faces = []
		faces_list = []
		for i in range(len(data['arr'])):
			faceRectangle = data['arr'][i]['faceRectangle']
			faceAttribute = data['arr'][i]['faceAttributes']
			(x, y, w, h) = (faceRectangle['left'], faceRectangle['top'], faceRectangle['width'], faceRectangle['height'])
			emotion = faceAttribute['emotion']
			label_array = []
			for key in emotion.keys():
				label_array.append(emotion[key])
			roi_cropped = image[y:y + h, x: x + w]
			cv2.imwrite(train_image_path + str(counter) + '.jpg', roi_cropped)
			label.append(label_array)
			filenames.append(train_image_path + str(counter) + '.jpg')
			counter += 1
	return filenames, label
	sys.stdout.write("\033[K")
			

