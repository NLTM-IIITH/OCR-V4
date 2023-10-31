import json
import os
import sys


def main(modality, lang):
	"""
	There is a bug in the V2 OCR where if we only give one word image as input,
	then the output of the ocr will only contain the first character of the word,
	instead of the complete word, while the OCR is working perfectly fine when
	the number of input word images > 1.
	The issue appears to be on ajoy side's in the testing code somewhere.
	for now, I added a dummy word image (hindi) called "-1.jpg" so that the OCR will
	always get >1 word images as input, and I remove the dummy image inference
	when creating the out.json file finally.
	"""
	if lang in ['hindi', 'marathi'] and modality in ['handwritten', 'scenetext']:
		os.system(f'rm -f /app/alphabet/{lang}_lexicon.txt')
		os.system(f'mv /app/alphabet/{lang}_{modality}_lexicon.txt /app/alphabet/{lang}_lexicon.txt')
	os.makedirs('/lmdb')
	os.makedirs('/out')
	# adding a dummy image, creating the lmdb, then removing the dummy image.
	os.system('cp /app/testing/-1.jpg /data')
	os.system('python create_lmdb.py')
	os.system('rm -rf /data/-1.jpg')
	command = [
		'python lang_train.py',
		'--mode test',
		'--lang {}'.format(lang),
		'--pretrained /model/out/crnn_results/best_cer.pth',
		'--valRoot /lmdb',
		'--out /out',
		'--cuda --adadelta'
	]
	os.system(' '.join(command))
	with open('/out/test_gt_and_predicted_text.txt', 'r') as f:
		b = f.readlines()
	a = []
	for i in b:
		x = i.strip().split('\t')[-1].strip()
		if x.endswith('.jpg'):
			x = ''
		a.append(x)
	# a = [i.strip().split('\t')[-1] for i in a]
	# first line in the inference is deleted because its the OCR output of dummy image.
	del a[0]
	b = os.listdir('/data')
	try:
		b = sorted(b, key=lambda x:int(x.strip().split('.')[0]))
	except:
		b.sort()
	ret = {}
	for i in range(len(b)):
		try:
			ret[b[i]] = a[i]
		except:
			ret[b[i]] = ''
	with open('/data/out.json', 'w', encoding='utf-8') as f:
		f.write(json.dumps(ret))

if __name__ == '__main__':
	main(sys.argv[-2] ,sys.argv[-1])

