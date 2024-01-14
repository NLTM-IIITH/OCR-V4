# Printed Word Recognition for Indic Languages

## Dataset preparation:
- Please follow README under "create_lmdb_dataset" folder

## Pretrained Models:
- You can find the pretrained models for V4 printed for 13 languages under the [Assets](https://github.com/NLTM-OCR/OCR-V4/releases/tag/v4).

## Setup
- Using Python = 3.10+
- Install Dependencies `pip install -r requirements.txt`

## Evaluation and testing:

```bash
python3 lang_train.py --mode test --lang bengali --valRoot  bengali/test_lmdb --pretrained  out/crnn_results/best_cer.pth --cuda  --out  out --adadelta
```
`--pretrained` is trained model path containing trained model

- Please see config.py file, if you want to change these parameters 
- Path "out/test_accuracy.txt" contains  test accuracy with respect to CRR and WRR
- Path "out/test_gt_and_predicted_text.txt" contains predicted output (text) of each word image. 1st column is the ground truth text and 2nd column is the predicted text
- Path "out/max_prob" folder containing ".csv" file corresponding to each test word image. It provides probability values of predicted characters.
- Path "out/data_prob" folder containing ".csv" file corresponding to each test word image. It provides a matrix of probability values of a test word image. Matrix dimention is (no of characters) x (length of word). If (iii) and (iv) are not required, you can skip them.    

## Inference (New)

For Inference please call the `new_infer.py` file. The OCR outputs are generated in TXT file and saved in the directory specified by `out_dir` argument.

### Arguments
* `--pretrained`: Path to pretrained model file (.pth)
* `--test_root`: Path to directory with input images
* `--out_dir`: Path to folder where TXT OCR output is saved.
* `--language`: language of the input images

### Example

```bash
python new_infer.py --pretrained=/home/ocr/model/best_cer.pth --test_root=/home/ocr/data --language=bengali --out_dir=/home/ocr/out
```

## Contact

You can contact **[Ajoy Mondal](mailto:ajoy.mondal@iiit.ac.in)** for any issues or feedbacks.

## Citation

```
@InProceedings{iiit_hw,
	author="Gongidi, Santhoshini and Jawahar, C. V.",
	editor="Llad{\'o}s, Josep and Lopresti, Daniel and Uchida, Seiichi",
	title="iiit-indic-hw-words: A Dataset for Indic Handwritten Text Recognition",
	booktitle="Document Analysis and Recognition -- ICDAR 2021",
	year="2021",
	pages="444--459"
}
```
