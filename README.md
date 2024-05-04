# Printed Word Recognition for Indic Languages

## Pretrained Models:
- You can find the pretrained models for V4 printed for 13 languages under the [Assets](https://github.com/NLTM-OCR/OCR-V4/releases/tag/v4).

## Setup
- Using Python = 3.10+
- Install Dependencies `pip install -r requirements.txt`

## Inference (New)

For Inference please call the `new_infer.py` file. The OCR outputs are generated in JSON file and saved in the directory specified by `out_dir` argument.

### Arguments
* `--pretrained`: Path to pretrained model file (.pth)
* `--test_root`: Path to directory with input images
* `--out_dir`: Path to folder where JSON OCR output is saved.
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
