# Query By Strings and Return Ranking Word Regions with Only One Look
This is an implementation of paper "Query By Strings and Return Ranking Word Regions with Only One Look". The complete code will be provided soon. Please wait patiently!

## Requirements
* Python 3.5
* PyTorch v1.1.0
* shapely
* pillow
* opencv-python
* scipy
* tqdm
* scikit-image
* numpy
```
pip install -r requirements.txt
```

## Dataset preparation
1. Download the training, validation and testing dataset
    <p>The Konzilsprotokolle dataset can be downloaded from [ICFHR 2016 Handwritten Keyword Spotting Competition (H-KWS2016)](https://www.prhlt.upv.es/contests/icfhr2016-kws/data.html).</p>
    <p>The BH2M dataset can be downloaded from [IEHHR2017 competition](https://rrc.cvc.uab.es/?ch=10&com=downloads)</p>
2. Convert the downloaded dataset into the format we need
    ```
    python ./tools/tools_Konzilsprotokolle.py
    python ./tools/tools_BH2M.py
    ```
3. Augmenting training data offline
    ```
    python ./tools/tools_Konzilsprotokolle_docaug.py
    python ./tools/tools_BH2M_docaug.py
    ```

## Training
```
python train.py
```

## Testing
```
python predict.py
```
The cropped word images of the query and visual document images will be saved to `./output/~/QbS_word_res/` and `./output/~/QbS_res/` by default.

## Citation
If you find our method useful for your reserach, please cite:

## Feedback
Suggestions and discussions are greatly welcome. Please contact the authors by sending email to 18120456@bjtu.edu.cn
