
# Preprocess
Import the experimental virtual environment in conda
```
conda env -f DG_WSDH.yaml
```
then enter the environment:
```
conda activate DG_WSDH
```
### Process raw pathology images
Sample dataset download: [Link 1](https://drive.google.com/file/d/13Fb2U59KiXnhqLfjSwgZ3Vpr2uBf7c2I/view?usp=sharing) [Link 2](https://pan.baidu.com/s/1C-xuMsTrVKLYyEGONUocwg?pwd=pz75) 

After downloading, extract `example.zip` to the `MRAN/WSI/` directory, then preprocess the original image:
```
cd pre
python run_preprocess.py
```
The storage format of the original data set can refer to the sample data set in `MRAN/WSI/example ` : 
> DG_WSDH/WSI/dataset_name/\*/slide-1.svs
> 
> ...
> 
> DG_WSDH/WSI/dataset_name/\*/slide-n.svs

### Divide the dataset
```
cd pre
python pro_csv.py
```
The format of the label file of the original dataset can refer to `DG_WSDH/csv/example/sheet/total.csv`:
| File Name | Sample Type |
|--|--|
| slide-1.svs | Primary Tumor |
| slide-2.svs |Solid Tissue Normal  |
|....|...|

# Train and test

### train

```
python main.py
```

