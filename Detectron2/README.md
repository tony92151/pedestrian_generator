# Usage




## Training uage

```bash=
python training_FPN.py --data_dir '<data dir>' --out_dir '<result dir>' --num_of_data <how many image use while traning> | tee <log name>.log
```

### Training baseline

[Download](https://drive.google.com/file/d/1aXBoZXSi4ASSkKo6uTJsqKCAZlx8j7Hy/view?usp=sharing) base line dataset and unzip

```bash=
# training single dataset
python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --out_dir '/root/notebooks/final/detectron2_baseline_out/' --num_of_data 42000 | tee baseline_out.log
```


### Training mixture dataset

Download two file and unzip (unzip `output_8.zip `in `caltech_origin_mask8_42000` folder) \
1. [caltech_origin_mask8_42000.zip](https://drive.google.com/file/d/1YrVsXYS3qYge5wEe42cGwksSBIAjzvsL/view?usp=sharing) \
2. [output_8.zip(gandata generade from caltech_origin_mask8_42000.zip)](https://drive.google.com/file/d/1ifYbp3PEnsCG3VYi2tVSiTyan355b0Zs/view?usp=sharing)

```bash=
# training with other dataset
python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '1.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_0p/' --num_of_data 42000 | tee part3_0p.log
# This training with mixture of 100% first-dataset and 0% second-dataset

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.75' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_25p/' --num_of_data 42000 | tee part3_25p.log
# This training with mixture of 75% first-dataset and 25% second-dataset
```

## MR-FPPI 


## Plot

```bash=
python plot.py --csv_path '/root/notebooks/Module_final/detectron2_output/detectron2_out_part3_100p/model_result.csv' --output_dir '/root/notebooks/Module_final/detectron2_output/detectron2_out_part3_100p/'
```