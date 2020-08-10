# Usage

## Training

```shell=
# training single dataset
python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --out_dir '/root/notebooks/final/detectron2_out_part2/' --num_of_data 42000 | tee part2.log

# training with other dataset
python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '1.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_0p/' --num_of_data 42000 | tee part3_0p.log
# This training with mixture of 100% first-dataset and 0% second-dataset

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.75' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_25p/' --num_of_data 42000 | tee part3_25p.log
# This training with mixture of 75% first-dataset and 25% second-dataset
```