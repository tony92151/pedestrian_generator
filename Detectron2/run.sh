python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '1.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_0p/' --num_of_data 42000 | tee part3_0p.log

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.75' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_25p/' --num_of_data 42000 | tee part3_25p.log

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.50' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_50p/' --num_of_data 42000 | tee part3_50p.log

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_100p/' --num_of_data 42000 | tee part3_100p.log