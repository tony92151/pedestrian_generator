# python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '1.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_0p/' --num_of_data 42000 | tee part3_0p.log

# python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.75' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_25p/' --num_of_data 42000 | tee part3_25p.log

# python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.50' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_50p/' --num_of_data 42000 | tee part3_50p.log

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.15' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part3_15r/' --num_of_data 42000 --num_of_iter 50000 | tee part5_15r.log

# python eval_MR_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part3_0p/' --num_of_data 42000 | tee part4_eval_100p.log



# python testing_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --out_dir '/root/notebooks/final/detectron2_out_part3_0p/test/' --num_of_data 400 | tee part3_test.log

python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part3_15r/' --num_of_data 4000 --batch 10 --num_worker 5 | tee part5_15r_eval.log

# python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part3_25p/' --num_of_data 4000 --batch 10 --num_worker 5 | tee part4_eval_2_25p.log


python plot.py --csv_path '/root/notebooks/final/detectron2_out_part3_15r/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part3_15r/'

# python plot.py --csv_path '/root/notebooks/final/detectron2_out_part3_25p/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part3_25p/'