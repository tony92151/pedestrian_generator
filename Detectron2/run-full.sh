echo "Training task    : these tasks might take 40 hours."
echo "Evaluating task  : these tasks might take 6 hours."
echo "Plot task        : these tasks might take 1 minute."
echo "TOTAL 46 HOURS"

read -p "Press [Enter] to continue... or [Control + c] to stop..."

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '1.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part4_0p/' --num_of_data 42000 --num_of_iter 100000 | tee part4_0p.log

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.75' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part4_25p/' --num_of_data 42000 --num_of_iter 100000 | tee part4_25p.log

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.50' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part4_50p/' --num_of_data 42000 --num_of_iter 100000 | tee part4_50p.log

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part4_100p/' --num_of_data 42000 --num_of_iter 100000 | tee part4_100p.log


python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part4_0p/' --num_of_data 4000 --batch 10 --num_worker 5 | tee part4_eval_2_0p.log

python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part4_25p/' --num_of_data 4000 --batch 10 --num_worker 5 | tee part4_eval_2_25p.log

python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part4_50p/' --num_of_data 4000 --batch 10 --num_worker 5 | tee part4_eval_2_50p.log

python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part4_100p/' --num_of_data 4000 --batch 10 --num_worker 5 | tee part4_eval_2_100p.log


python plot.py --csv_path '/root/notebooks/final/detectron2_output/detectron2_out_part3_0p/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part3_0p/'

python plot.py --csv_path '/root/notebooks/final/detectron2_output/detectron2_out_part3_25p/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part3_25p/'

python plot.py --csv_path '/root/notebooks/final/detectron2_output/detectron2_out_part3_50p/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part3_50p/'

python plot.py --csv_path '/root/notebooks/final/detectron2_output/detectron2_out_part3_100p/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part3_100p/'