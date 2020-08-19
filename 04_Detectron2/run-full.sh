echo "Training task    : these tasks might take 22 hours."
echo "Evaluating task  : these tasks might take 6 hours."
echo "Plot task        : these tasks might take 1 minute."
echo "TOTAL 28 HOURS"

read -p "Press [Enter] to continue... or [Control + c] to stop..."

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '1.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part5_100r0g/' --num_of_data 20000 --num_of_iter 100000 | tee part5_100r0g.log

#python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.75' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part4_75r25g/' --num_of_data 42000 --num_of_iter 40000 | tee part4_75r25g.log

python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.50' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part5_50r50g/' --num_of_data 40000 --num_of_iter 100000 | tee part5_50r50g.log

#python training_FPN.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --data_p '0.0' --second_data_dir '/root/notebooks/final/caltech_origin_mask8_42000/' --out_dir '/root/notebooks/final/detectron2_out_part4_0r100g/' --num_of_data 42000 --num_of_iter 100000 | tee part4_0r100g.log


python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part5_100r0g/' --num_of_data 1000 --batch 10 --num_worker 5 | tee part5_eval_100r0g.log

#python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part4_75r25g/' --num_of_data 1000 --batch 10 --num_worker 5 | tee part4_eval_75r25g.log

python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part5_50r50g/' --num_of_data 1000 --batch 10 --num_worker 5 | tee part5_eval_50r50g.log

#python eval_MR_FPN2.py --data_dir '/root/notebooks/final/caltech_origin_data_refine/' --model_dir '/root/notebooks/final/detectron2_out_part4_0r100g/' --num_of_data 1000 --batch 10 --num_worker 5 | tee part4_eval_0r100g.log


#python plot.py --csv_path '/root/notebooks/final/detectron2_out_part4_100r0g/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part4_100r0g/plot_100r0g.jpg'

#python plot.py --csv_path '/root/notebooks/final/detectron2_out_part4_75r25g/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part4_75r25g/plot_75r25g.jpg'

#python plot.py --csv_path '/root/notebooks/final/detectron2_out_part4_50r50g/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part4_50r50g/plot_50r50g.jpg'

#python plot.py --csv_path '/root/notebooks/final/detectron2_out_part4_0r100g/model_result.csv' --output_dir '/root/notebooks/final/detectron2_out_part4_0r100g/plot_0r100g.jpg'