mkdir ~/pedestrian_generator_data

cd ~/pedestrian_generator_data

ID="1Rr0_StjqbP9LMnOe9LtQkE4ri4xf9mkU"
Path="./market_mask2.zip"

gdown --id "$ID" -O "$Path"

unzip "$Path"

cd market_mask2

ID="195C65Q91b1hrXisKcg6XQ_Lb_8sY5Os0"
Path="./market_mask_refine_6467.zip"

gdown --id "$ID" -O "$Path"

unzip "$Path"

echo "Dataset save at : $HOME/pedestrian_generator_data/market_mask2/"