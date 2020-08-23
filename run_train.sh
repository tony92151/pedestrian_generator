# echo "Training task    : these tasks might take 22 hours."
# echo "Evaluating task  : these tasks might take 6 hours."
# echo "Plot task        : these tasks might take 1 minute."
# echo "TOTAL 28 HOURS"

echo '
                    __          __       _                                                 __            
    ____  ___  ____/ /__  _____/ /______(_)___ _____     ____ ____  ____  ___  _________ _/ /_____  _____
   / __ \/ _ \/ __  / _ \/ ___/ __/ ___/ / __ `/ __ \   / __ `/ _ \/ __ \/ _ \/ ___/ __ `/ __/ __ \/ ___/
  / /_/ /  __/ /_/ /  __(__  ) /_/ /  / / /_/ / / / /  / /_/ /  __/ / / /  __/ /  / /_/ / /_/ /_/ / /    
 / .___/\___/\__,_/\___/____/\__/_/  /_/\__,_/_/ /_/   \__, /\___/_/ /_/\___/_/   \__,_/\__/\____/_/     
/_/                                                   /____/                                             
'


check_exis(){
    if [ -d "$1" ]; then
        echo "Directory exists."
	exit 1
    else
        echo "Directory does not exists."
        mkdir -p "$1"
    fi
}



echo -e "This script will creat a folder at \e[32m$HOME/pedestrian_generator_data/\e[0m and download related data in it.\n"
echo -e "Data preprocess will be ignore. This script will download dataset we provide. (More detail in \e[32m01_gene_train_dataset\e[0m) \n"

echo -e "STEP \e[32m Download Image Data \e[0m >> \e[32m Training GAN model \e[0m >> \e[32m Generate GAN dataset \e[0m "
echo    "       ^^^^"

read -p "Press [Enter] to continue... or [Control + c] to stop..."

echo ""


check_exis "$HOME/pedestrian_generator_data"

echo ""

REPO_PATH=$PWD
DATA_PATH=$HOME/pedestrian_generator_data

echo -e "REPO_PATH : \e[33m$REPO_PATH \e[0m"
echo -e "DATA_PATH : \e[33m$DATA_PATH \e[0m"

cd $DATA_PATH

ID="1Rr0_StjqbP9LMnOe9LtQkE4ri4xf9mkU"
Path="./market_mask2.zip"

gdown --id "$ID" -O "$Path"

unzip "$Path" 

cd $DATA_PATH/market_mask2

ID="195C65Q91b1hrXisKcg6XQ_Lb_8sY5Os0"
Path="./market_mask_refine_6467.zip"

gdown --id "$ID" -O "$Path"

unzip "$Path"

cd $DATA_PATH

ID="195C65Q91b1hrXisKcg6XQ_Lb_8sY5Os0"
Path="./market_mask_refine_6467.zip"

echo " --------"
echo "|Done... |"
echo " --------"





