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
    else
        echo "Directory does not exists."
        mkdir -p "$1"
    fi
}



echo -e "This script will creat a folder at \e[32m$HOME/pedestrian_generator_data/\e[0m and download traning data in it.\n"

read -p "Press [Enter] to continue... or [Control + c] to stop..."

echo ""

check_exis "$HOME/pedestrian_generator_data"

DATA_PATH=$HOME/pedestrian_generator_data

cd $DATA_PATH

ID="169RtS8oOuNV8ZgtwONdCU-Dx3uMHgvYm"
Path="./caltech_origin_mask10_100000.zip"

gdown --id "$ID" -O "$Path"

unzip "$Path"


echo " --------"
echo "|Done... |"
echo " --------"
