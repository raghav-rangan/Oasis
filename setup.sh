set -e

sudo mkdir /mydata/Oasis -p
sudo chown $USER /mydata/Oasis

cd /mydata/Oasis
git clone git@github.com:raghav-rangan/Oasis.git

mkdir /mydata/Oasis/miniconda
cd /mydata/Oasis/miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

source install/etc/profile.d/conda.sh
source ~/.bashrc
conda create --name oasis python=3.9
conda init
sleep 1
conda activate oasis

conda install -c nvidia cuda cudnn

cd /mydata/Oasis/Oasis
pip install -r requirements.txt

nvcc --version #check

sudo apt update
sudo apt install nvidia-driver-535
sudo reboot