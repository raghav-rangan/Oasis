set -e

sudo mkdir /mydata/Oasis
sudo chown $USER /mydata/Oasis

cd mydata/Oasis
git clone git@github.com:raghav-rangan/Oasis.git

mkdir /mydata/Oasis/miniconda
cd /mydata/Oasis/miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

conda create --name oasis python=3.9
conda activate oasis

conda install -c nvidia cuda cudnn

cd /mydata/Oasis/Oasis
pip install -r requirements.txt

nvcc --version #check

sudo apt update
sudo apt install nvidia-driver-535
sudo reboot