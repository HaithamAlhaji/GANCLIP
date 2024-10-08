# git clone https://github.com/HaithamAlhaji/GANCLIP.git && cd GANCLIP/ && bash ./code/scripts/init.sh
# cd GANCLIP/
# bash ./code/scripts/init.sh

# git fetch --all
# git fetch --all && git reset --hard origin/main

# conda create --name myenv python=3.9 -y
# conda activate myenv

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install setuptools==59.5.0
pip install numpy
pip install gdown
sudo apt-get install unzip

mkdir downloaded_files
mkdir -p data/birds
gdown -O downloaded_files/preprocessed_birds.zip --id 1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj
unzip -oq downloaded_files/preprocessed_birds.zip -d data

wget -O downloaded_files/birds.tgz https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar xvzf downloaded_files/birds.tgz -C data/birds

python code/src/train.py --cfg code/cfg/birds.yml

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=11266 code/src/train.py --cfg code/cfg/birds.yml

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=11266 code/src/train.py --cfg code/cfg/birds.yml

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11266 code/src/train_multi.py --cfg code/cfg/birds_multi.yml
