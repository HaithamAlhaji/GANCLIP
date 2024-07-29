git clone https://github.com/HaithamAlhaji/GANCLIP.git && cd GANCLIP/ && bash ./code/scripts/init_server.sh



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

