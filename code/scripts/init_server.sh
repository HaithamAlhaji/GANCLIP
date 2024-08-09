# git clone https://github.com/HaithamAlhaji/GANCLIP.git && cd GANCLIP/ && bash ./code/scripts/init_server.sh

git config --global user.name "haitham_seerver"
git config user.email "haithamtalhaji@yahoo.com"

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install setuptools==59.5.0
pip install numpy
pip install gdown
sudo apt-get install unzip


