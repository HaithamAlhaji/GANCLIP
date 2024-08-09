
mkdir downloaded_files
mkdir -p data/birds
gdown -O downloaded_files/preprocessed_birds.zip --id 1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj
unzip -oq downloaded_files/preprocessed_birds.zip -d data


wget -O downloaded_files/birds.tgz https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar xvzf downloaded_files/birds.tgz -C data/birds

