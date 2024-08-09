
mkdir downloaded_files
mkdir -p data/coco
gdown -O downloaded_files/preprocessed_coco.zip --id 15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB
unzip -oq downloaded_files/preprocessed_coco.zip -d data


wget -O downloaded_files/train2014.zip http://images.cocodataset.org/zips/train2014.zip
wget -O downloaded_files/val2014.zip http://images.cocodataset.org/zips/val2014.zip
wget -O downloaded_files/test2014.zip http://images.cocodataset.org/zips/test2014.zip
wget -O downloaded_files/annotations_trainval2014.zip http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget -O downloaded_files/image_info_test2014.zip http://images.cocodataset.org/annotations/image_info_test2014.zip
unzip -oq downloaded_files/train2014.zip -C data/coco
unzip -oq downloaded_files/val2014.zip -C data/coco
unzip -oq downloaded_files/test2014.zip -C data/coco
unzip -oq downloaded_files/annotations_trainval2014.zip -C data/coco
unzip -oq downloaded_files/image_info_test2014.zip -C data/coco

