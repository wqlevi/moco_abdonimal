ROOT_DIR=$HOME/git_wq/SSL_veronika
DATA_DIR=$ROOT_DIR/datasets
mkdir $DATA_DIR
wget -O $DATA_DIR/example_data_tsne.pkl.gz http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz
