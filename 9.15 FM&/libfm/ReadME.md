首先，利用train_test_split_for_movielen.ipynb将ratings.csv数据集划分成train.csv和test.csv
接着，使用triple_format_to_libfm.pl脚本，将train.csv和test.csv转换成libsvm格式，即train.csv.libfm和test.csv.libfm
最后，利用libfm.exe做训练和预测，得到test对应的预测评分out.txt。(1.png和2.png为训练过程)