python predict_tda.py --input_dir GCDC_Dataset --feat_dir gcdc_tda_features_xlnet/ --domain clinton --max_tokens 512 --model xlnet-base-cased
python predict_tda.py --input_dir GCDC_Dataset --feat_dir gcdc_tda_features_xlnet/ --domain enron --max_tokens 512 --model xlnet-base-cased
python predict_tda.py --input_dir GCDC_Dataset --feat_dir gcdc_tda_features_xlnet/ --domain yahoo --max_tokens 512 --model xlnet-base-cased
python predict_tda.py --input_dir GCDC_Dataset --feat_dir gcdc_tda_features_xlnet/ --domain yelp --max_tokens 512 --model xlnet-base-cased