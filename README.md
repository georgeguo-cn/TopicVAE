## TopicVAE
This is the Pytorch implementation for our ACM MM 2022 paper:
'''
Zhiqiang Guo, Guohui Li, Jianjun Li, Huaicong Chen. TopicVAE: Topic-aware Disentanglement Representation Learning for Enhanced Recommendation. MM 2022. 
'''

# checkpoints
This folder contains the trained model of TopicVAE and TopicDAE on Industrial dataset.
# data
This folder contains several files of Industrial dataset for training.
# Run
1. TopicDAE.
    You can test TopicDAE on Industrial dataset by 'python TopicDAE.py --data=data/fasttext/Industrial --mode=tst', or retrain TopicDAE by 'python TopicVAE.py --data=data/fasttext/Industrial --mode=trn'.
2. TopicVAE.
    You can test TopicVAE on Industrial dataset by 'python TopicVAE.py --data=data/fasttext/Industrial --mode=tst', or retrain TopicVAE by 'python TopicVAE.py --data=data/fasttext/Industrial --mode=trn'.
