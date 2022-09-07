## TopicVAE
This is the Tensorflow  implementation for our ACM MM 2022 paper:
>Zhiqiang Guo, Guohui Li, Jianjun Li, Huaicong Chen. TopicVAE: Topic-aware Disentanglement Representation Learning for Enhanced Recommendation. In MM 2022. [Paper](#)

### Environment
pip install -r requirements.txt

### Data processing

### Run
* TopicDAE.
    You can test TopicDAE on Industrial dataset by 'python TopicDAE.py --data=data/fasttext/Industrial --mode=tst', or retrain TopicDAE by 'python TopicVAE.py --data=data/fasttext/Industrial --mode=trn'.
* TopicVAE.
    You can test TopicVAE on Industrial dataset by 'python TopicVAE.py --data=data/fasttext/Industrial --mode=tst', or retrain TopicVAE by 'python TopicVAE.py --data=data/fasttext/Industrial --mode=trn'.
