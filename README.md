### File Explanation
CNN-BiLSTM original : 논문 그대로를 재현한 버전


CNN-BiLSTM self attention : Encoder를 self-attention / Classifier를 CNN-BiLSTM하고, balancing contrastive loss를 수행 >> Supervised contrastive loss도 하나의 대안이다.


CNN-BiLSTM siamese network : siamese network를 이용한 버전. 아직 쓰지는 않지만 언제든지 가능성은 열려있다. (https://github.com/diheal/resampling/blob/main/SimCLRHAR.py)이 siamese network이다.


CNN-BiLSTM supervised contrastive loss
(https://keras.io/examples/vision/supervised-contrastive-learning/#supervised-contrastive-learning)

CNN-BiLSTM balancing contrastive loss
(https://www.scitepress.org/Papers/2020/101353/pdf/index.html)


Self-Attention Balancing : AttentionContext() 대신 tf.keras.layers.GlobalMaxPool1D() 사용 결과 성능 대폭 
