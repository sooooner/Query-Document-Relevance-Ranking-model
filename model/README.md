# model
Keras modeling에 필요한 커스텀 레이어 모듈

## layers.py
모델에 필요한 custom layers

## loss.py
Pairwise ranking loss  
$$ L(q, d^+, d^-; \theta) = max(0, 1 - s(q, d^+) + s(q, d^-)) $$


## models
각 .py 파일에서 Gen_{model_name}_Model 함수로 모델 생성
+ drmm.py
+ pacrr.py
+ pacrr_drmm.py


## callback.py
Custom callback 함수  
LossHistory, _metric : loss와 matric 기록

## Example
**drmm model training**
```
import tensorflow as tf
from model.loss import Pairwise_ranking_loss
from model.drmm import Gen_DRMM_Model
from model.callback import LossHistory, _metric
from utility.utility import history_plot

drmm = Gen_DRMM_Model()
drmm.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=.1), loss=Pairwise_ranking_loss)

history_drmm = LossHistory(metadata_dev)
drmm_metric = _metric(test)

total_epoch_count = 100
batch_size = 128
drmm.fit(x=metadata, y=tf.constant([0.]*len(train)),
         validation_data=(metadata_dev, tf.constant([0.]*len(dev))),
         shuffle=True,
         epochs=total_epoch_count,
         batch_size=batch_size,
         callbacks=[history_drmm, drmm_metric])
         
history_plot(history_drmm, drmm_metric, batch_size, df=train)
```



