#!/bin/bash

model=resnet20
echo "python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model
