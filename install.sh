conda create -p ./envs
conda activate ./envs

conda install python=3.10 pytorch=2.0 pytorch-cuda=11.8 torchaudio -c pytorch -c nvidia
pip install -r requirements.txt
