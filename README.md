In .zshrc file, need this to initalize base conda environment


 15 __conda_setup="$('/opt/homebrew/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
 16 if [ $? -eq 0 ]; then
 17     eval "$__conda_setup"
 18 else
 19     if [ -f "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" ]; then
 20         . "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"
 21     else
 22         export PATH="/opt/homebrew/anaconda3/bin:$PATH"
 23     fi
 24 fi

 Then run

 exec zsh

 to launch (base) conda environment

 then run 
 
 conda env create -f environment.yml

The environment yaml often does not add tensorflow, so once you activate the environment using

 conda activate denoising-autoencoder

run 

pip install tensorflow

for Q1, run 

python numsvisualization.py

for Q2 run

python basic_denoise_MNIST.py

for Q3 run

python conv_denoise_MNIST.py

for Q4 run

python autoencoder_denoise_documents.py

NOTE Q4 runs pretty slowly for extreme precision ~10 min runtime. If you want faster execution time ~1:45 runtime, reduce the epochs to 1, increase the batch size to 8, and comment out the lines with the #COMMENT OUT FOR SPEED TAG above them. It still runs shockingly well. Please note that the test images are extracted from the training set so as to be completely fresh data, but you can move around any of the other testing files as needed. 

In th project directories, we included scripts for padding the test images to make them all the same size for input, and a script to check the dimensions of the resulting directory.