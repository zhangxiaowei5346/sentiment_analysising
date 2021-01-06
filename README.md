# sentiment_analysising
You are recommended to use conda create env by   
  ```python
  conda env create -f environment.yml
  ```  
Then activate the environment  by   
  ```Python 
  conda activate environment
  ```

  After create and activate the environment, you can quickly run the project（default args are RNN model）:  
  ```Python   
  python sentiment_analysis.py 
  ```  
  
**Parameters:**  
  ```--data ```:specify the folder name of the dataset.  
  ```--epoch_num ```:number of epochs.  
  ```--embedding_size ```:Size of embeddings.  
  ```--model_choose ```:model choose: RNN/LSTM/GRU.   
  ```--n_layers ```:specify the model layer.  
  ```--bidirectional ```:if use bidirectional RNN model.     
  ```--drop_out ```:drop_out(float).  
  ```--hidden_dim```:size of hidden dim.   
  ```--output_dim```:size of output dim.  
  ```--include_length```:LSTM is True others False/default.  
  ```--sort_within_batch```:LSTM is True others False/default.  
  ```--batch_size```:training/validation batch size.  
  ```--pretrained_emb```:LSTM is glove.6B.100d.  
  
  
**Run Project with Parameters:**  
&nbsp;&nbsp;&nbsp;&nbsp;**LSTM:**  
  ```Python
  python sentiment_analysis.py --model_choose LSTM --n_layers 2 --bidirectional True --drop_out 0.5 --include_length True --sort_within_batch True --pretrained_emb glove.6B.100d 
  ```  
&nbsp;&nbsp;&nbsp;&nbsp;**GRU:**  
  ```Python
  python sentiment_analysis.py --model_choose GRU --n_layers 2 --bidirectional True --drop_out 0.25 
  ```   
  
**Reference**  
<https://github.com/bentrevett/pytorch-sentiment-analysis> 
  

  
 
  
