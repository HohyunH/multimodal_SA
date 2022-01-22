# multimodal_SA

![image](https://user-images.githubusercontent.com/46701548/133446286-227d8c67-e509-433c-9550-027853d87fe1.png)


main.py : training baseline models

c_main.py : Simply by adding category information and learning.

m_main.py : Multimodal learning progress by learning metadata by MLP.

![멀티모달 그림](https://user-images.githubusercontent.com/46701548/133445070-869e9492-ad2f-4634-a5cb-15a02f439943.png)

### How to use

<pre>
<code>
python main.py --embed 200 --windows 3 --maxlen 64 --epochs 10 --batch 64 --model [gru, lstm, ssan, bert]
</code>
</pre>

![image](https://user-images.githubusercontent.com/46701548/133445570-87b9c6a0-0d99-46ea-94a6-a88cd9aa690b.png)

### Final code => notebook

- wusinsa_nn_model.ipynb : training with GRU and LSTM
- wusinsa_attention_model.ipynb : training with multihead attention classifier
- wusinsa_pytorch_kobert.ipynb : training with Ko-BERT
