
 WSM Project 1: Ranking by Vector Space Models


####################功能描述###################
使用不同的 weighting 以及 similarity 計算方法，
將文章排序，再利用relevence feedback的方法改進
排序結果。


####################測試執行###################
切換至本專案位置

測試第一題&第二題:
python main.py --query <query>


測試第三題(Bonus):
python main2.py --query <query>

##################開發環境與套件###############
python==3.8.5
numpy==1.19.2
nltk==3.5
textblob==0.15.3
jieba==0.42.1

####################專案結構###################
|-- README.txt
|-- main.py              #主程式  (第一題&第二題)
|-- Parser.py            #文章字詞前處理(英文)
|-- PorterStemmer.py     #字根還原
|-- util.py              #向量運算
|-- main2.py		 #主程式2 (第三題)
|-- Parser2.py		 #文章字詞前處理(中文)
|-- chinesedict.txt      #中文斷詞詞庫
|-- EnglishStopwords.txt
|-- ChineseStopwords.txt
|
|---/EnglishNews 
|    |--News100012
|    |--News100020
|    |--News100021
|	...
|
|---/ChineseNews
|    |--News200010
|    |--News200017
|    |--News200019
|       ...



######################作者#####################

資科三 林晉毅 107207426