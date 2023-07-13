## 这是什么？
这是我的毕业论文《聚焦句法相似度的EFL写作句法复杂度研究》所有的相关代码，主要功能如下：
+ 使用`transformer`库提供的`deberta-v3-large`模型，提取每篇作文1024维的embeddings。
+ 使用`spaCy`提供的语言模型，进行dependency parsing和part of speech tagging，据此计算每篇作文的句法复杂度指标。
+ 实现了卷积树核函数(tree kernel)，得出作文中任意两句话间的句法相似度。
+ 根据以上特征，使用xgb回归模型预测写作分数。
+ 使用rule-based dependency matching提取动词结构，对动词结构进行隐语义分析(latent semantic analysis, LSA)和聚类分析。  
以下是论文的摘要：  
Syntactic complexity is an important topic in EFL writing research because it evaluates the level of syntactic sophistication and similarity in language production. Recent studies have developed fine-grained complexity measures. Compared to the large-grained measures previously used by most studies, they are more comprehensive and address some limitations in previous studies. However, syntactic similarity as a crucial component of complexity has not received much attention and is often excluded from complexity measures. Also, previous research has mainly focused on building regression models to predict writing scores from complexity measures, treating essays as isolated samples. The possible presence of similar syntactic patterns across different essays is rarely investigated.    
This study focuses on syntactic similarity to analyze syntactic complexity in 3830 essays written by EFL students at a Chinese university. We compute five sentence similarity measures by using a tree kernel that measures structural similarity between dependency trees. These similarity measures are combined with other fine-grained measures to predict writing scores. We also apply clustering to group essays that share similar syntactic patterns, so as to identify similar patterns present across multiple essays, rather than just in individual ones.   
Our results show that syntactic similarity measures significantly predict writing scores, explaining about 23% of the variance in scores. Higher-scoring students tend to produce sentence structures that are less similar to each other. In terms of predicting writing scores, fine-grained measures that include similarity measures perform better than large-grained measures. Furthermore, clustering reveals several similar patterns that emerge in different groups of essays. Some of these patterns are related to the writing prompts, which implies that prompts influence the structures that students choose to use. These findings highlight the importance of including syntactic similarity measures in future studies, and provide a new corpus-based perspective on syntactic similarity. They may also provide references for EFL classroom instruction.

## 文件说明
+ `functions.py`: 所有主要功能实现和OOP封装
+ `playground.ipynb`对关键功能实现的详细解释和测试用例
+ `reg.ipynb`, `class.ipynb`：分别是回归和分类的对作文进行自动评分的例子，使用[ELL](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)和[ASAP #2](https://www.kaggle.com/c/asap-aes)数据集。
+ `final-*.ipynb`：所有毕业论文相关的数据分析代码，其中，`final-feature.ipynb`提取特征，包括句法复杂度、相似度指标、动词结构和BERT embeddings，`final-regression.ipynb`对作文进行自动评分，`final-sent.ipynb`计算并分析句法相似度指标，`final-sim.ipynb`基于动词结构进行LSA和聚类分析。
## 文件路径配置说明
数据放在`data`文件夹下：
```
data
|
|---asap-aes
|   |---training_set_rel3.xls
|---Final-all
|   |---winter.csv
|---L2writing
|   |---train.csv
```
模型放在`model`文件下：
```
model
|
|---deberta-v3-large
|---tokenizer
|   |---deverta-v3-large
```
以上可在`function.py`中`DataLoader`类和`GetBERTEmbeddings`类的实现中自行配置    
特征提取的结果放在`feature`文件夹下  
## 环境
代码用到的库有：  
Python 3.10.8, NumPy 1.23.3, pandas 1.5.0, NetworkX 3.0, scikit-learn 1.1.2, statsmodels 0.13.5, xgboost 1.6.2, pingouin 0.5.3, Markov-clustering 0.0.6, transformers 4.21.3, 3.5.2, SuPar 1.1.4, NLTK 3.5. 