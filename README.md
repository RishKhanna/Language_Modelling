# INLP Assignment-1 Submission

## System Requirements
- `Python 3.9` is reqiured to run the code, other versions of `python` do not work due to changes in the `pickle` module over time.
- Other requirements:
	``` 
	keras==2.11.0 
	numpy==1.24.2
	regex==2022.10.31
	tqdm==4.64.1 
	```

## How to run
### Language Modelling
`$ python language_model.py <smoothing type> <path to corpus>`
Where \<smoothing type> is :
	- `k` for Kneser-Ney
	- `w` for Witten-Bell
### Neural Language Modelling
`$ python neural_language_model.py <path to trained model>`
Language models on both corpus have been trained and uploaded to drive due to memory constraints. The models are already trained and stored as pickle files.

When you run the model it asks for an input sentence and returns it's  perplexity scores, indicating how similar the sentence is to the corpus corresponding to the model.

## Submission Format
```
- README.md
- Report.pdf
- language_model.py
- neural_language_model.py
- language model.ipynb
- neural_language_model.ipynb
- results
	- 2019113025_LM1_train-perplexity.txt
	- 2019113025_LM1_test-perplexity.txt
	- 2019113025_LM2_train-perplexity.txt
	- 2019113025_LM2_test-perplexity.txt
	- 2019113025_LM3_train-perplexity.txt
	- 2019113025_LM3_test-perplexity.txt
	- 2019113025_LM4_train-perplexity.txt
	- 2019113025_LM4_test-perplexity.txt
	- 2019113025_LM5_train_perplexity.txt
	- 2019113025_LM5_test_perplexity.txt
	- 2019113025_LM6_train_perplexity.txt
	- 2019113025_LM6_test_perplexity.txt
- corpus
	- Pride and Prejudice - Jane Austen.txt
	- Ulysses - James Joyce.txt
```
On cloud: [link](https://drive.google.com/drive/folders/1j3LgR1HzFwM4hVj9DzH21xEdlmH1e9qX?usp=share_link)
```
- PP_nlm.pkl
- U_nlm.pkl
```

## Modifications to the corpus
### Language Modelling
A number of modifications have been made to the corpus, as it initially had extremely large sentences. A number of "." were added manually at end of several sentences to prevent the probablity going to zero over multiple multiplications.
### Neural Language Modelling
Due to hardware restrictions I was forced to remove 8k lines from the corpus and perform the activity.  The entire corpus, of 35k lines, could not be loaded into the memory and be processed even on high performing computers [Ada].