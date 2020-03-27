# Natural Language Processing Assignment

In this assignment, we will be working on the problem of Question Answering (QA) using NLP.  
The problem of QA is very well defined and is very under very active research. Given a corpus of text, and a question which relies on context based on the corpus of text, the system has to extract and return the words from the given corpus which sufficiently answer the given question.

###### Example 

Qestion: When will the wedding be?

Corpus: Prince Harry and fiancee American actress Meghan Markle have released more details about their May 19. A wedding, revealing that the event will include a carriage ride through Windsor so they can share the big day with the public. The couple will marry at noon in St. George’s Chapel, the 15th century church on the grounds of Windsor Castle that has long been the backdrop of choice for royal occasions. Harry’s grandmother, Queen Elizabeth II, gave permission for use of the venue and will attend the wedding. Kensington Palace said in a statement that the couple is “hugely grateful” for the many good wishes they have received and they hope the carriage ride will give the general public a chance to take part.  

Answer: May 19

More such examples can be seen at:  
1) https://demo.allennlp.org/reading-comprehension  
2) https://www.pragnakalp.com/demos/BERT-NLP-QnA-Demo/

# Assignment

In this assignment, you will be implementing a baseline model on Question Answering using the Stanford Question Answering Dataset (SQuAD)  
The dataset can be found at:  
  * Training: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json  
  * Testing: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

The model you will be implementing is described [here](https://cs224d.stanford.edu/reports/StrohMathur.pdf). On page 3 of this paper, you will find a sequence-to-sequence model described.

This paper uses GloVe for word embeddings but you are free to use any word embedding model.

### Task  
In this notebook you will aim to implement a model for question answering trained on the dataset linked above.

We expect you to:

-> **Write relevant data loaders**  
-> **Implement your model from scratch**   
-> **Perform hyperparameter optimization**   

You are free to use any deep learning framework although PyTorch is preferable.
Please share your QA model weights and language model weights / embeddings along with your submission.

## Submission 

To submit, fork this repository, finish and push your notebook on the forked repository then perform a pull request to this repository.  
Deadline for the submission is Monday, 27 March 2020 11:59 PM.
