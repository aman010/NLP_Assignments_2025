| Model Type        | Training Loss | Test Set Performance |
|-------------------|---------------|----------------------|
| even_distilled    | 0.6212        | 0.9250               |
| odd_distilled     | 0.6068        | 0.8910               |
| even_lora_init    | 0.1357        | 0.0346               |
| odd_lora_init     | 0.1496        | 0.0324               |



The even model and odd model followed the similar trajectory , but they are overfitting, tried to add weight_decay 0.01 and lr_scheduler.The surpriszing factor was that the after removing lr_scheduler the model stop showing any prohress in evaluation. After few search we got to know that we are facing problem with exploding gradient because of the size of the model, which lr scheduler helped. In the lora model we tend to use gradient clipping and keep fix the lr_scheduler because of overfitting. Now the obervation was that the LoRA model stop giving imporvements in evaluation loss which were increasing and evaluation accuracy which was getting better. So by this we can prove that LoRA being a low rank solution can cause implicit reglularaization or low rank approximation that control overfitting.

For the web page we had issue with the dependecy and due to shortage of time we were not able to deploy it. Please follow the screen-cast for the working of thw webapp. We use huggying_face dataset with 3 classes 0 hate speech, 1 offensive language, 2 neither of them. 

[![Watch Video](https://github.com/aman010/NLP_A7/blob/main/Screenshot%20from%202025-03-23%2016-55-59.png)](https://youtu.be/TQxoCmlqVUY)


Thank you.


