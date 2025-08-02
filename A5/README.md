
[![Watch Video](https://github.com/aman010/A5_-dpo-nlp/blob/main/Screenshot%20from%202025-03-02%2018-28-17.png)](https://youtu.be/TAxzp0Dh7h8)



![Web image](https://github.com/aman010/A5_-dpo-nlp/blob/main/Screenshot%20from%202025-03-01%2013-15-09.png)

Tensorboard tracking , in the notebook i wad not able to resolve the issue with validation scores just added a print statment in evaluation loop,
that could be the one of the reason because evaluation score is not visible on tensorboard


Following Link to the GPT2-DPO model and config file
* https://huggingface.co/Aman010/GPT2-DPO/tree/main
* https://huggingface.co/Aman010/GPT2-DPO/blob/main/config.json




Observation challanges and mistakes

   * our objective was to make summarization model
   * we are using feedback_suumarization dataset
   * each prompt have multiple responses, each and each set of reponse have preference score 1 or 0 which is human in loop
   * our objective was to score more and better generalized model
   * now at the end with the given provied time we tried to make things possible and due to shortage of time could not make changes
   * i realied that GPT-2 is not suitable for summarization
   * we named it text generation or where given the input the model tend to generate text as per the genrated text above

It was good expereince to design and work with huggyface api's but because of shortage of time we made few mistakes above. 
Although the model is trained and uploaded and short time so no changes were made.



