
[![Watch Video](https://github.com/aman010/A4-Bert/blob/main/Screenshot%20from%202025-02-24%2000-38-34.png)](https://youtu.be/xoO7oBN6fGM)


The above videos links is about how the app works, in this work we are not able to deploy the web app, possibly tried to push the model to huggyface and run it with streamlet 
which is still not possoble because of size limitation. 

![Web image](https://github.com/aman010/A4-Bert/blob/main/Screenshot%20from%202025-02-23%2015-57-42.png)

Other options was added to take input from user.

![BERT Model Screenshot](https://raw.githubusercontent.com/aman010/A4-Bert/main/Screenshot%20from%202025-02-23%2012-30-56.png)


With the above loss of pretrain Bert model we observe that it over optimzed very quickly with the training , we took 100000 samples from openweb text.
Tried few experiments
  *  number of layer to 8
  *  added dropout to 0.3
  *  added ReduceLROnPlateau
  *  added weighted decay (L2)
  *  This is the best we can get the above one is number of layers 8
  *  The sample is non random of fisrt 100000 samples the training took almost an 30-45 mins
  *  If enough time is left we will try to do other experiements to optimize the pretrain model
  *  with this lets move to siamese network







| Model                  | Accuracy_MNLI | Traning Loss MNIL | Traning Time | sample size |    
|------------------------|---------------|-------------------|--------------|-------------|
| Bert Pretrain siamese  | 0.997         |   2.71            |     <40min   |   > 1000    |

Even though the performance is showing 0.99 but it is hard to justify the models i train that are not able to predict neutral examples very well may be because of fine-tuning is not done properly since the memory limitation and crashing of notebook 
again and again. I took random shuffle sample > 1000 which also took around 16 gb of ram. Moreover GPUs were not efficient enough to perform and gave out of memory, tarin in the loop have to convert all tensors into numpy array which added more overhead but doing this was able have more control over memory. The other techniques to work around with limited memory than just gc.collect ,could be pyarrow paging , but i still doubt if there is any support for pyarrow and tensor type. More over model distillation from large model would still cause the same issue. We can not just rely on those cosine similiray we need human in loop metric to evaluate best model.

Thanks to professor that gave us chance to work on such an awsome thing doing this have raise my to-do list.
