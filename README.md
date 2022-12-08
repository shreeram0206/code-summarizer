# Code-Summarizer: Pre-trained Encoder-Decoder Models for Code Summarization

# Steps to follow to run Code Summarizer:

1. Clone the repo

2. Python Packages to be installed (recommend running pip3 install --upgrade pip before installing the below packages) : 
 - torch 
 - tensorboard
 - tree-sitter
 - transformers
 - gsutil 

3. Run the below command inside the working directory of code-summarizer repo to download the data

```
gsutil -m cp -r "gs://sfr-codet5-data-research/data" .
```

Optional: You can download the pre-trained and fine-tuned models using the below commands.

```
gsutil -m cp -r "gs://sfr-codet5-data-research/pretrained_models" .
```

```
gsutil -m cp -r "gs://sfr-codet5-data-research/finetuned_models" .
```

4. Change the working directory inside /sh/exp_with_args.sh (WORKDIR) to the path of your repository.
 
5. Batch sizes and number of epochs can be modified inside run_exp.py. The batch_size is set as 48 and can be changed on line 41 inside run_exp.py.
The training epochs can be set on line 29.


6. Set the python version according to the version on your machine (line 86 of the same file).

7. To build on the Seq2Seq model, we use the model tag 'roberta' to get the exact errors printed on the terminal. 

8. Command to initiate training on the model : 

```
python3 run_exp.py --model_tag roberta --task summarize --sub_task python
```

9. If you face an error similar to _AttributeError: module 'setuptools._distutils' has no attribute 'version'_ during training, it can be resolved by using the workaround 

```
pip3 install setuptools==59.5.0
```
