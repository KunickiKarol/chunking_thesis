---
license: apache-2.0

task_categories:
- question-answering

size_categories:
- 10K<n<100K
---


# Dataset Card for NovelQA

<!-- Provide a quick summary of the dataset. -->

NovelQA is the first benchmark with the average input length over 200K for testing the long-text ability of LLMs. It comprises the texts of 89 novels (among which 28 are copy-write protected and not publicly published) and 2305 question-answer pairs on the details of these novels.

NOTICE: The Dataset Viewer is automatically generated and only display the novel content. We do not know how to remove or remake it, so you can just ignore it.


## Dataset Details

### Dataset Description

- **Language:** English

### Dataset Sources

- **Repository:** https://github.com/NovelQA/novelqa.github.io
- **Leaderboard:** https://novelqa.github.io
- **Competition:**: https://www.codabench.org/competitions/2727/#/pages-tab
- **Paper:** https://arxiv.org/abs/2403.12766


## Usage

We recommend you to use `git clone` command to download this dataset. The steps are detailed as follows.

1. Install Hugging Face Hub Client (if you have not done this).
    
    ```
    pip install huggingface_hub
    ```

2. Login to your Hugging Face account (if you have not done this).
    
    ```
    huggingface-cli login
    ```

3. Follow the prompt to input your username and your Hugging Face token (if you have not done this).
    
    If you have not obtained your token, you can set it by navigating to your Hugging Face account - 'Profile Setting' - 'Access Tokens', and generate a new one with at least read authority. Then you can copy your generated token and paste it in the CLI.

4. Clone this repository through 
   
   ```
   git clone https://huggingface.co/datasets/NovelQA/NovelQA
   ```


Or simply download the zip file `NovelQA.zip`.


## Dataset Structure

The dataset is structured as follows.

```
.
├── NovelQA.zip                 // the latest format dataset with bid/qid/eid as indices for better reading efficiency.
├── NovelQA-prev-format.zip     // the previous format dataset.
├── Books                       // the book contents of the novels.
│   └── PublicDomain
│       ├── bookid1.txt
│       └── bookid2.txt
├── Data                        // the corresponding QA-pairs of each book.
│   ├── CopyrightProtected
│   │   ├── bookid1.json
│   │   └── bookid2.json
│   └── PublicDomain
│       ├── bookid3.json
│       └── bookid4.json
├── Demonstration                // demonstration book and QA with gold answer.
├── Scripts                      // the dataloader script for reading/writing the most common data formats.
└── bookmeta.json                // books' metadata.

```

Among the `json` files, each file includes a `list` of `dict`s, each of which is structured as follows.

```json
[
    "QID":
        {
            "Aspect": "the question classification in 'aspect', e.g., 'times'",
            "Complexity": "the question classification in complexity, e.g., 'mh'",
            "Question": "the input question",
            "Options": {
                "A": "Option A",
                "B": "Option B",
                "C": "Option C (not applicable in several yes/no questions)",
                "D": "Option D (not application in several yes/no questions)"
            },
        },
    ...
]
```


## Metadata

### Books

You can check the book's metadata from the file `bookmeta.json`.

- `BID`: a unique book id
- `title`: book title
- `txtfile`: txt filename
- `jsonfile`: json filename
- `source`: book source: gutenberg or self-owned
- `link`: download link: gutenberg or null
- `whichgtb`: [us, au, ca] for gutenberg, gutenberg.au and gutenberg.ca if gutenberg, else null
- `copyright`: have copyright or public domain
- `yearpub`: retrieve from goodreads
- `author`: author name
- `yearperish`: author perish year
- `period`: book classified into which period
- `tokenlen`: book token count, tokenized by `Xenova/gpt-3.5-turbo-16k`


### QAs

And for the QA's classifcations, 

Complxity: 
- `mh`: stands for multi-hop. solving this question requires knowledge across all over the book.
- `sh`: solving this problem requires knowledge from only one chapter. might be solved simply by recalling wo referring back.
- `dtl`: stands for detail. the question involves character or place or plot that only appears once or twice, and has no impacts to other plots, and thus reading other part of the book will  not remind the readers of this detail and the readers possibly does not remember this detail after finishing the book as well. 

Aspects:
- `times`: stands for "how many times does sth appear". asserted to be `mh`.
- `meaning`: stands for interpretation of words and sentences.
- `settg`: when/where for other single event.
- `span`: search for the earliest and latest time, all the places involved. asserted to be `mh`.
- `role`: ask about characters information. "who".
- `relat`: relationship of characters.
- `plot`: "something happens".


## Citation

**BibTeX:**

```bibtex
@misc{novelqa,
      title={NovelQA: Benchmarking Question Answering on Documents Exceeding 200K Tokens}, 
      author={Cunxiang Wang and Ruoxi Ning and Boqi Pan and Tonghui Wu and Qipeng Guo and Cheng Deng and Guangsheng Bao and Xiangkun Hu and Zheng Zhang and Qian Wang and Yue Zhang},
      year={2024},
      eprint={2403.12766},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.12766}, 
}
```

## Term

Your participation and submission to this benchmark will naturally give your consent to the following terms.

The input data are only for internal evaluation use. 
Please do not publicly spread the input data online. 
The competition hosts are not responsible for any possible violation of novel copyright caused by the participants' spreading the input data publicly online.

## Contact

We welcome you to contact us if you:

- find problems downloading or using this dataset,
- find a necessity to get access to the original dataset (with answers and evidences),

please feel free to contact the first authors from the Arxiv paper!

**Important Note on Getting Access to Full Dataset**: 
Please do not send the data you received to others or publish the data online.
When we grant you access to the full dataset, an ID will be embedded into the data so that if possible data leakage happens, we are able to identify who published them online. 
So please remember to keep them secret.
