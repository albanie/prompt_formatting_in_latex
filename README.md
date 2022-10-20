# Formatting script for appendix

A script to format prompts for insertion into a latex appendix.

## Installation

```bash

# install the current repo
git clone git@github.com:albanie/prompt_formatting_in_latex.git
cd prompt_formatting_in_latex

# different prompts are stored on different branches of promptsource - we need
# multiple branches to be accessible for formatting purposes
git clone git@github.com:Muennighoff/promptsource.git promptsource_tr13
cd promptsource_tr13 ; git fetch ; git switch tr13 ; cd ..
git clone git@github.com:Muennighoff/promptsource.git promptsource_xp3mt
cd promptsource_xp3mt ; git fetch ; git switch xp3mt ; cd ..

# install python dependencies
pip install -q iso-639
pip install tqdm
pip install beartype
pip install pandas
pip install bibtexparser
pip install zsvision
```

## Generate latex appendix

```bash

# Replace this with your local copy of story_cloze (which is not available through the datasets hub)
PATH_TO_STORY_CLOZE_DIR=~/data/shared-datasets/story_cloze

# English-only (note that if t0 datasets are excluded, this will not produce any output)
ipy generate_latex_appendix.py -- \
  --language_status english_only \
  --english_only_promptsource_dir promptsource_tr13 \
  --story_cloze_dir $PATH_TO_STORY_CLOZE_DIR \
  --exclude_t0_datasets 1 \
  --refresh

# Multilingual
ipy generate_latex_appendix.py -- \
  --language_status multilingual \
  --english_only_promptsource_dir promptsource_xp3mt \
  --exclude_t0_datasets 1 \
  --refresh
```
