"""Generate latex appendix of formatted prompts.

Note: the latex generation functions used in this script are derived from the latex.py
file accompanying the T0 arxiv source code which can be found here:
https://arxiv.org/abs/2110.08207
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import re
from zsvision.zs_utils import BlockTimer


import datasets
import pandas as pd
import bibtexparser
import tqdm
from beartype import beartype

from xp3_datasets import (EVAL_DATASETS_L1, EVAL_DATASETS_L2, LANGUAGE_STATUS_MAP,
                          TASK_HIERARCHY, TRAIN_DATASETS, T0_DATASETS, FLORES_LANGS)
from citation_fixes import MANUAL_CITATION_FIXES

# Certain datasets appear to break sometimes when streamed. We handle such
# datasets explicitly by fetching a full copy of the dataset.
NON_STREAMABLE_DATASETS = {
    "web_questions", # possibly breaking due to the flaky hosting of codalab
    # the following datasets do not support streaming mode on tar archives
    "trivia_qa unfiltered",
    "GEM/wiki_lingua ar",
    "GEM/wiki_lingua en",
    "GEM/wiki_lingua es",
    "GEM/wiki_lingua fr",
    "GEM/wiki_lingua hi",
    "GEM/wiki_lingua id",
    "GEM/wiki_lingua pt",
    "GEM/wiki_lingua vi",
    "GEM/wiki_lingua zh",
    "openai_humaneval", # breaks with infinite loop
    "multi_eurlex all_languages" # just hangs
}

# We further avoid streaming dataset families that employ the following prefixes
# (prefixes are used, because there are large numbers of such datasets)
NON_STREAMABLE_DATASET_PREFIXES = {
    "GEM/xlsum", # datasets crash due to a misconfigured file path in streaming mode
    "GEM/BiSECT", # datasets have streaming issues with HTTP
    "facebook/flores", # datasets crash on encoding errors when streaming
}

# We exclude datasets for which source access has not yet been obtained
TEMPORARY_EXCLUSION = {
    ('Muennighoff/xstory_cloze', 'ar'),
    ('Muennighoff/xstory_cloze', 'es'),
    ('Muennighoff/xstory_cloze', 'eu'),
    ('Muennighoff/xstory_cloze', 'id'),
    ('Muennighoff/xstory_cloze', 'hi'),
    ('Muennighoff/xstory_cloze', 'te'),
    ('Muennighoff/xstory_cloze', 'sw'),
    ('Muennighoff/xstory_cloze', 'zh'),
}

# Individual prompts which appear to have issues are excluded here
EXLCUDE_SPECIFIC_PROMPTS = {
    "d11ff462-c976-4fcd-8a66-2a11dbae1f9f"  # this prompt lacks a ||| separator
}

# A mapping from language codes to poylgossia-compatible language names (used
# for typesetting)
POLYGLOSSIA_LANGUAGE_CODE_MAP = {
    "aeb" : "arabic",# Tunisia
    "af" : "afrikaans",
    "afb" : "arabic",# Gulf States
    "am" : "amharic",
    "apd" : "arabic",# Sudan
    "ar" : "arabic",
    "ar-MR" : "arabic",# Mauritania
    "ar-IQ" : "arabic",# Iraq
    "ar-JO" : "arabic",# Jordan
    "ar-LB" : "arabic",# Lebanon
    "ar-PS" : "arabic",# Lebanon
    "ar-SY" : "arabic",# Syria
    "ar-YE" : "arabic",# Yemen
    "arq" : "arabic",# Algeria
    "ary" : "arabic",# Morocco
    "arz" : "arabic",# Egypt
    "ast" : "asturian",
    "ayl" : "arabic",# Libya
    "be" : "belarusian",
    "be-tarask" : "belarusian",
    "bg" : "bulgarian",
    "bn" : "bengali",
    "bo" : "tibetan",
    "br" : "breton",
    "bs" : "bosnian",
    "ca" : "catalan",
    "ckb" : "kurdish",
    "ckb-Arab" : "kurdish",
    "ckb-Latn" : "kurdish",
    "cop" : "coptic",
    "cy" : "welsh",
    "cz" : "czech",
    "da" : "danish",
    "de" : "german",
    "de-Latf-AT" : "german",
    "de-AT" : "german",
    "de-Latf-CH" : "german",
    "de-CH" : "german",
    "de-Latf-DE" : "german",
    "de-DE" : "german",
    "de-AT-1901" : "german",
    "de-AT-1996" : "german",
    "de-CH-1901" : "german",
    "de-CH-1996" : "german",
    "de-DE-1901" : "german",
    "de-DE-1996" : "german",
    "de-Latf" : "german",
    "de-Latf-AT-1901" : "german",
    "de-Latf-AT-1996" : "german",
    "de-Latf-CH-1901" : "german",
    "de-Latf-CH-1996" : "german",
    "de-Latf-DE-1901" : "german",
    "de-Latf-DE-1996" : "german",
    "dsb" : "sorbian",
    "dv" : "divehi",
    "el" : "greek",
    "el-monoton" : "greek",
    "el-polyton" : "greek",
    "en" : "english",
    "en-AU" : "english",
    "en-CA" : "english",
    "en-GB" : "english",
    "en-NZ" : "english",
    "en-US" : "english",
    "eo" : "esperanto",
    "es" : "spanish",
    "es-ES" : "spanish",
    "es-MX" : "spanish",
    "et" : "estonian",
    "eu" : "basque",
    "fa" : "persian",
    "fi" : "finnish",
    "fr" : "french",
    "fr-CA" : "french",
    "fr-FR" : "french",
    "fur" : "friulian",
    "ga" : "gaelic",
    "gd" : "gaelic",
    "gl" : "galician",
    "grc" : "greek",
    "he" : "hebrew",
    "hi" : "hindi",
    "hr" : "croatian",
    "hsb" : "sorbian",
    "hu" : "hungarian",
    "hy" : "armenian",
    "ia" : "interlingua",
    "id" : "malay",
    "is" : "icelandic",
    "it" : "italian",
    "ja" : "japanese",
    "ka" : "georgian",
    "km" : "khmer",
    "kmr" : "kurdish",
    "kmr-Arab" : "kurdish",
    "kmr-Latn" : "kurdish",
    "kn" : "kannada",
    "ko" : "korean",
    "ku" : "kurdish",
    "ku-Arab" : "kurdish",
    "ku-Latn" : "kurdish",
    "la" : "latin",
    "la-x-classic" : "latin",
    "la-x-ecclesia" : "latin",
    "la-x-medieval" : "latin",
    "lo" : "lao",
    "lt" : "lithuanian",
    "lv" : "latvian",
    "mk" : "macedonian",
    "ml" : "malayalam",
    "mn" : "mongolian",
    "mr" : "marathi",
    "nb" : "norwegian",
    "nko" : "nko",
    "nl" : "dutch",
    "nn" : "norwegian",
    "oc" : "occitan",
    "pl" : "polish",
    "pms" : "piedmontese",
    "pt" : "portuguese",
    "pt-BR" : "portuguese",
    "pt-PT" : "portuguese",
    "rm" : "romansh",
    "ro" : "romanian",
    "ru" : "russian",
    "ru-petr1708" : "russian",
    "ru-luna1918" : "russian",
    "sa" : "sanskrit",
    "sa-Deva" : "sanskrit",
    "sa-Beng" : "sanskrit",
    "sa-Gujr" : "sanskrit",
    "sa-Knda" : "sanskrit",
    "sa-Mlym" : "sanskrit",
    "sa-Telu" : "sanskrit",
    "se" : "sami",
    "sk" : "slovak",
    "sl" : "slovenian",
    "sq" : "albanian",
    "sr" : "serbian",
    "sr-Cyrl" : "serbian",
    "sr-Latn" : "serbian",
    "sv" : "swedish",
    "syr" : "syriac",
    "ta" : "tamil",
    "te" : "telugu",
    "th" : "thai",
    "tk" : "turkmen",
    "tr" : "turkish",
    "uk" : "ukrainian",
    "ur" : "urdu",
    "vi" : "vietnamese",
    "zsm" : "malay",
}

# Various datasets use different langauge naming conventions - we provide
# a small number of manual mappings below
FLORES_LANG_MAP = {code: lang_name for (lang_name, code) in FLORES_LANGS}

SMALL_WMT_LANG_MAP = {
    "ibo": "igbo",
    "eng": "english",
    "fra": "french",
    "swh": "swahili",
}

SMALL_TATOEBA_LANG_MAP = {
    "ara": "arabic",
    "fra": "french",
    "eng": "english",
    "spa": "spanish",
    "ben": "bengali",
}

SMALL_XCOPA_LANG_MAP = {
    "zh": "chinese", 
    "it": "italian",
    "vi": "vietnamese",
}

SMALL_XQUAD_LANG_MAP = {
    "ar": "arabic",
    "zh": "chinese",
    "vi": "vietnamese",
    "en": "englsh",
    "es": "spanish",
}

@beartype
def capitalise_each_word(st: str) -> str:
    """For a given string of words separated by spaces, capitalise the first letter
    of each word and leave a trailing space.

    Args:
        st: the input string to be capitalised

    Returns:
        the capitalised string with a trailing space

    Examples (note the trailing spaces in the outputs):
    input: "the"
    output: "The "
    input: "the cat sat on the mat"
    output: "The Cat Sat On The Mat "
    """
    s = ""
    for t in st.split():
        s += t[0].upper() + t[1:] + " "
    return s

@beartype
def find_tags(text: str) -> list:
    """Parse the location of tags within a string. The tags supported are:
    (i) {{ tag_content }}
    (ii) {% tag_content %}

    Args:
        text: the text string to be parsed

    Returns:
        a list of dictionaries containing the start and end indices of each tag
    """
    matches = list(re.finditer(r"{{([^}])*}}|{%([^%])*%}", text))
    return [{"start": m.start(), "end": m.end()} for m in matches]


@beartype
def get_language_tags(dataset: str, key: str, dic: dict = {}) -> tuple:
    """Get polyglossia language tags for the given dataset and key combination.
    These tags are used to wrap the text so that it can be typeset correctly by
    XeLaTeX.

    Args:
        dataset: the name of the dataset
        key: the key the specifies the kind of data to be tagged
        dic: an data example from the dataset (or empty)

    Returns:
    """
    # For non-latin characters, we need to use Poylglossia
    use_lang_tags, lang = False, ""
    if dataset.startswith("GEM/wiki_lingua"):
        # single language: 'GEM/wiki_lingua ar'
        # multi-language: 'GEM/wiki_lingua ar_fr'
        num_langs = len(dataset.split()[1].split("_"))
        if key in {"source", "target", "references"}:
            if num_langs == 1:
                # only the examples use non-English content
                if dic:
                    polyglossia_key = dataset.split()[1]
                else:
                    polyglossia_key = "en"
            elif num_langs == 2:
                if key in {"source", "references"}:
                    polyglossia_key = dic["source_language"]
                elif key == "target":
                    polyglossia_key = dic["target_language"]
            lang = POLYGLOSSIA_LANGUAGE_CODE_MAP[polyglossia_key]
            use_lang_tags = True
    elif dataset.startswith("GEM/xlsum"):
        lang = dataset.split()[1]
        use_lang_tags = key in {"title", "target", "references", "text", "source"}
    elif dataset.startswith("facebook/flores"):
        # example dataset string: "facebook/flores aka_Latn-asm_Beng"
        langs = dataset.split()[1].split("-")
        if key == "source":
            lang = FLORES_LANG_MAP[langs[0]]
        elif key == "target":
            lang = FLORES_LANG_MAP[langs[1]]
        elif key.startswith("sentence"):
            # example: "sentence_asm_Beng"
            lang = FLORES_LANG_MAP[key.split("_", 1)[1]]
        lang = lang.lower() # polyglossia uses lowercase language names (apart from Arabic)
        use_lang_tags = lang in {"arabic", "bengali"}
    elif dataset.startswith("allenai/wmt22_african"):
        # example: 'allenai/wmt22_african eng-ibo'
        langs = dataset.split()[1].split("-")
        if key == "source":
            lang = SMALL_WMT_LANG_MAP[langs[0]]
        elif key == "target":
            lang = SMALL_WMT_LANG_MAP[langs[1]]
        use_lang_tags = True
    elif dataset.startswith("Helsinki-NLP/tatoeba"):
        # example: 'Helsinki-NLP/tatoeba_mt spa-vie'
        langs = dataset.split()[1].split("-")
        if key in {"sourceString", "targetString"}:
            if key == "sourceString":
                lang = SMALL_TATOEBA_LANG_MAP[langs[0]]
            elif key == "targetString":
                lang = SMALL_TATOEBA_LANG_MAP[langs[1]]
            use_lang_tags = True
    elif dataset.startswith("khalidalt/tydiqa"):
        # example 'khalidalt/tydiqa-goldp arabic'
        lang = dataset.split()[1]
        if key in {"source", "target", "passage_text", "document_title", "question_text"}:
            use_lang_tags = True
    elif dataset.startswith("xnli"):
        # example 'xnli ar'
        polyglossia_key = dataset.split()[1]
        lang = POLYGLOSSIA_LANGUAGE_CODE_MAP[polyglossia_key]
        use_lang_tags = True
    elif dataset.startswith("xquad"):
        lang_code = dataset.split()[1].split(".")[1]
        lang = SMALL_XQUAD_LANG_MAP[lang_code]
        use_lang_tags = True
    # note: for arabic, we have to use a capital a to avoid a particular latex command
    lang = {"arabic": "Arabic"}.get(lang, lang)
    # disable language wrapping for languages with native support in XeLaTeX
    if lang in {"english", "chinese", "spanish", "vietnamese"}:
        use_lang_tags = False
    return use_lang_tags, lang

@beartype
def wrap_text_in_lang_tags(text, lang) -> str:
    # fix common formatting issue where newlines are broken
    text = text.replace(" \ n ", " \n ")
    tag_positions = find_tags(text)
    start_tag, end_tag = r"\begin{" + lang + "}", r"\end{" + lang + "}"
    combined = ""
    ptr = 0
    for tag_positions in tag_positions:
        start, end = tag_positions["start"], tag_positions["end"]
        sliced = text[ptr:start]
        if lang == "bengali":
            # for Bengali, we have to remove newlines
            sliced = sliced.replace("\\n", " ")
        if lang == "Arabic":
            if sliced:
                combined += r"\textarabic{" + sliced + "}"
        else:
            if sliced:
                combined += start_tag + sliced + end_tag
        if "+" in text[start:end]:
            combined += r"\verb&" + text[start:end] + "&"
        else:
            combined += r"\verb+" + text[start:end] + "+"
        ptr = end
    return combined


@beartype
def is_supported_latex_language(dataset_name: tuple) -> bool:
    # supported_long = {"arabic", "bengali", "english", "french"}
    # supported_short = {"ar", "en", "fr", "es", "pt", "zh", "vi"}
    # drop arabic support due to quote issues for now
    supported_long = {"bengali", "english", "french"}
    supported_short = {"en", "fr", "es", "pt", "zh", "vi"}

    name, subset = dataset_name
    if name == "facebook/flores":
        lang_codes = subset.split("-")
        lang_names = [FLORES_LANG_MAP[code] for code in lang_codes]
        supported_language = all([lang.lower() in supported_long for lang in lang_names])
    elif name == "GEM/wiki_lingua":
        supported_language = subset in supported_short
    elif name == "GEM/xlsum":
        supported_language = subset in supported_long
    elif name == "Helsinki-NLP/tatoeba_mt":
        # example 'ara-eng'
        lang_codes = subset.split("-")
        if not all([code in SMALL_TATOEBA_LANG_MAP for code in lang_codes]):
            supported_language = False
        else:
            lang_names = [SMALL_TATOEBA_LANG_MAP[code] for code in lang_codes]
            supported_language = all([lang in supported_long for lang in lang_names])
    elif name == "allenai/wmt22_african":
        lang_codes = subset.split("-")
        if not all([code in SMALL_WMT_LANG_MAP for code in lang_codes]):
            supported_language = False
        else:
            lang_names = [SMALL_WMT_LANG_MAP[code] for code in lang_codes]
            supported_language = all([lang in supported_long for lang in lang_names])
    elif name.startswith("khalidalt/tydiqa"):
        # I don't yet have a mechanism to blend languages, so we cannot support this yet
        supported_language = False
        # supported_language = subset in supported_long
    elif name.startswith("xcopa"):
        lang_code = subset
        supported_language = lang_code in supported_short
    elif name == "xnli":
        lang_code = subset
        supported_language = lang_code in supported_short
    elif name == "mlqa":
        # example value for subset: "mlqa.vi.vi"
        lang_codes = subset.split(".")[1:]
        supported_language = all([lang_code in supported_short for lang_code in lang_codes])
    elif name == "xquad":
        # example value for subset: "xquad.ar"
        lang_code = subset.split(".")[1]
        supported_language = lang_code in supported_short
    else:
        supported_language = True
    return supported_language


@beartype
def parse_tasks_and_type_info(
        dataset_metadata: List[dict],
        template_collection,
) -> Tuple[dict, dict]:
    """Parse information about individual prompts from the dataset metadata and group
    this information by task.

    Args:
        dataset_metadata: the metadata associated with the collection of datasets to
            be processed.
        template_collection: an instance of promptsource.tempalates.TemplateCollection

    Returns:
        A tuple of two dictionaries. The first dictionary maps from high-level task
        category names to dictionaries of datasets and prompts. The second indicates
        whether the given dataset was used during training or evaluation 
    """

    types = {}
    task_info = {}

    skipped_stats = defaultdict(int)

    for metadata in tqdm.tqdm(dataset_metadata):

        hf_name = metadata["HF_name"]
        subset = metadata["subset"]
        temp = template_collection.get_dataset(hf_name, subset)

        # skip this template collection if it is empty
        if not list(temp.templates.keys()):
            continue

        for k in temp.templates.keys():
            temp2 = temp.templates[k]

            # Skip empty prompt templates
            if not temp2.jinja:
                print(f"skipping empty prompt for prompt key {k}")
                skipped_stats["empty"] += 1
                continue
            
            # Skip prompts which appear to have issues
            if temp2.id in EXLCUDE_SPECIFIC_PROMPTS:
                print(f"skipping prompt {temp2.id} for prompt key {k}")
                skipped_stats["manually_excluded"] += 1
                continue

            source, target = temp2.jinja.split("|||")
            category, dataset_tag = metadata["task_by_convention"], hf_name + " " + subset
            category = category.strip()
            dataset_tag = dataset_tag.strip()

            types.setdefault(category, {})
            types[category].setdefault(dataset_tag, [])

            # handle languages
            source = source.strip()
            use_lang_tags, source_lang = get_language_tags(dataset=dataset_tag, key="source")
            if use_lang_tags:
                source = wrap_text_in_lang_tags(source, lang=source_lang)

            target = target.strip()
            use_lang_tags, target_lang = get_language_tags(dataset=dataset_tag, key="target")
            if use_lang_tags:
                target = wrap_text_in_lang_tags(target, lang=target_lang)

            types[category][dataset_tag].append({
                "source": {"text": source, "language": source_lang},
                "target": {"text": target, "language": target_lang},
                "uuid" : temp2.id,
                "original" : temp2.metadata.original_task,
                "reference": temp2.reference,
                "choices": temp2.get_answer_choices_expr(),
            })
            task_info[dataset_tag] = {"eval" : bool(metadata["do_eval"])}

    # Provide a summary of the number of prompts which were skipped (and the reasons)
    print(f"Skipped statistics: {dict(skipped_stats)}")

    return types, task_info


@beartype
def generate_latex_and_bib(
        prompt_citation_csv: Path,
        dest_latex_path: Path,
        dest_bib_path: Path,
        language_status: str,
        exclude_t0_datasets: bool,
        redundant_datasets: List[str],
        max_reduandant_dataset_count: int,
        story_cloze_dir: Path,
        filter_str: str,
        limit: int,
        refresh: bool,
        dest_nocite_path: Path,
        template_collection,
):
    """Generate latex and corresponding bib files that can be used in the paper appendix
    to provide an overview of the prompts used for training and evaluation.
    """

    # Avoid overwriting exisitng appendix file unless explicitly requested
    if dest_latex_path.exists() and not refresh:
        print(f"Found existing latex prompt appendix at {dest_latex_path}, skipping...")
        return

    # build a look up table mapping from prompt UUIDs to citation references
    prompt_citations = pd.read_csv(prompt_citation_csv).fillna("")
    cite_dict = {}
    for _, c in prompt_citations.iterrows():
        cite_dict[c["uuid"]] = c[r"\cite ref"]

    # combine all datasets for prompt generation
    all_dataset_names = TRAIN_DATASETS + EVAL_DATASETS_L1 + EVAL_DATASETS_L2

    # keep track of which datasets were used for training and evaluation
    eval_datasets = set(EVAL_DATASETS_L1 + EVAL_DATASETS_L2)

    # link each dataset to its task cluster
    task_map = {}
    for task, dataset_names in TASK_HIERARCHY.items():
        for dataset_name in dataset_names:
            task_map[dataset_name] = task

    # we filter datasets according to their language status. To perform the filter
    # we convert the datasets lists to sets for membership testing.
    language_status_map = {language: set(datasets) for language, datasets
                           in LANGUAGE_STATUS_MAP.items()}
    all_datasets_with_known_language_status = set.union(*language_status_map.values())

    # if requested, we remove datasets from the T0 paper (since the corresponding prompts
    # have already been included in the appendix). To support this, we build a filter set
    t0_dataset_filter = set(T0_DATASETS["T0_TRAIN"] + T0_DATASETS["T0+_TRAIN"] +
                            T0_DATASETS["T0++_TRAIN"] + T0_DATASETS["EVAL"])
    exclude_t0_datasets_counter = 0

    # if requested, limit the number of prompts used for each kind of translation dataset
    # to avoid heavy duplication. We also skip datasets for which we cannot find latex
    # font support
    if max_reduandant_dataset_count:
        pre_filter_len = len(all_dataset_names)
        prefix_counter = defaultdict(int)
        drop = []
        for name in all_dataset_names:
            prefix = name[0]
            if not is_supported_latex_language(name):
                drop.append(name)
            elif prefix in redundant_datasets and prefix_counter[prefix] >= max_reduandant_dataset_count:
                drop.append(name)
            else:
                prefix_counter[prefix] += 1
        drop = set(drop)
        all_dataset_names = [name for name in all_dataset_names if name not in drop]
        print(f"Limiting redundant datasets to avoid prompt duplication. This reduces "
              f"the total datasets from {pre_filter_len} to {len(all_dataset_names)}")

    all_dataset_names = sorted(all_dataset_names)

    # build a lookup table of relevant metadata for each dataset
    dataset_metadata = []
    for dataset_name in all_dataset_names:

        # skip datasets that do not have the requested language status
        assert dataset_name in all_datasets_with_known_language_status, (
            f"the language status of {dataset_name} is unknown - cannot safely process..."
        )
        if dataset_name not in language_status_map[language_status]:
            continue

        # if requested, remove datasets from the T0 paper
        if exclude_t0_datasets and dataset_name in t0_dataset_filter:
            exclude_t0_datasets_counter += 1
            continue

        if dataset_name in TEMPORARY_EXCLUSION:
            print(f"Skipping {dataset_name} until I can download it")
            continue

        hf_name, subset = dataset_name
        dataset_metadata.append({
            "HF_name": hf_name,
            "subset": subset if subset else "",
            "task_by_convention": task_map[dataset_name],
            "do_eval": dataset_name in eval_datasets,
        })

    if filter_str:
        print(f"Filtering to only include datasets containing '{filter_str}'")
        dataset_metadata = [metadata for metadata in dataset_metadata
                            if filter_str in metadata["HF_name"]]

    # process a limited subset of the datasets (for fast debugging)
    if limit:
        print(f"Filtering to only include the first {limit} datsaets")
        dataset_metadata = dataset_metadata[:limit]

    # provide a status summary of what is going to be processed
    print(f"Generating prompts spanning {len(dataset_metadata)} {language_status} datasets")
    if exclude_t0_datasets_counter:
        print(f"Note that {exclude_t0_datasets_counter} datasets were excluded, since "
              "they were included in the T0 paper")

    types, task_info = parse_tasks_and_type_info(
        dataset_metadata=dataset_metadata,
        template_collection=template_collection,
    )

    write_latex_and_bib_entries_to_disk(
        types=types,
        task_info=task_info,
        dest_latex_path=dest_latex_path,
        dest_bib_path=dest_bib_path,
        dest_nocite_path=dest_nocite_path,
        story_cloze_dir=story_cloze_dir,
        cite_dict=cite_dict,
    )


@beartype
def write_latex_and_bib_entries_to_disk(
        types: dict,
        task_info: dict,
        cite_dict: dict,
        story_cloze_dir: Path,
        dest_latex_path: Path,
        dest_nocite_path: Path,
        dest_bib_path: Path,
):
    # avoid adding duplicate citations to the bib file
    seen_citations = set()
    all_bib_keys = set()

    with open(dest_latex_path, "w") as lf, open(dest_bib_path, "w") as bib_file:
        for task_idx, task in enumerate(types):
            print(r"\subsection{%s}"%capitalise_each_word(task.replace("_", " ")), file=lf)

            for dataset_idx, dataset in enumerate(types[task]):
                print(f"Processing task {task_idx}/{len(types)}, "
                      f"dataset {dataset_idx}/{len(types[task])}")
                print(f"Task: {task}, dataset: {dataset}")

                p = dataset.split()

                # Some datasets appear to break when used with the streaming interface
                # so we use the non-streaming interface instead for these datasets
                use_streaming = (dataset not in NON_STREAMABLE_DATASETS and
                                 not any(dataset.startswith(prefix) for
                                 prefix in NON_STREAMABLE_DATASET_PREFIXES))
                kwargs = {"streaming": use_streaming}

                # Story cloze has to be handled as a special case
                if "story_cloze" in p[0]:
                    kwargs["data_dir"] = story_cloze_dir

                dataset_data = datasets.load_dataset(p[0], p[1] if len(p) == 2 else None,
                                                     **kwargs)

                print(r"\subsubsection{%s}"%dataset.replace("_", "\_"), file=lf)
                tr = list(dataset_data.keys())

                cit = dataset_data[tr[0]].info.citation
                # apply any manual citation fixes that are required
                cit = MANUAL_CITATION_FIXES.get(cit, cit)
                bib_entry = bibtexparser.loads(cit)
                bib_keys = [entry["ID"] for entry in bib_entry.entries]
                all_bib_keys.update(bib_keys)

                if cit and ("{" in cit) and (cit not in seen_citations):
                    print(cit, file=bib_file)
                    x = cit.split("{")[1].split(",")[0]
                    print(r"\noindent Dataset from \citet{%s}."%x, file=lf)
                    print(r"Used in %s."%("evaluation" if task_info[dataset]["eval"] else "training"), file=lf)
                    seen_citations.add(cit)

                # Use iterator to load example to be compatible with the streaming interface
                # Use the second example rather than the first
                dataset_iterator = iter(dataset_data[tr[0]])
                next(dataset_iterator) # skip the first example for excitement
                dic = next(dataset_iterator)

                print(r"\paragraph{Data Example}\mbox{}\\", file=lf)
                print("", file=lf)
                print(r"\begin{table}[h]", file=lf)
                print(r"\small", file=lf)
                print(r"\begin{tabular}{ll}", file=lf)
                print(r"\toprule ", file=lf)
                print(r"Key & Value\\", file=lf)
                print(r"\midrule ", file=lf)

                for key in dic:
                    if isinstance(dic[key], list):
                        value_str = ";".join([str(x) for x in dic[key]])
                    else:
                        value_str = str(dic[key])
                    if len(value_str) > 35:
                        value_str = f"{value_str[:35]}..."
                    # handle newlines (this can occur in source code, for example)
                    tokens = value_str.split("\n")
                    # For latin characters, we can print verbatim
                    if "+" in value_str:
                        start, end = r"\verb&", r"&"
                    else:
                        start, end = r"\verb+", r"+"
                    # For non-latin, we need to use polyglossia
                    use_lang_tags, lang = get_language_tags(dataset=dataset, key=key, dic=dic)

                    if use_lang_tags:
                        start, end = r"\begin{" + lang + "}", r"\end{" + lang + "}"

                    for token_idx, token in enumerate(tokens):
                        if token_idx == 0: # only print the key for the first token
                            print("\t", key.replace("_", "\_"), "&", start, token, end, r"\\", file=lf)
                        elif token.strip(): # only print if the token is not empty
                            print("\t", "&", start, token, end, r"\\", file=lf)

                print(r"\bottomrule", file=lf)
                print(r"\end{tabular}", file=lf)
                print(r"\end{table}", file=lf)

                print(r"\paragraph{Prompts}\mbox{}\\", file=lf)
                print("", file=lf)
                
                for t in types[task][dataset]:
                    if t["uuid"] in cite_dict:
                        print(r"\noindent{\small Prompt from \cite{%s}}"%(cite_dict[t["uuid"]]), file=lf)
                    if not t["original"]:
                        print(r"\noindent{\small \nooriginal}", file=lf)
                        print("", file=lf)
                    print(r"\inputtemplate", file=lf)
                    source_text, source_lang = t["source"]["text"], t["source"]["language"]

                    if source_lang in {"Arabic", "bengali"}:
                        print(r"\begin{mdframed}[hidealllines=true,backgroundcolor=bga]", file=lf)
                        print(source_text, file=lf)
                        print(r"\end{mdframed}", file=lf)
                    else:
                        print(r"\begin{minted}[breaklines, tabsize=2,breaksymbolleft=, fontsize=\small,bgcolor=bga]{django}", file=lf)
                        print(source_text, file=lf)
                        print(r"\end{minted}", file=lf)
                        print(r"\vspace*{-0.2cm}", file=lf)
                    print("", file=lf)

                    print(r"\targettemplate", file=lf)
                    target_text, target_lang = t["target"]["text"], t["target"]["language"]

                    if target_lang in {"Arabic", "bengali"}:
                        print(r"\begin{mdframed}[hidealllines=true,backgroundcolor=bg]", file=lf)
                        print(target_text, file=lf)
                        print(r"\end{mdframed}", file=lf)
                    else:
                        print(r"\begin{minted}[breaklines, tabsize=2,breaksymbolleft=, fontsize=\small,bgcolor=bg]{django}", file=lf)
                        print(target_text, file=lf)
                        print(r"\end{minted}", file=lf)

                    print(r"\textcolor[RGB]{220,220,220}{\rule{\linewidth}{0.2pt}}", file=lf)
                    if t["choices"]:
                        print(r"\choicestemplate", file=lf)
                        print(r"\begin{minted}[breaklines, tabsize=2,breaksymbolleft=, fontsize=\small, bgcolor=bgb]{django}", file=lf)
                        print(t["choices"], file=lf)
                        print(r"\end{minted}", file=lf)
                        print(r"\vspace*{-0.3cm}", file=lf)
                    print("", file=lf)

    print(f"Printing {len(all_bib_keys)} unique citations to {dest_nocite_path}")
    with open(dest_nocite_path, "w") as f:
        for bib_key in sorted(all_bib_keys):
            print(f"\\nocite{{{bib_key}}}", file=f)


@beartype
def parse_args() -> argparse.Namespace:
    # pylint: disable=line-too-long
    # flake8: noqa: E501
    parser = argparse.ArgumentParser()
    parser.add_argument("--language_status", choices=["english_only", "multilingual"])
    parser.add_argument("--english_only_promptsource_dir", default="promptsource_tr13", type=Path)
    parser.add_argument("--multilingual_promptsource_dir", default="promptsource_xp3mt", type=Path)
    parser.add_argument("--prompt_citation_csv", default="PromptCite.csv", type=Path)
    parser.add_argument("--dest_latex_suffix_path", default="promptgen.tex", type=Path)
    parser.add_argument("--dest_bib_suffix_path", default="promptgen.bib", type=Path)
    parser.add_argument("--dest_nocite_path", default="prompt_appendix_nocite.tex", type=Path, help="location to store a `nocite` file to synchronise the bibliographies")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--exclude_t0_datasets", type=int, default=1, help="exclude datasets that were used in the T0 paper (and thus already provide prompts in its appendix)")
    parser.add_argument("--redundant_datasets", nargs="+", type=str,
                       default=["GEM/wiki_lingua", "GEM/xlsum", "Helsinki-NLP/tatoeba_mt",
                                "xcopa", "xnli", "Muennighoff/xstory_cloze", "Muennighoff/xwinograd", "mlqa", "paws-x", "xquad",
                                "khalidalt/tydiqa-primary", "khalidalt/tydiqa-goldp", "GEM/wiki_lingua", "facebook/flores", "allenai/wmt22_african"],
                                help="since translation prompts tend to be duplicated many times, we only use one per dataset")
    parser.add_argument("--max_reduandant_dataset_count", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="use a small subset of datasets for fast debugging")
    parser.add_argument("--filter_str", default="", help="if provided, only included datasets that contain this filter string")
    parser.add_argument("--story_cloze_dir", default=Path("data/story_cloze_dir"), type=Path, help="location of local copy of story_cloze dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    # import the relevant template collection
    if args.language_status == "english_only":
        promptsource_dir = args.english_only_promptsource_dir
    elif args.language_status == "multilingual":
        promptsource_dir = args.multilingual_promptsource_dir
    else:
        raise ValueError(f"unknown language status: {args.language_status}")
    promptsource_module = f"{promptsource_dir}.promptsource.templates"
    class_name = "TemplateCollection"
    module = __import__(promptsource_module, fromlist=[class_name])
    with BlockTimer("importing promptsource module"):
        if args.filter_str:
            template_collection = getattr(module, class_name)(filter_str=args.filter_str)
        else:
            template_collection = getattr(module, class_name)()

    prefix = args.language_status
    if args.limit:
        prefix = f"{prefix}_limit{args.limit}"
    dest_latex_path = Path(f"{prefix}_{args.dest_latex_suffix_path}")
    dest_bib_path = Path(f"{prefix}_{args.dest_bib_suffix_path}")

    generate_latex_and_bib(
        prompt_citation_csv=args.prompt_citation_csv,
        language_status=args.language_status,
        dest_latex_path=dest_latex_path,
        dest_bib_path=dest_bib_path,
        template_collection=template_collection,
        exclude_t0_datasets=bool(args.exclude_t0_datasets),
        filter_str=args.filter_str,
        limit=args.limit,
        redundant_datasets=args.redundant_datasets,
        max_reduandant_dataset_count=args.max_reduandant_dataset_count,
        story_cloze_dir=args.story_cloze_dir,
        refresh=args.refresh,
        dest_nocite_path=args.dest_nocite_path,
    )


if __name__ == "__main__":
    main()
