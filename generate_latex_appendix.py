"""Generate latex appendix of formatted prompts.

Note: the latex generation functions used in this script are derived from the latex.py
file accompanying the T0 arxiv source code which can be found here:
https://arxiv.org/abs/2110.08207
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import datasets
import pandas as pd
import tqdm
from beartype import beartype

from xp3_datasets import (EVAL_DATASETS_L1, EVAL_DATASETS_L2, LANGUAGE_STATUS_MAP,
                          TASK_HIERARCHY, TRAIN_DATASETS, T0_DATASETS)

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
}

NON_STREAMABLE_DATASET_PREFIXES = {
    "GEM/xlsum", # datasets crash due to a misconfigured file path in streaming mode
    "GEM/BiSECT", # datasets have streaming issues with HTTP
    "facebook/flores", # datasets crash on encoding errors when streaming
}

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

# Prompts which appear to have issues
EXLCUDE_SPECIFIC_PROMPTS = {
    "d11ff462-c976-4fcd-8a66-2a11dbae1f9f"  # this prompt lacks a ||| separator
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

            try:
                q, r = temp2.jinja.split("|||")
            except:
                import ipdb; ipdb.set_trace()

            category, dataset_tag = metadata["task_by_convention"], hf_name + " " + subset
            category = category.strip()
            dataset_tag = dataset_tag.strip()

            types.setdefault(category, {})
            types[category].setdefault(dataset_tag, [])

            types[category][dataset_tag].append({
                "q": q.strip(),
                "r": r.strip(),
                "uuid" : temp2.id,
                "original" : temp2.metadata.original_task,
                "c": temp2.get_answer_choices_expr(),
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
        filter_str: str,
        limit: int,
        refresh: bool,
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

    # we filter datasets according to their langauge status. To perform the filter
    # we convert the datasets lists to sets for membership testing.
    language_status_map = {language: set(datasets) for language, datasets
                           in LANGUAGE_STATUS_MAP.items()}
    all_datasets_with_known_language_status = set.union(*language_status_map.values())

    # if requested, we remove datasets from the T0 paper (since the corresponding prompts
    # have already been included in the appendix). To support this, we build a filter set
    t0_dataset_filter = set(T0_DATASETS["T0_TRAIN"] + T0_DATASETS["T0+_TRAIN"] +
                            T0_DATASETS["T0++_TRAIN"] + T0_DATASETS["EVAL"])
    exclude_t0_datasets_counter = 0


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
        print(f"Note that {exclude_t0_datasets_counter} datasets were excluded, since"
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
        cite_dict=cite_dict,
    )


@beartype
def write_latex_and_bib_entries_to_disk(
        types: dict,
        task_info: dict,
        cite_dict: dict,
        dest_latex_path: Path,
        dest_bib_path: Path,
):

    with open(dest_latex_path, "w") as lf, open(dest_bib_path, "w") as bib_file:
        for task in tqdm.tqdm(types):
            print(r"\subsection{%s}"%capitalise_each_word(task.replace("_", " ")), file=lf)

            for dataset in types[task]:
                print(f"Processing task {task}, dataset {dataset}")

                p = dataset.split()

                # Some datasets appear to break when used with the streaming interface
                # so we use the non-streaming interface instead for these datasets
                use_streaming = (dataset not in NON_STREAMABLE_DATASETS and
                                 not any(dataset.startswith(prefix) for
                                 prefix in NON_STREAMABLE_DATASET_PREFIXES))
                dataset_data = datasets.load_dataset(p[0], p[1] if len(p) == 2 else None,
                                                    streaming=use_streaming)

                print(r"\subsubsection{%s}"%dataset.replace("_", "\_"), file=lf)
                tr = list(dataset_data.keys())

                cit = dataset_data[tr[0]].info.citation
                
                if cit and "{" in cit:
                    print(cit, file=bib_file)
                    x = cit.split("{")[1].split(",")[0]
                    print(r"\noindent Dataset from \citet{%s}."%x, file=lf)
                    print(r"Used in %s."%("evaluation" if task_info[dataset]["eval"] else "training"), file=lf)

                # Use iterator to load example to be compatible with the streaming interface
                try:
                    dic = next(iter(dataset_data[tr[0]]))
                except:
                    import ipdb; ipdb.set_trace()
                print(r"\paragraph{Data Example}\mbox{}\\", file=lf)
                print("", file=lf)
                print(r"\begin{table}[h]", file=lf)
                print(r"\small", file=lf)
                print(r"\begin{tabular}{ll}", file=lf)
                print(r"\toprule ", file=lf)
                print(r"Key & Value\\", file=lf)
                print(r"\midrule ", file=lf)
                
                for k in dic:
                    d = str(dic[k])
                    if len(d) > 50:
                        print("\t", k.replace("_", "\_"), "&", r"\verb+", d[:50]+"...", r"+", r"\\", file=lf)
                    else:
                        print("\t", k.replace("_", "\_"), "&", r"\verb+", d, r"+", r"\\", file=lf)
                print(r"\bottomrule", file=lf)
                print(r"\end{tabular}", file=lf)
                print(r"\end{table}", file=lf)

                print(r"\paragraph{Prompts}\mbox{}\\", file=lf)
                print("", file=lf)
                
                for t in types[task][dataset]:
                    if t["uuid"] in cite_dict:
                        print(r"\noindent{\small Prompt from \cite{%s}}"%(cite_dict[t["uuid"]]), file=lf)
                    if not t["original"]:
                        print(r"\noindent{\small Prompt not from the original task.}", file=lf)
                    if t["c"] is not None:
                        print(r"\begin{minted}[breaklines, tabsize=2,breaksymbolleft=, fontsize=\small, bgcolor=bgb]{django}", file=lf)
                        print(t["c"], file=lf)
                        print(r"\end{minted}", file=lf)
                        print(r"\vspace*{-0.3cm}", file=lf)
                        print("", file=lf)
                    print(r"\begin{minted}[breaklines, tabsize=2,breaksymbolleft=, fontsize=\small]{django}", file=lf)
                    print(t["q"], file=lf)
                    print(r"\end{minted}", file=lf)
                    print(r"\vspace*{-0.2cm}", file=lf)
                    print("", file=lf)
                    print(r"\begin{minted}[breaklines, tabsize=2,breaksymbolleft=, fontsize=\small,bgcolor=bg]{django}", file=lf)
                    print(t["r"], file=lf)
                    print(r"\end{minted}", file=lf)
                    print(r"\textcolor[RGB]{220,220,220}{\rule{\linewidth}{0.2pt}}", file=lf)


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
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--exclude_t0_datasets", type=int, default=1, help="exclude datasets that were used in the T0 paper (and thus already provide prompts in its appendix)")
    parser.add_argument("--limit", type=int, default=0, help="use a small subset of datasets for fast debugging")
    parser.add_argument("--filter_str", default="", help="if provided, only included datasets that contain this filter string")
    return parser.parse_args()


def main():
    args = parse_args()

    # import the relevant template collection
    if args.language_status == "english_only":
        promptsource_dir = args.english_only_promptsource_dir
    elif args.language_status == "multilingual":
        promptsource_dir = args.multilingual_promptsource_dir
    else:
        raise ValueError(f"unknown langauge status: {args.language_status}")
    promptsource_module = f"{promptsource_dir}.promptsource.templates"
    class_name = "TemplateCollection"
    module = __import__(promptsource_module, fromlist=[class_name])
    template_collection = getattr(module, class_name)()

    dest_latex_path = Path(f"{args.language_status}_{args.dest_latex_suffix_path}")
    dest_bib_path = Path(f"{args.language_status}_{args.dest_bib_suffix_path}")

    generate_latex_and_bib(
        prompt_citation_csv=args.prompt_citation_csv,
        language_status=args.language_status,
        dest_latex_path=dest_latex_path,
        dest_bib_path=dest_bib_path,
        template_collection=template_collection,
        exclude_t0_datasets=bool(args.exclude_t0_datasets),
        filter_str=args.filter_str,
        limit=args.limit,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
