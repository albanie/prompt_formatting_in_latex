"""Generate latex appendix file for inclusion in paper.

Note: this script is derived from the latex.py file in the T0 arxiv source code
which can be found here https://arxiv.org/abs/2110.08207

TODO: Explain why we convert to csv as an intermediate step.
"""

import datasets
import tqdm
import argparse
from beartype import beartype
import pandas as pd
from pathlib import Path
import promptsource.templates
from typing import Tuple
from xp3_datasets import TRAIN_DATASETS, EVAL_DATASETS_L1, EVAL_DATASETS_L2


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
        dataset_metadata: pd.DataFrame,
) -> Tuple[dict, dict]:

    types = {}
    task_info = {}

    # Sets up Jinja environment
    # env = promptsource.templates.env

    # Loads templates and iterates over each data (sub)set
    template_collection = promptsource.templates.TemplateCollection()

    skipped_empty = 0

    for idx, p in tqdm.tqdm(dataset_metadata.iterrows()):

        hf_name = p["HF_name"]
        subset = p["subset"]
        temp = template_collection.get_dataset(hf_name, subset)
        
        # skip this template collection if it is empty
        if not list(temp.templates.keys()):
            continue

        for k in temp.templates.keys():
            temp2 = temp.templates[k]

            # Skip empty prompt templates
            if not temp2.jinja:
                print(f"skipping empty prompt for prompt key {k}")
                skipped_empty += 1
                continue

            q, r = temp2.jinja.split("|||")

            cat, task = p["task_by_convention"], hf_name + " " + subset
            cat = cat.strip()
            task = task.strip()

            types.setdefault(cat, {})
            types[cat].setdefault(task, [])

            types[cat][task].append({"q": q.strip(), "r": r.strip(),
                                    "uuid" : temp2.id,
                                    "original" : temp2.metadata.original_task,
                                    "c": temp2.get_answer_choices_expr()})
            task_info[task] = {"eval" : bool(p["do_eval"])}
    return types, task_info


@beartype
def generate_latex(
        dataset_metadata_csv: Path,
        prompt_citation_csv: Path,
        dest_latex_path: Path,
        dest_bib_path: Path,
        limit: int,
        refresh: bool,
):

    # Avoid overwriting exisitng appendix file unless explicitly requested
    if dest_latex_path.exists() and not refresh:
        print(f"Found existing latex prompt appendix at {dest_latex_path}, skipping...")
        return

    # build a look up table mapping from prompt UUIDs to citation references
    prompt_citations = pd.read_csv(prompt_citation_csv).fillna("")
    cite_dict = {}
    for _, c in prompt_citations.iterrows():
        cite_dict[c["uuid"]] = c[r"\cite ref"]

    # Read in the metadata corresponding to the datasets of interest
    dataset_metadata = pd.read_csv(dataset_metadata_csv).fillna("")

    # process a limited subset of the datasets (for fast debugging)
    if limit:
        dataset_metadata = dataset_metadata[:limit]

    types, task_info = parse_tasks_and_type_info(
        dataset_metadata=dataset_metadata,
    )

    write_latex_and_bib_entries_to_disk(
        types=types,
        task_info=task_info,
        dest_latex_path=dest_latex_path,
        dest_bib_path=dest_bib_path,
        cite_dict=cite_dict,
    )


@beartype
def generate_metadata_csv_file(
        dest_dataset_metadata_csv: Path,
):
    """Generate a csv file containing metadata about the datasets used by BLOOMZ
    in the format expected by the LaTeX generator.

    Args:
        dest_dataset_metadata_csv: the path to the csv file to be generated
    """
    pass


@beartype
def write_latex_and_bib_entries_to_disk(
        types: dict,
        task_info: dict,
        cite_dict: dict,
        dest_latex_path: Path,
        dest_bib_path: Path,
):

    with open(dest_latex_path, "w") as lf:
        for tk in tqdm.tqdm(types):
            print(r"\subsection{%s}"%capitalise_each_word(tk.replace("_", " ")), file=lf)

            for dataset in types[tk]:

                p = dataset.split()
                dataset_data = datasets.load_dataset(p[0], p[1] if len(p) == 2 else None)
                print(r"\subsubsection{%s}"%dataset.replace("_", "\_"), file=lf)
                tr = list(dataset_data.keys())

                cit = dataset_data[tr[0]].info.citation
                if cit and "{" in cit:
                    with open(dest_bib_path, "w") as bib_file:
                        print(cit, file=bib_file)
                    x = cit.split("{")[1].split(",")[0]
                    print(r"\noindent Dataset from \citet{%s}."%x, file=lf)
                    print(r"Used in %s."%("evaluation" if task_info[dataset]["eval"] else "training"), file=lf)

                dic = dataset_data[tr[0]][0]
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
                
                for t in types[tk][dataset]:
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
    parser.add_argument("--task", choices=["generate_metadata_csv", "generate_latex"])
    parser.add_argument("--dataset_metadata_csv", default="xP3.csv", choices=["xP3.csv", "D4.csv"], type=Path)
    parser.add_argument("--prompt_citation_csv", default="PromptCite.csv", type=Path)
    parser.add_argument("--dest_latex_path", default="promptgen.tex", type=Path)
    parser.add_argument("--dest_bib_path", default="promptgen.bib", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="use a small subset of datasets for fast debugging")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "generate_metadata_csv":
        generate_metadata_csv_file(
            dest_dataset_metadata_csv=args.dataset_metadata_csv,
        )
    elif args.task == "generate_latex":
        generate_latex(
            dataset_metadata_csv=args.dataset_metadata_csv,
            prompt_citation_csv=args.prompt_citation_csv,
            dest_latex_path=args.dest_latex_path,
            dest_bib_path=args.dest_bib_path,
            limit=args.limit,
            refresh=args.refresh,
        )


if __name__ == "__main__":
    main()