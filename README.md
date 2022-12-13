# Computational RepresentationS of Therapist Language (CRSTL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://tldrlegal.com/license/mit-license)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-383)

## Table of Contents

1. [Introduction](#1-introduction)
2. [Generating CRSTL](#2-generating-crstl)
3. [Citation](#3-citation)

## 1. Introduction

### 1.1 Abstract

Although individual psychotherapy is generally effective for a range of mental health conditions, little is known about the moment-to-moment language use of effective therapists. Increased access to computational power, coupled with a rise in computermediated communication (telehealth), makes feasible the large-scale analyses of language use during psychotherapy. Transparent methodological approaches are lacking, however. Here we present novel methods to increase the efficiency of efforts to examine language use in psychotherapy. We evaluate three important aspects of therapist language use - timing, responsiveness, and consistency - across five clinically relevant language domains: pronouns, time orientation, emotional polarity, therapist tactics, and paralinguistic style. We find therapist language is dynamic within sessions, responds to patient language, and relates to patient symptom diagnosis but not symptom severity. Our results demonstrate that analyzing therapist language at scale is feasible and may help answer longstanding questions about specific behaviors of effective therapists.

### 1.2 Acknowledgements

A.S.M. was supported by grants from the National Institutes of Health, National Center for Advancing Translational Science, Clinical and Translational Science Award (KL2TR001083 and UL1TR001085), the Stanford Department of Psychiatry Innovator Grant Program, and the Stanford Human-Centered AI Institute. S.L.F. was supported by a Big Data to Knowledge (BD2K) grant from the National Institutes of Health (T32 LM012409) and a National Defense Science and Engineering Graduate Fellowship from the Department of Defense. N.H.S acknowledges support from the Mark and Debra Lesli Endowment for the Program for AI in Healthcare at Stanford Medicine. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health. We thank Fei-Fei Li and G. Terence Wilson for project guidance. We thank members of 2018 Spring AI Bootcamp, Pranav Rajpurkar, Andrew Ng, Suvadip Paul, Ben Cohen-Wang, Matthew Sun for collaboration. We thank Jon-Michael Knapp for assistance editing the manuscript. We thank all of the participating counseling centers, directors, therapists, and student patients.

## 2. Generating CRSTL

[Return to top](#computational-representations-of-therapist-language-crstl)

### 2.1 Prerequisites

Certain features (e.g., emotional polarity from EmoLex, various LIWC features) require access to established lexicons. These lexicons must be placed in the appropriate folder for the package to work.

#### 2.1.1 Obtaining EmoLex

In order to generate EmoLex features you must download the [NRC Word-Emotion Association Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm). 

#### 2.1.2 Obtaining LIWC

Similarly, in order to generate LIWC features you must have access to a LIWC dictionary. The LIWC lexicon is proprietary, so it is _not_ included in this repository.

The lexicon data can be acquired (purchased) from [liwc.net](http://liwc.net/).

* If you are a researcher at an academic institution, please contact [Dr. James W. Pennebaker](https://liberalarts.utexas.edu/psychology/faculty/pennebak) directly.
* For commercial use, contact [Receptiviti](https://www.receptiviti.com/), which is the company that holds exclusive commercial license.

If the version of LIWC that you purchased (or otherwise legitimately obtained as a researcher at an academic institution) does not provide a machine-readable `*.dic` file, please contact the distributor directly.

#### 2.1.3 Setting the configuration file

Once you have acquired the appropriate lexicon files (e.g., `LIWC2007_English100131.dic` for LIWC, `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` for EmoLex), adjust the `psynlp/features/config.py` file to reflect the path to where the lexicon files are stored.

### 2.2 Parsing Transcripts

Set the location of the transcripts you would like to have parsed in the `psynlp/features/config.py` file. Then call
```
python psynlp/features/parse.py
```
from the command line. The script will generate CRSTL for the transcripts stored in the location specified and will save a `transcripts.tsv` (tab-separated) file containing the results. This file can be read using e.g., `pandas` and is the basis for all other analyses (utterance-level, quintile-level, and session-level) in the associated paper.

## 3. Citation

[Return to top](#computational-representations-of-therapist-language-crstl)

Miner, A.S., Fleming, S.L., Haque, A. et al. A computational approach to measure the linguistic characteristics of psychotherapy timing, responsiveness, and consistency. *npj Mental Health Research* **1**, 19 (2022)  
[doi:10.1038/s44184-022-00020-9](https://doi.org/10.1038/s44184-022-00020-9)

```text
@article{miner2022computational,
  title={A computational approach to measure the linguistic characteristics of psychotherapy timing, responsiveness, and consistency},
  author={Miner, Adam S and Fleming, Scott L and Haque, Albert and Fries, Jason A and Althoff, Tim and Wilfley, Denise E and Agras, W Stewart and Milstein, Arnold and Hancock, Jeff and Asch, Steven M and others},
  journal={npj Mental Health Research},
  volume={1},
  number={1},
  pages={1--12},
  year={2022},
  publisher={Nature Publishing Group}
}
```

Uncovering the linguistic characteristics of psychotherapy: a computational approach to measure therapist language timing, responsiveness, and consistency

Methods for measuring and analyzing therapist language
