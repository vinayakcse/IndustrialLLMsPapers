# IndustrialSurveyLLM
This repository contains the Industrial Survey of LLMs papers and is based on our paper, ["LLMs with Industrial Lens: Deciphering the Challenges and Prospects – A Survey"]

You can cite our paper as the following
```
@misc{,
      title={LLMs with Industrial Lens: Deciphering the Challenges and Prospects – A Survey}, 
      author={A and B and C and D},
      year={2024},
      eprint={},
      archivePrefix={},
      primaryClass={}
}
```

We group the papers according to the application as [NLP](#nlp), [Tools-and-Frameworks](#tools-and-frameworks), [Code-generation](#code-generation), [Trustworthy-AI](#trustworthy-ai), [Retrival-and-Recommendation](#retrival-and-recommendation), [Security](#security), [Societal-impact](#societal-impact) and [Miscellaneous-applications](#miscellaneous-applications)

With in NLP subgroups Summarization, Question-Answering, Machine translation, Conversational, Sentiment analysis, Reasoning, Table-to-text generation can be found.
Under miscellaneous applications category subgroups Cloud management, Task planning, Forecasting-Analytics can be found.
 
## NLP
### Summarization
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|InstructPTS: Instruction-Tuning LLMs for Product Title Summarization|EMNLP  Industry Track|2023|FLAT-T5|Not mentioned explicitly|-|
|LLM Based Generation of Item-Description for Recommendation System|RecSys|2023|Alpaca-LoRa|MovieLens, Goodreads Book graph|-|
|Assess and Summarize: Improve Outage Understanding with Large Language Models|ESEC/FSE|2023|GPT-3.X|historical data of 3 years cloud systems|-|
|Beyond Summarization: Designing AI Support for Real-World Expository Writing Tasks|CHI In2Writing Workshop|2023|-|-|-|
|Building Real-World Meeting Summarization Systems using Large Language Models: A Practical Perspective|EMNLP Industry Track|2023|GPT-4, GPT3.5, PaLM-2, and LLaMA-2 13b, 7b|QMSUM, AMI, ICSI|-|

### Question-Answering
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|FlowMind: Automatic Workflow Generation with LLMs|ICAIF|2023|gpt-3.5-turbo|NCEN-QA, NCEN-QA-Easy, NCEN-QA-Intermediate, NCEN-QA-Hard|-|
|PROMPTCAP: Prompt-Guided Task-Aware Image Captioning|ICCV|2023|GPT-3|COCO, OK-VQA, A-OKVQA, WebQA|https://yushi-hu.github.io/promptcap_demo/; https://huggingface.co/tifa-benchmark/promptcap-coco-vqa|
|Benchmarking Large Language Models on CMExam - A Comprehensive Chinese Medical Exam Dataset|NeurIPS|2023|GPT-3.5 turbo, GPT-4, ChatGLM, LLaMA, Vicuna, Alpaca|CMExam|https://github.com/williamliujl/CMExam/tree/main|
|Empower Large Language Model to Perform Better on Industrial Domain-Specific Question Answering|EMNLP Industry Track|2023|GPT-4, GPT3.5, LLaMA-2|MSQA|-|


### Machine translation
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|Bootstrapping Multilingual Semantic Parsers using Large Language Models|EACL|2023|mT5-Large, PaLM|MTOP, MASSIVE|-|

### Conversational
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|Understanding the Benefits and Challenges of Deploying Conversational AI Leveraging Large Language Models for Public Health Intervention|CHI|2023|HyperCLOVA|-|https://guide.ncloud-docs.com/docs/en/clovacarecall-overview|
|“The less I type, the beter”: How AI Language Models can Enhance or Impede Communication for AAC Users|CHI|2023|-|Collected own data|-|
|I wouldn’t say offensive but...: Disability-Centered Perspectives on Large Language Models|FAccT|2023|LaMDA|-|-|

### Sentiment analysis
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis|WWW|2023|GPT-3, PaLM|custom reddit dataset, FiQA-News|-|

### Reasoning
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|MathPrompter: Mathematical Reasoning using Large Language Models|EMNLP Industry Track|2023|text-davinci-002, PaLM|MultiArith dataset|-|
|Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models|NeurIPS|2023|gpt-3.5-turbo , GPT-4|ScienceQA, TabMWP|https://github.com/lupantech/chameleon-llm|
|On the steerability of large language models toward data-drivenpersonas|CIKM|2023|GPT-Neo-1.3B,  GPT-Neo-2.7B,  GPT-J-6B, Falcon-7B-Instruct|OpinionQA|-|
|Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning|SIGIR|2023|CODEX|TabFact, WikiTableQuestion, FetaQA|-|
|Answering Causal Questions with Augmented LLMs|ICML Worshop|2023|GPT-3.5, GPT-4|-|-|

### Table-to-text generation
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|Investigating Table-to-Text Generation Capabilities of LLMs in Real-World Information Seeking Scenarios|EMNLP Industry Track|2023|GPT4, TULU, Pythia,  Alpaca, Vicuna, LLaMA-2, GPT-3.5|LOTNLG, F2WTQ|-|
|Unleashing the Potential of Data Lakes with Semantic Enrichment Using Foundation Model|ISWC|2023|GPT4, Llama2, FLAN-T5|-|-|
|Tabular Representation, Noisy Operators, and Impacts on Table Structure Understanding Tasks in LLMs|NeurIPS|2023|GPT-3.5 (text-davinci-003 endpoint)|AirQuality, HousingData, Diabetes, Wine Testing, Iris, Titanic, and ENB2012_data|https://github.com/microsoft/prose|

### Data generation
|LayoutGPT: Compositional Visual Planning and Generation with Large Language Models|NeurIPS|2023|Codex, GPT-3.5, GPT-3.5-chat and GPT-4|NSR-1K, 3D-FRONT|https://github.com/weixi-feng/LayoutGPT|
|FABRICATOR: An Open Source Toolkit for Generating Labeled Training Data with Teacher LLMs|ACL|2023|Used existing LLMs from Hugginhface, openAI, Azure, Anthropic, Cohere|IMDB, MRPC, SNLI, TREC-6, SQUAD|https://github.com/flairNLP/fabricator|

## Tools-and-Frameworks
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|Automatic Linking of Judgements to UK Supreme Court Hearings|EMNLP Industry Track|2023|GPT- text-embedding-ada-002|UK National Archive|-|
|LLMR: Real-time Prompting of Interactive Worlds using Large Language Models|NeurIPS|2023|Language model for mixed reality (LLMR) Dall.E-2 gpt-4|Not mentioned|-|
|Enabling Conversational Interaction with Mobile UI using Large Language Models|CHI|2023|PaLM |PixelHelp , AndroidHowTo,  Rico, Screen2Words, |https://github.com/google-research/google-research/tree/master/llm4mobile|
|PromptInfuser: Bringing User Interface Mock-ups to Life with Large Language Models|CHI Extended Abstract|2023|-|-|-|
|LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models|ACL|2023|-|Proprietory (Not mentioned)|https://microsoft.github.io/lida/|
|RALLE: A Framework for Developing and Evaluating Retrieval-Augmented Large Language Models|EMNLP System demonstrations|2023|Llama-2 Chat (13B, 70B), WizardVicunaLM-13B, Vicuna|KILT Benchmark|https://github.com/yhoshi3/RaLLe|
|PROGPROMPT: Generating Situated Robot Task Plans using Large Language Models|ICRA|2023|text-davinci-*, Codex, GPT3|-|https://github.com/NVlabs/progprompt-vh|
|Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance|CoRL|2023|LLaMA-13b|ALFRED|-|
|Exploring the Boundaries of GPT-4 in Radiology|EMNLP|2023|gpt-3.5-turbo, text-davinci-003, gpt-4-32k|MS-CXR-T, RadNLI, Chest ImaGenome, MIMIC, Open-i|-|

## Code-generation
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|CodePlan: Repository-level Coding using LLMs and Planning|FMDM@NeurIPS|2023|GPT-4-32k|Construct the own dataset|-|
|Enhancing Network Management Using Code Generated by Large Language Models|HotNet's|2023|GPT-4, GPT-3, Text-davinci-003 (a variant of GPT 3.5)  and Google Bard|Public code repositories|https://github.com/microsoft/NeMoEval|
|A Static Evaluation of Code Completion by Large Language Models|ACL|2023|CodeGen-350M, CodeGen-2B, CodeGen-6B, CodeGen-16B, |function completion dataset|-|
|Using LLMs to Customize the UI of Web Pages|UIST|2023| gpt3.5   Legacy (text-davinci-003),  Legacy (code-davinci-002),  Legacy (text-davinci-edit-001)|-|-|
|Generative AI for Programming Education: Benchmarking ChatGPT, GPT-4, and Human Tutors|ICER|2023|GPT-3.5, GPT-4|-|-|
|Grace: Language Models Meet Code Edits|ESEC/FSE|2023|CODEX, CODEt5|C3PO, Overwatch|-|
|Large Language Model fail at completing code with potential bugs|NeurIPS|2023|CODEGEN, INCODER|Buggy-HumanEval; Buggy-FixEval|https://github.com/amazon-science/buggy-code-completion|
|Multilingual evaluation of code generation models|ICLR|2023|Decoder-only transformer model|MBXP, Multilingual HumanEval, MathQA-X|https://github.com/amazon-science/mxeval|



## Trustworthy-AI
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|Finspector: A Human-Centered Visual Inspection Tool for Exploring and Comparing Biases among Foundation Models|ACL|2023|BERT, ALBERT, RoBERTa|CrowS-Pairs|https://github.com/IBM/finspector|
|INVITE: a Testbed of Automatically Generated Invalid Questions to Evaluate Large Language Models for Hallucinations|EMNLP|2023|GPTNeo-2.7B, GPTJ-6B, Open-LLaMA-7B,  RedPajama-7B,GPT3.5-Turbo, GPT4|DBpedia, TriviaQA|https://github.com/amazon-science/invite-llm-hallucinations|
|Gender bias and stereotypes in Large Language Models|Collective Intelligence Conference (CI)|2023|Not disclosed (Used four LLMs)|Own dataset created|-|
|“Kelly is a Warm Person, Joseph is a Role Model”: Gender Biases in LLM-Generated Reference Letters|EMNLP|2023|ChatGPT, Alpaca|WikiBias-Aug|https://github.com/uclanlp/biases-llm-reference-letters|
|ProPILE: Probing Privacy Leakage in Large Language Models|Neurips|2023|OPT-350M OPT-1.3B OPT- 2.7B|Pile|-|
|NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails|EMNLP System demonstrations|2023|text-davinci-003, gpt-3.5-turbo, falcon-7b-instruct, llama2-13b-chat|Anthropic Red-Teaming and Helpful datasets|https://github.com/NVIDIA/NeMo-Guardrails/|
|H2O Open Ecosystem for State-of-the-art Large Language Models|EMNLP System demonstrations|2023|Generic|-|https://github.com/h2oai/h2ogpt|


## Retrival-and-Recommendation
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|FETA: Towards Specializing Foundation Models for Expert Task Applications|NeurIPS|2022|CLIP|FETA|-|
|GENERATE RATHER THAN RETRIEVE: LARGE LANGUAGE MODELS ARE STRONG CONTEXT GENERATORS|ICLR|2023|InstructGPT|TriviaQA, WebQ|https://github.com/wyu97/GenRead|
|Query2doc: Query Expansion with Large Language Models|EMNLP|2023|Text-davinci-001, Text-davinci-003, GPT-4, babbage, curie|MS-MARCO, TREC DL 2019|-|
|Visual Captions: Augmenting Verbal Communication with On-the-fly Visuals|CHI|2023|GPT3|VC 1.5K|https://github.com/google/archat|
|Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences|RecSys|2023|PaLM |Created own dataset|-|
|Effectively Fine-tune to Improve Large Multimodal Models for Radiology Report Generation|Neurips|2023|GPT2-S (117M),  GPT2-L (774M) [29], OpenLLaMA-7B (7B)|MIMIC-CXR |https://aws.amazon.com/machine-learning/responsible-machine-learning/aws-healthscribe/|
|Building a hospitable and reliable dialogue system for android robots: ascenario-based approach with large language models|Advanced robotics|2023|Hyperclova|None (private database + jalan + trip advisor)|-|
|Can Generative LLMs Create Query Variants for Test Collections?|SIGIR|2023|text-davinci-003|UQV100|-|
|LLM-Based Aspect Augmentations for Recommendation Systems|ICML Workshop|2023|PaLM2|Created own dataset|-|

## Security
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|A Pretrained Language Model for Cyber Threat Intelligence|EMNLP Industry Track|2023|CTI-BERT|Attack description, Security Textbook, Academic Paper, Security Wiki, Threat reports, Vulnerability|-|
|Matching Pairs: Attributing Fine-Tuned Models to their Pre-Trained Large Language Models|ACL|2023|BERT, GPT, BLOOM, codegen-350M, DialoGPT, DistilGPT2, OPT, GPT-Neo, xlnet-base-cased, multilingual-miniLM-L12-v2|GitHub, The BigScience ROOTS Corpus, CC-100, Reddit, and THEPILE|-|
|Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark|ACL|2023|text-embedding-ada-002, BERT|SST2, Mind, Enron Spam, AG news|-|

## Societal-impact
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|KOSBI: A Dataset for Mitigating Social Bias Risks Towards Safer Large Language Model Applications|EMNLP Industry Track|2023|HyperCLOVA (30B and 82B), and GPT-3|KoSBi|https://github.com/naver-ai/korean-safety-benchmarks|
|DELPHI: Data for Evaluating LLMs’ Performance in Handling Controversial Issues|EMNLP Industry Track|2023|GPT-3.5-turbo-0301, Falcon 40B-instruct, Falcon 7B-insturct, Dolly-v2-12b|DELPHI|https://github.com/apple/ml-delphi|
|DisasterResponseGPT: Large Language Models for Accelerated Plan of Action Development in Disaster Response Scenarios|ICML Workshop|2023|GPT-3.5, GPT-4, Bard|Created own dataset|-|


## Miscellaneous-applications
### Cloud-management
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|Automatic Root Cause Analysis via Large Language Models for Cloud Incidents|EuroSys|2024|GPT-3.5, GPT4|653 incidents from Microsoft's Transport service to investigate RCACopilot's efficicay|-|
|LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models|EMNLP|2023|GPT-3.5-Turbo-0301 and Claude-v1.3|GSM8K, BBH, ShareGPT, Arxiv-March23|https://github.com/microsoft/LLMLingua|


### Task-planning
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|ChatGPT Empowered Long-Step Robot Control in Various Environments: A Case Application|IEEE Access|2023|ChatGPT|Defined the own prompts|https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts|

### Forecasting-analytics
|Paper|Venue|Year|LLMs names|Dataset Name|Github Link|
|:----|:----|:---|:---------|:-----------|:----------|
|Harnessing LLMs for Temporal Data - A Study on Explainable Financial Time Series Forecasting|EMNLP Industry Track|2023|GPT-4, LLaMA|Stock price data, Company profile data,   Finance/Economy News Data|-|
|Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? A Study on Several Typical Tasks|EMNLP Industry Track|2023|ChatGPT, GPT-4, BloombergGPT, GPT-NeoX, OPT66B, BLOOM176B, FinBERT|FPB/FiQA/TweetFinSent, Headline, NER, REFinD, FinQA/ConvFinQA|-|



