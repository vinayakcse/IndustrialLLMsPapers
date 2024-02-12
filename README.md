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
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[InstructPTS: Instruction-Tuning LLMs for Product Title Summarization](https://aclanthology.org/2023.emnlp-industry.63)|EMNLP  Industry Track|2023|FLAT-T5|Not mentioned explicitly|
|[LLM Based Generation of Item-Description for Recommendation System](https://dl.acm.org/doi/abs/10.1145/3604915.3610647)|RecSys|2023|Alpaca-LoRa|MovieLens, Goodreads Book graph|
|[Assess and Summarize: Improve Outage Understanding with Large Language Models](https://arxiv.org/abs/2305.18084)|ESEC/FSE|2023|GPT-3.X|historical data of 3 years cloud systems|
|[Beyond Summarization: Designing AI Support for Real-World Expository Writing Tasks](https://arxiv.org/abs/2304.02623)|CHI In2Writing Workshop|2023|-|-|
|[Building Real-World Meeting Summarization Systems using Large Language Models: A Practical Perspective](https://aclanthology.org/2023.emnlp-industry.33)|EMNLP Industry Track|2023|GPT-4, GPT3.5, PaLM-2, and LLaMA-2 13b, 7b|QMSUM, AMI, ICSI|

### Question-Answering
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[FlowMind: Automatic Workflow Generation with LLMs](https://dl.acm.org/doi/abs/10.1145/3604237.3626908)|ICAIF|2023|gpt-3.5-turbo|NCEN-QA, NCEN-QA-Easy, NCEN-QA-Intermediate, NCEN-QA-Hard|
|[PROMPTCAP: Prompt-Guided Task-Aware Image Captioning](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_PromptCap_Prompt-Guided_Image_Captioning_for_VQA_with_GPT-3_ICCV_2023_paper.html) [code](https://yushi-hu.github.io/promptcap_demo/) [code](https://huggingface.co/tifa-benchmark/promptcap-coco-vqa)|ICCV|2023|GPT-3|COCO, OK-VQA, A-OKVQA, WebQA|
|[Benchmarking Large Language Models on CMExam - A Comprehensive Chinese Medical Exam Dataset](https://arxiv.org/abs/2306.03030) [code](https://github.com/williamliujl/CMExam/tree/main)|NeurIPS|2023|GPT-3.5 turbo, GPT-4, ChatGLM, LLaMA, Vicuna, Alpaca|CMExam|
|[Empower Large Language Model to Perform Better on Industrial Domain-Specific Question Answering](https://aclanthology.org/2023.emnlp-industry.29)|EMNLP Industry Track|2023|GPT-4, GPT3.5, LLaMA-2|MSQA|


### Machine translation
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[Bootstrapping Multilingual Semantic Parsers using Large Language Models](https://aclanthology.org/2023.eacl-main.180)|EACL|2023|mT5-Large, PaLM|MTOP, MASSIVE|

### Conversational
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[Understanding the Benefits and Challenges of Deploying Conversational AI Leveraging Large Language Models for Public Health Intervention](https://dl.acm.org/doi/10.1145/3544548.3581503) [code](https://guide.ncloud-docs.com/docs/en/clovacarecall-overview)|CHI|2023|HyperCLOVA|-|
|[“The less I type, the beter”: How AI Language Models can Enhance or Impede Communication for AAC Users](https://dl.acm.org/doi/fullHtml/10.1145/3544548.3581560)|CHI|2023|-|Collected own data|
|[I wouldn’t say offensive but...: Disability-Centered Perspectives on Large Language Models](https://dl.acm.org/doi/10.1145/3593013.3593989)|FAccT|2023|LaMDA|-|

### Sentiment analysis
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis](https://dl.acm.org/doi/abs/10.1145/3543873.3587324)|WWW|2023|GPT-3, PaLM|custom reddit dataset, FiQA-News|

### Reasoning
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[MathPrompter: Mathematical Reasoning using Large Language Models](https://aclanthology.org/2023.acl-industry.4)|EMNLP Industry Track|2023|text-davinci-002, PaLM|MultiArith dataset|
|[Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models](https://arxiv.org/abs/2304.09842) [code](https://github.com/lupantech/chameleon-llm)|NeurIPS|2023|gpt-3.5-turbo , GPT-4|ScienceQA, TabMWP|
|[On the steerability of large language models toward data-drivenpersonas](https://www.amazon.science/publications/on-the-steerability-of-large-language-models-toward-data-driven-personas)|CIKM|2023|GPT-Neo-1.3B,  GPT-Neo-2.7B,  GPT-J-6B, Falcon-7B-Instruct|OpinionQA|
|[Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning](https://dl.acm.org/doi/10.1145/3539618.3591708)|SIGIR|2023|CODEX|TabFact, WikiTableQuestion, FetaQA|
|[Answering Causal Questions with Augmented LLMs](https://openreview.net/pdf?id=ikLvibXZid)|ICML Worshop|2023|GPT-3.5, GPT-4|-|

### Table-to-text generation
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[Investigating Table-to-Text Generation Capabilities of LLMs in Real-World Information Seeking Scenarios](https://aclanthology.org/2023.emnlp-industry.17)|EMNLP Industry Track|2023|GPT4, TULU, Pythia,  Alpaca, Vicuna, LLaMA-2, GPT-3.5|LOTNLG, F2WTQ|
|[Unleashing the Potential of Data Lakes with Semantic Enrichment Using Foundation Model](https://hozo.jp/ISWC2023_PD-Industry-proc/ISWC2023_paper_513.pdf)|ISWC|2023|GPT4, Llama2, FLAN-T5|-|
|[Tabular Representation, Noisy Operators, and Impacts on Table Structure Understanding Tasks in LLMs](https://arxiv.org/abs/2310.10358) [code](https://github.com/microsoft/prose)|NeurIPS|2023|GPT-3.5 (text-davinci-003 endpoint)|AirQuality, HousingData, Diabetes, Wine Testing, Iris, Titanic, and ENB2012_data|

### Data-generation
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[LayoutGPT: Compositional Visual Planning and Generation with Large Language Models](https://arxiv.org/abs/2305.15393) [code](https://github.com/weixi-feng/LayoutGPT)|NeurIPS|2023|Codex, GPT-3.5, GPT-3.5-chat and GPT-4|NSR-1K, 3D-FRONT|
|[FABRICATOR: An Open Source Toolkit for Generating Labeled Training Data with Teacher LLMs](https://aclanthology.org/2023.emnlp-demo.1) [code](https://github.com/flairNLP/fabricator)|ACL|2023|Used existing LLMs from Hugginhface, openAI, Azure, Anthropic, Cohere|IMDB, MRPC, SNLI, TREC-6, SQUAD|

## Tools-and-Frameworks
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[Automatic Linking of Judgements to UK Supreme Court Hearings](https://aclanthology.org/2023.emnlp-industry.47)|EMNLP Industry Track|2023|GPT- text-embedding-ada-002|UK National Archive|
|[LLMR: Real-time Prompting of Interactive Worlds using Large Language Models](https://arxiv.org/abs/2309.12276)|NeurIPS|2023|Language model for mixed reality (LLMR) Dall.E-2 gpt-4|Not mentioned|
|[Enabling Conversational Interaction with Mobile UI using Large Language Models](https://dl.acm.org/doi/abs/10.1145/3544548.3580895) [code](https://github.com/google-research/google-research/tree/master/llm4mobile)|CHI|2023|PaLM |PixelHelp , AndroidHowTo,  Rico, Screen2Words |
|[PromptInfuser: Bringing User Interface Mock-ups to Life with Large Language Models](https://dl.acm.org/doi/abs/10.1145/3544549.3585628)|CHI Extended Abstract|2023|-|-|
|[LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models](https://aclanthology.org/2023.acl-demo.11) [code](https://microsoft.github.io/lida/)|ACL|2023|-|Proprietory (Not mentioned)|
|[RALLE: A Framework for Developing and Evaluating Retrieval-Augmented Large Language Models](https://aclanthology.org/2023.emnlp-demo.4/) [code](https://github.com/yhoshi3/RaLLe)|EMNLP System demonstrations|2023|Llama-2 Chat (13B, 70B), WizardVicunaLM-13B, Vicuna|KILT Benchmark|
|[PROGPROMPT: Generating Situated Robot Task Plans using Large Language Models](https://link.springer.com/article/10.1007/s10514-023-10135-3) [code](https://github.com/NVlabs/progprompt-vh)|ICRA|2023|text-davinci-*, Codex, GPT3|-|
|[Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance](https://arxiv.org/abs/2310.10021) |CoRL|2023|LLaMA-13b|ALFRED|
|[Exploring the Boundaries of GPT-4 in Radiology](https://aclanthology.org/2023.emnlp-main.891)|EMNLP|2023|gpt-3.5-turbo, text-davinci-003, gpt-4-32k|MS-CXR-T, RadNLI, Chest ImaGenome, MIMIC, Open-i|

## Code-generation
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[CodePlan: Repository-level Coding using LLMs and Planning](https://arxiv.org/abs/2309.12499)|FMDM@NeurIPS|2023|GPT-4-32k|Construct the own dataset|
|[Enhancing Network Management Using Code Generated by Large Language Models](https://arxiv.org/abs/2308.06261) [code](https://github.com/microsoft/NeMoEval)|HotNet's|2023|GPT-4, GPT-3, Text-davinci-003 (a variant of GPT 3.5)  and Google Bard|Public code repositories|
|[A Static Evaluation of Code Completion by Large Language Models](https://aclanthology.org/2023.acl-industry.34)|ACL|2023|CodeGen-350M, CodeGen-2B, CodeGen-6B, CodeGen-16B, |function completion dataset|
|[Using LLMs to Customize the UI of Web Pages](https://dl.acm.org/doi/abs/10.1145/3586182.3616671)|UIST|2023| gpt3.5   Legacy (text-davinci-003),  Legacy (code-davinci-002),  Legacy (text-davinci-edit-001)|-|
|[Generative AI for Programming Education: Benchmarking ChatGPT, GPT-4, and Human Tutors](https://dl.acm.org/doi/10.1145/3568812.3603476)|ICER|2023|GPT-3.5, GPT-4|-|
|[Grace: Language Models Meet Code Edits](https://dl.acm.org/doi/10.1145/3611643.3616253)|ESEC/FSE|2023|CODEX, CODEt5|C3PO, Overwatch|
|[Large Language Model fail at completing code with potential bugs](https://arxiv.org/abs/2306.03438) [code](https://github.com/amazon-science/buggy-code-completion)|NeurIPS|2023|CODEGEN, INCODER|Buggy-HumanEval; Buggy-FixEval|
|[Multilingual evaluation of code generation models](https://www.amazon.science/publications/multi-lingual-evaluation-of-code-generation-models) [code](https://github.com/amazon-science/mxeval)|ICLR|2023|Decoder-only transformer model|MBXP, Multilingual HumanEval, MathQA-X|

## Trustworthy-AI
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[Finspector: A Human-Centered Visual Inspection Tool for Exploring and Comparing Biases among Foundation Models](https://aclanthology.org/2023.acl-demo.4) [code](https://github.com/IBM/finspector)|ACL|2023|BERT, ALBERT, RoBERTa|CrowS-Pairs|
|[INVITE: a Testbed of Automatically Generated Invalid Questions to Evaluate Large Language Models for Hallucinations](https://aclanthology.org/2023.findings-emnlp.360) [code](https://github.com/amazon-science/invite-llm-hallucinations)|EMNLP|2023|GPTNeo-2.7B, GPTJ-6B, Open-LLaMA-7B,  RedPajama-7B,GPT3.5-Turbo, GPT4|DBpedia, TriviaQA|
|[Gender bias and stereotypes in Large Language Models](https://dl.acm.org/doi/10.1145/3582269.3615599)|Collective Intelligence Conference (CI)|2023|Not disclosed (Used four LLMs)|Own dataset created|
|[“Kelly is a Warm Person, Joseph is a Role Model”: Gender Biases in LLM-Generated Reference Letters](https://aclanthology.org/2023.findings-emnlp.243) [code](https://github.com/uclanlp/biases-llm-reference-letters)|EMNLP|2023|ChatGPT, Alpaca|WikiBias-Aug|
|[ProPILE: Probing Privacy Leakage in Large Language Models](https://arxiv.org/abs/2307.01881)|Neurips|2023|OPT-350M OPT-1.3B OPT- 2.7B|Pile|
|[NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails](https://aclanthology.org/2023.emnlp-demo.40) [code](https://github.com/NVIDIA/NeMo-Guardrails/)|EMNLP System demonstrations|2023|text-davinci-003, gpt-3.5-turbo, falcon-7b-instruct, llama2-13b-chat|Anthropic Red-Teaming and Helpful datasets|
|[H2O Open Ecosystem for State-of-the-art Large Language Models](https://aclanthology.org/2023.emnlp-demo.6) [code](https://github.com/h2oai/h2ogpt)|EMNLP System demonstrations|2023|Generic|-|

## Retrival-and-Recommendation
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[FETA: Towards Specializing Foundation Models for Expert Task Applications](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c12dd3034259fc000d80db823041c187-Abstract-Datasets_and_Benchmarks.html)|NeurIPS|2022|CLIP|FETA|
|[GENERATE RATHER THAN RETRIEVE: LARGE LANGUAGE MODELS ARE STRONG CONTEXT GENERATORS](https://openreview.net/forum?id=fB0hRu9GZUS) [code](https://github.com/wyu97/GenRead)|ICLR|2023|InstructGPT|TriviaQA, WebQ|
|[Query2doc: Query Expansion with Large Language Models](https://arxiv.org/abs/2303.07678)|EMNLP|2023|Text-davinci-001, Text-davinci-003, GPT-4, babbage, curie|MS-MARCO, TREC DL 2019|
|[Visual Captions: Augmenting Verbal Communication with On-the-fly Visuals](https://dl.acm.org/doi/fullHtml/10.1145/3544548.3581566) [code](https://github.com/google/archat)|CHI|2023|GPT3|VC 1.5K|
|[Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences](https://dl.acm.org/doi/10.1145/3604915.3608845)|RecSys|2023|PaLM |Created own dataset|
|[Effectively Fine-tune to Improve Large Multimodal Models for Radiology Report Generation](https://www.amazon.science/publications/effectively-fine-tune-to-improve-large-multimodal-models-for-radiology-report-generation) [code](https://aws.amazon.com/machine-learning/responsible-machine-learning/aws-healthscribe/)|Neurips|2023|GPT2-S (117M),  GPT2-L (774M) [29], OpenLLaMA-7B (7B)|MIMIC-CXR |
|[Building a hospitable and reliable dialogue system for android robots: ascenario-based approach with large language models](https://www.tandfonline.com/doi/full/10.1080/01691864.2023.2244554)|Advanced robotics|2023|Hyperclova|None (private database + jalan + trip advisor)|
|[Can Generative LLMs Create Query Variants for Test Collections?](https://dl.acm.org/doi/abs/10.1145/3539618.3591960)|SIGIR|2023|text-davinci-003|UQV100|
|[LLM-Based Aspect Augmentations for Recommendation Systems](https://openreview.net/pdf?id=bStpLVqv1H)|ICML Workshop|2023|PaLM2|Created own dataset|

## Security
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[A Pretrained Language Model for Cyber Threat Intelligence](https://aclanthology.org/2023.emnlp-industry.12)|EMNLP Industry Track|2023|CTI-BERT|Attack description, Security Textbook, Academic Paper, Security Wiki, Threat reports, Vulnerability|
|[Matching Pairs: Attributing Fine-Tuned Models to their Pre-Trained Large Language Models](https://aclanthology.org/2023.acl-long.410)|ACL|2023|BERT, GPT, BLOOM, codegen-350M, DialoGPT, DistilGPT2, OPT, GPT-Neo, xlnet-base-cased, multilingual-miniLM-L12-v2|GitHub, The BigScience ROOTS Corpus, CC-100, Reddit, and THEPILE|
|[Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark](https://aclanthology.org/2023.acl-long.423)|ACL|2023|text-embedding-ada-002, BERT|SST2, Mind, Enron Spam, AG news|

## Societal-impact
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[KOSBI: A Dataset for Mitigating Social Bias Risks Towards Safer Large Language Model Applications](https://aclanthology.org/2023.acl-industry.21) [code](https://github.com/naver-ai/korean-safety-benchmarks)|EMNLP Industry Track|2023|HyperCLOVA (30B and 82B), and GPT-3|KoSBi|
|[DELPHI: Data for Evaluating LLMs’ Performance in Handling Controversial Issues](https://aclanthology.org/2023.emnlp-industry.76) [code](https://github.com/apple/ml-delphi)|EMNLP Industry Track|2023|GPT-3.5-turbo-0301, Falcon 40B-instruct, Falcon 7B-insturct, Dolly-v2-12b|DELPHI|
|[DisasterResponseGPT: Large Language Models for Accelerated Plan of Action Development in Disaster Response Scenarios](https://arxiv.org/abs/2306.17271)|ICML Workshop|2023|GPT-3.5, GPT-4, Bard|Created own dataset|

## Miscellaneous-applications
### Cloud-management
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[Automatic Root Cause Analysis via Large Language Models for Cloud Incidents](https://arxiv.org/abs/2305.15778)|EuroSys|2024|GPT-3.5, GPT4|653 incidents from Microsoft's Transport service to investigate RCACopilot's efficicay|
|[LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://aclanthology.org/2023.emnlp-main.825) [code](https://github.com/microsoft/LLMLingua)|EMNLP|2023|GPT-3.5-Turbo-0301 and Claude-v1.3|GSM8K, BBH, ShareGPT, Arxiv-March23|


### Task-planning
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[ChatGPT Empowered Long-Step Robot Control in Various Environments: A Case Application](https://ieeexplore.ieee.org/document/10235949) [code](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts)|IEEE Access|2023|ChatGPT|Defined the own prompts|

### Forecasting-analytics
|Paper|Venue|Year|LLMs names|Dataset Name|
|:----|:----|:---|:---------|:-----------|
|[Harnessing LLMs for Temporal Data - A Study on Explainable Financial Time Series Forecasting](https://aclanthology.org/2023.emnlp-industry.69)|EMNLP Industry Track|2023|GPT-4, LLaMA|Stock price data, Company profile data,   Finance/Economy News Data|
|[Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? A Study on Several Typical Tasks](https://aclanthology.org/2023.emnlp-industry.39)|EMNLP Industry Track|2023|ChatGPT, GPT-4, BloombergGPT, GPT-NeoX, OPT66B, BLOOM176B, FinBERT|FPB/FiQA/TweetFinSent, Headline, NER, REFinD, FinQA/ConvFinQA|



