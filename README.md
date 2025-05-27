# Which Cultural Lens Do Models Adopt? Unmasking Cultural Positioning Bias in Large Language Model-Generated Interview Scripts

This repository contains the code for the paper *Which Cultural Lens Do Models Adopt? Unmasking Cultural Positioning Bias in Large Language Model-Generated Interview Scripts*. It provides tools to evaluate cultural positioning bias in LLMs through interview script generation and to mitigate these biases using Fairness Intervention Pillars (FIP).

## Generating Scripts 
We provide the code for generating the **CultureLens** dataset. ```constants.py``` includes prompt templates and descriptors specifying countries and cultures, culture-indicative names, occupations, gender, and age. 

To generate dataset, add in your OpenAI account configuration in ```generation_util.py``` and run:
```
sh ./scripts/run_generate_docs.sh
```
Alternatively, the folder ```generated_scripts``` includes the complete dataset that you can directly use for evaluation. 

## Evaluating Scripts 
To evaluate cultural positioning on the generated interview scripts, run 
```
sh ./scripts/run_evaluate_outputs.sh
``` 

## Result Analysis
For comprehensive qualitative and quantitative analysis of the evaluated outputs including Cultural Externality Percentage, log-odds ratio analysis of culturally salient words and thematic analysis via topic modeling, run the analysis script:
```
sh ./scripts/run_quantitative_analysis.py 
``` 
