# Experiencers, Stimuli, or Targets: Which Semantic Roles Enable Machine Learning to Infer the Emotions?

## Extract the data
- Extract data via https://github.com/sarnthil/extract-emotion-data/
- `mv extract-emotion-data/workdata/splitted.json emotion-classification-roles/sources/`

# Reference

```
@inproceedings{oberlander-etal-2020-experiencers,
    title = "Experiencers, Stimuli, or Targets: Which Semantic Roles Enable Machine Learning to Infer the Emotions?",
    author = {Oberl{\"a}nder, Laura Ana Maria  and
      Reich, Kevin  and
      Klinger, Roman},
    booktitle = "Proceedings of the Third Workshop on Computational Modeling of People's Opinions, Personality, and Emotion's in Social Media",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.peoples-1.12",
    pages = "119--128",
    abstract = "Emotion recognition is predominantly formulated as text classification in which textual units are assigned to an emotion from a predefined inventory (e.g., fear, joy, anger, disgust, sadness, surprise, trust, anticipation). More recently, semantic role labeling approaches have been developed to extract structures from the text to answer questions like: {``}who is described to feel the emotion?{''} (experiencer), {``}what causes this emotion?{''} (stimulus), and at which entity is it directed?{''} (target). Though it has been shown that jointly modeling stimulus and emotion category prediction is beneficial for both subtasks, it remains unclear which of these semantic roles enables a classifier to infer the emotion. Is it the experiencer, because the identity of a person is biased towards a particular emotion (X is always happy)? Is it a particular target (everybody loves X) or a stimulus (doing X makes everybody sad)? We answer these questions by training emotion classification models on five available datasets annotated with at least one semantic role by masking the fillers of these roles in the text in a controlled manner and find that across multiple corpora, stimuli and targets carry emotion information, while the experiencer might be considered a confounder. Further, we analyze if informing the model about the position of the role improves the classification decision. Particularly on literature corpora we find that the role information improves the emotion classification.",
}
```
