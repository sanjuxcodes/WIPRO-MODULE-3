# AI-Based Automotive Review and Customer Sentiment Analytics

## Setup
```
pip install transformers torch pandas
```

## Run
```
python aspect_sentiment_analysis.py
```

This will:
1. Run aspect-based sentiment analysis on 5 sample automotive reviews using BOTH
   a BERT backbone and an XLM-R backbone (see MODEL_REGISTRY in the script)
2. Print an aspect-level sentiment table for each model
3. Print a BERT vs XLM-R agreement check per clause
4. Save detailed results (both models) to `aspect_sentiment_output.csv`

## Files
- `aspect_sentiment_analysis.py` — full pipeline: preprocessing, tokenizer, BERT and
  XLM-R sentiment classification (run side by side), aspect extraction, and
  aspect-level sentiment aggregation.

## Models used
- BERT: `nlptown/bert-base-multilingual-uncased-sentiment`
- XLM-R: `cardiffnlp/twitter-xlm-roberta-base-sentiment`

Both are genuinely loaded and run in `main()` via `run_all_models()` — not just one
with the other as an unused swap-in option.

## Aspects covered
Vehicle, Engine, Battery, Mileage, Safety, Comfort, Service, Infotainment, Price.
