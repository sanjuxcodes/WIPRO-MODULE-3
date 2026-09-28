

import re
import string
from dataclasses import dataclass, field
from typing import List, Dict

import pandas as pd



ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Vehicle":      ["vehicle", "car", "suv", "sedan", "hatchback", "build quality", "design", "look"],
    "Engine":       ["engine", "power", "horsepower", "torque", "acceleration", "performance", "pickup"],
    "Battery":      ["battery", "range", "charge", "charging", "battery life", "battery range"],
    "Mileage":      ["mileage", "fuel efficiency", "fuel economy", "kmpl", "mpg", "fuel consumption"],
    "Safety":       ["safety", "airbag", "abs", "braking", "brakes", "crash", "stability", "safe"],
    "Comfort":      ["comfort", "seat", "seats", "legroom", "cabin", "suspension", "ride quality", "space"],
    "Service":      ["service", "maintenance", "dealership", "warranty", "repair", "after-sales", "support"],
    "Infotainment": ["infotainment", "display", "touchscreen", "screen", "android auto", "carplay", "audio", "speakers", "connectivity"],
    "Price":        ["price", "cost", "value for money", "expensive", "cheap", "affordable", "pricing"],
}

CLAUSE_SPLIT_PATTERN = re.compile(
    r"\b(but|however|although|though|while|whereas|on the other hand|yet)\b",
    flags=re.IGNORECASE,
)


class TextPreprocessor:
    """Cleans raw review text before it is tokenized and fed to the model."""

    def __init__(self) -> None:
        self.punct_table = str.maketrans("", "", string.punctuation.replace(".", ""))

    def clean(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"http\S+|www\.\S+", "", text)          # strip URLs
        text = re.sub(r"\s+", " ", text)                       # collapse whitespace
        text = text.replace("’", "'")
        return text

    def split_sentences(self, text: str) -> List[str]:
        text = self.clean(text)
        # simple sentence splitter (., !, ?)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]

    def split_clauses(self, sentence: str) -> List[str]:
        """Split a sentence into aspect-scoped clauses on contrast words."""
        parts = CLAUSE_SPLIT_PATTERN.split(sentence)
        # re.split with a capturing group returns the separators too; drop them
        clauses = [p.strip(" ,.") for p in parts if p and p.lower() not in
                   {"but", "however", "although", "though", "while", "whereas", "yet", "on the other hand"}]
        return clauses if clauses else [sentence]
class AspectExtractor:
    """Identifies which of the 9 automotive aspects a clause talks about."""

    def __init__(self, aspect_keywords: Dict[str, List[str]] = None) -> None:
        self.aspect_keywords = aspect_keywords or ASPECT_KEYWORDS

    def extract(self, clause: str) -> List[str]:
        clause_lower = clause.lower()
        found = [
            aspect
            for aspect, keywords in self.aspect_keywords.items()
            if any(kw in clause_lower for kw in keywords)
        ]
        return found  # a clause may legitimately mention more than one aspect


@dataclass
class SentimentResult:
    label: str          # "Positive" / "Negative" / "Neutral"
    score: float


class BertSentimentClassifier:
  

    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment") -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    @staticmethod
    def _map_label(raw_label: str) -> str:
        """Maps model-specific labels (e.g. '5 stars', 'LABEL_2') to 3 classes."""
        raw_label = raw_label.lower()
        if "star" in raw_label:
            stars = int(raw_label[0])
            if stars <= 2:
                return "Negative"
            if stars == 3:
                return "Neutral"
            return "Positive"
        if raw_label in {"positive", "label_2"}:
            return "Positive"
        if raw_label in {"negative", "label_0"}:
            return "Negative"
        return "Neutral"

    def predict(self, text: str) -> SentimentResult:
        result = self.pipe(text, truncation=True)[0]
        return SentimentResult(label=self._map_label(result["label"]), score=round(result["score"], 3))

@dataclass
class AspectSentiment:
    aspect: str
    clause: str
    sentiment: str
    confidence: float


class AspectSentimentPipeline:


    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment") -> None:
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()
        self.extractor = AspectExtractor()
        self.classifier = BertSentimentClassifier(model_name=model_name)

    def analyze_review(self, review: str) -> List[AspectSentiment]:
        results: List[AspectSentiment] = []
        for sentence in self.preprocessor.split_sentences(review):
            for clause in self.preprocessor.split_clauses(sentence):
                aspects = self.extractor.extract(clause)
                if not aspects:
                    continue
                sentiment = self.classifier.predict(clause)
                for aspect in aspects:
                    results.append(
                        AspectSentiment(
                            aspect=aspect,
                            clause=clause,
                            sentiment=sentiment.label,
                            confidence=sentiment.score,
                        )
                    )
        return results

    def analyze_batch(self, reviews: List[str]) -> pd.DataFrame:
        rows = []
        for i, review in enumerate(reviews, start=1):
            for r in self.analyze_review(review):
                rows.append(
                    {
                        "Review #": i,
                        "Review": review,
                        "Aspect": r.aspect,
                        "Clause": r.clause,
                        "Sentiment": r.sentiment,
                        "Confidence": r.confidence,
                    }
                )
        return pd.DataFrame(rows)


MODEL_REGISTRY: Dict[str, str] = {
    "BERT":  "nlptown/bert-base-multilingual-uncased-sentiment",
    "XLM-R": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
}


def run_all_models(reviews: List[str]) -> pd.DataFrame:
    """Runs the full pipeline once per model in MODEL_REGISTRY and returns
    a single DataFrame with a 'Model' column, so BERT and XLM-R outputs
    can be compared side by side on the exact same reviews/aspects."""
    frames = []
    for model_label, model_name in MODEL_REGISTRY.items():
        pipeline_ = AspectSentimentPipeline(model_name=model_name)
        df = pipeline_.analyze_batch(reviews)
        df.insert(0, "Model", model_label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
SAMPLE_REVIEWS = [
    "The battery range is excellent, but the charging time is too long.",
    "Engine performance is thrilling and the acceleration is smooth, however the mileage is disappointing.",
    "Loved the infotainment screen and connectivity, but the seats feel cramped on long drives.",
    "Great safety rating with strong airbags, though the service center support was slow.",
    "The price is a bit high for what you get, but overall the vehicle build quality is solid.",
]


def main() -> None:
    print("\nRunning pipeline with BERT and XLM-R backbones on the same reviews...\n")
    df = run_all_models(SAMPLE_REVIEWS)

    pd.set_option("display.max_colwidth", 60)
    print("=== Aspect-Level Sentiment Results (BERT vs XLM-R) ===\n")
    print(df.to_string(index=False))

    print("\n=== Aspect-wise Sentiment Summary per Model ===\n")
    summary = df.groupby(["Model", "Aspect", "Sentiment"]).size().unstack(fill_value=0)
    print(summary)

    print("\n=== Agreement Check: does BERT match XLM-R on the same clause? ===\n")
    pivot = df.pivot_table(index=["Review #", "Aspect", "Clause"], columns="Model",
                            values="Sentiment", aggfunc="first").reset_index()
    if "BERT" in pivot.columns and "XLM-R" in pivot.columns:
        pivot["Agree"] = pivot["BERT"] == pivot["XLM-R"]
        print(pivot.to_string(index=False))
        agreement_rate = pivot["Agree"].mean() * 100
        print(f"\nBERT vs XLM-R agreement rate: {agreement_rate:.1f}%")

    df.to_csv("aspect_sentiment_output.csv", index=False)
    print("\nSaved detailed results to aspect_sentiment_output.csv")


if __name__ == "__main__":
    main()
