from __future__ import annotations

# (floor, ceiling) observed from each pipeline's real evaluation runs.
# floor: roughly where that modality's own abstention threshold sits.
# ceiling: roughly the best score seen on a clean, confident, correct match.
TEXT_SCORE_RANGE = (0.30, 0.60)
IMAGE_SCORE_RANGE = (0.22, 0.38)
AUDIO_LOGPROB_RANGE = (-1.5, -0.05) 


def _normalize(value: float, floor: float, ceiling: float) -> float:
    if ceiling == floor:
        return 0.0
    return max(0.0, min(1.0, (value - floor) / (ceiling - floor)))


def text_confidence(cosine_score: float) -> float:
    return _normalize(cosine_score, *TEXT_SCORE_RANGE)


def image_confidence(cosine_score: float) -> float:
    return _normalize(cosine_score, *IMAGE_SCORE_RANGE)


def audio_confidence(avg_logprob: float) -> float:
    return _normalize(avg_logprob, *AUDIO_LOGPROB_RANGE)


def combined_confidence(confidences: list[float]) -> float:

    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


if __name__ == "__main__":
    print("Text-only query, top_score=0.42:")
    print(f"  confidence = {combined_confidence([text_confidence(0.42)]):.3f}\n")

    print("Image-only query, top_score=0.30:")
    print(f"  confidence = {combined_confidence([image_confidence(0.30)]):.3f}\n")

    print("Voice-only query, avg_logprob=-0.20:")
    print(f"  confidence = {combined_confidence([audio_confidence(-0.20)]):.3f}\n")

    print("Combined image+text query, image=0.30, text=0.42:")
    combined = combined_confidence([image_confidence(0.30), text_confidence(0.42)])
    print(f"  confidence = {combined:.3f}\n")

    print("Combined voice+image query, voice=-0.20, image=0.22 (near its floor):")
    combined = combined_confidence([audio_confidence(-0.20), image_confidence(0.22)])
    print(f"  confidence = {combined:.3f}")