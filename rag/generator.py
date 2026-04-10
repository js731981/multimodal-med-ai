from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

from rag.retriever import MedicalRetriever, RetrievedPassage


class MedicalExplanationGenerator(ABC):
    """Pluggable explanation backend (template now, LLM later)."""

    @abstractmethod
    def generate(
        self,
        *,
        prediction: str,
        symptoms: str | Sequence[str] | None,
        retrieved: Sequence[RetrievedPassage],
    ) -> str:
        ...


def _format_symptoms(symptoms: str | Sequence[str] | None) -> str:
    if symptoms is None:
        return "Not specified."
    if isinstance(symptoms, str):
        return symptoms.strip() or "Not specified."
    parts = [s.strip() for s in symptoms if s and str(s).strip()]
    return ", ".join(parts) if parts else "Not specified."


def _symptom_text_for_matching(symptoms: str | Sequence[str] | None) -> str:
    raw = _format_symptoms(symptoms)
    if raw == "Not specified.":
        return ""
    return raw.lower()


def _prediction_is_normal(prediction: str) -> bool:
    p = prediction.strip().lower()
    return p == "normal" or p.startswith("normal ")


def _symptoms_strongly_suggest_pneumonia(symptoms_lower: str) -> bool:
    """Heuristic: fever+cough-type patterns or explicit lower-respiratory / pneumonia cues."""
    if not symptoms_lower.strip():
        return False
    t = symptoms_lower
    for phrase in (
        "pneumonia",
        "consolidat",
        "lobar",
        "pleuritic",
        "hypox",
        "hypoxia",
        "spo2",
        "low oxygen",
        "crackles",
        "rales",
        "purulent",
        "productive cough",
        "rusty sputum",
        "infiltrat",
    ):
        if phrase in t:
            return True
    has_fever = any(x in t for x in ("fever", "febrile", "pyrexia", "temperature"))
    has_cough = "cough" in t
    has_dyspnea = any(
        x in t for x in ("dyspnea", "dyspnoea", "shortness of breath", "short of breath", "tachypnea", "labored breathing")
    )
    has_chills = any(x in t for x in ("chills", "rigor", "rigors", "shaking chill"))
    if has_fever and has_cough:
        return True
    if has_fever and has_dyspnea:
        return True
    if has_chills and has_cough:
        return True
    return False


def _symptom_imaging_mismatch_pneumonia_vs_normal(
    prediction: str,
    symptoms: str | Sequence[str] | None,
) -> bool:
    return _prediction_is_normal(prediction) and _symptoms_strongly_suggest_pneumonia(
        _symptom_text_for_matching(symptoms)
    )


def _alignment_uncertainty_block(symptoms_display: str) -> str:
    return (
        f"Although symptoms suggest an illness that can be consistent with pneumonia "
        f"(given: {symptoms_display}), the image analysis from this model did not strongly support "
        "acute pneumonia on the provided study. Early or atypical infection, dehydration with tachypnea, "
        "or illness without yet-visible infiltrates can explain this gap.\n\n"
        "Clinical correlation recommended."
    )


def _recommended_next_steps_block() -> str:
    """General, non-diagnostic follow-up language for end users."""
    return (
        "Recommended Next Steps\n\n"
        "- Consider further imaging if symptoms persist or your clinician advises follow-up.\n"
        "- Consult a physician if symptoms worsen or you have new or concerning symptoms.\n"
        "- These suggestions are general wellness guidance only; they do not diagnose or direct treatment."
    )


class TemplateMedicalExplanationGenerator(MedicalExplanationGenerator):
    """Rule-based explanation from prediction, symptoms, and retrieved context."""

    def generate(
        self,
        *,
        prediction: str,
        symptoms: str | Sequence[str] | None,
        retrieved: Sequence[RetrievedPassage],
    ) -> str:
        sym = _format_symptoms(symptoms)
        mismatch = _symptom_imaging_mismatch_pneumonia_vs_normal(prediction, symptoms)
        if retrieved:
            ctx_blocks = []
            for r in retrieved:
                ctx_blocks.append(f"[{r.document_id}] (relevance {r.score:.2f}): {r.text}")
            context_section = "\n".join(ctx_blocks)
        else:
            context_section = "I did not find closely matching reference passages for this label."

        if mismatch:
            body_paras = [
                "The fused model's leading read here is normal, yet the clinical text includes features that often "
                "raise pneumonia in the differential. That pattern is familiar: early pneumonia, viral bronchitis, "
                "or unrelated systemic illness can present with cough and fever before infiltrates are obvious—or "
                "when the study does not capture the affected region well.",
                _alignment_uncertainty_block(sym),
            ]
            if retrieved:
                body_paras.append(
                    "Use the retrieved notes below to frame typical pneumonia presentations and mimics; they support "
                    "teaching points, not proof of what this patient has.",
                )
            else:
                body_paras.append(
                    "Retrieved reference material was limited for this label; if the clinical picture stays "
                    "concerning, usual care pathways include reassessment, vitals, and follow-up imaging or testing "
                    "as appropriate.",
                )
        else:
            body_paras = [
                f"Given the symptoms described ({sym}), the multimodal model's leading label is {prediction}. "
                "Treat that as one line of evidence: symptoms set pre-test probability, while imaging (when used in "
                "the model) adjusts how much weight you give to parenchymal disease versus other explanations.",
                "When imaging contributes to the prediction, ask whether timing, technique, and clinical course fit; "
                "models summarize patterns they were trained on and will miss nuances your exam and trajectory provide.",
            ]
        if retrieved:
            body_paras.append(
                "The passages below are the closest material from our reference library; use them to anchor "
                "terminology and typical presentations, not as proof that this is your patient's diagnosis.\n"
                f"{context_section}"
            )
        else:
            body_paras.append(context_section)

        if not mismatch:
            body_paras.append("Clinical correlation recommended.")

        body_paras.append(_recommended_next_steps_block())
        body_paras.append(
            "Disclaimer: This text is for education and context only. It is not medical advice or a diagnosis. "
            "Seek urgent or emergency care for severe shortness of breath, chest pain, confusion, fainting, "
            "or rapidly worsening symptoms."
        )
        return "\n\n".join(body_paras)


class LLMMedicalExplanationGenerator(MedicalExplanationGenerator):
    """LLM-backed generator: inject any callable that accepts a prompt and returns text."""

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        system_preamble: str | None = None,
    ) -> None:
        self._complete = complete
        self._system_preamble = system_preamble or (
            "You write clear clinical reasoning as a physician might when teaching or summarizing for a colleague: "
            "natural prose, not bullet templates or robotic headings. Open with the clinical story when appropriate "
            "(e.g. \"Based on the symptoms provided...\"). When imaging is discussed in the retrieved text, you may use "
            "phrasing like \"The imaging features suggest...\" only if that is grounded in the passages. "
            "If the clinical notes sound pneumonia-like but the model label is normal, you must reconcile that tension: "
            "use an explicit uncertainty sentence in the form: although symptoms suggest a process consistent with "
            "pneumonia, the image analysis from this assessment did not strongly support acute pneumonia—and briefly "
            "name plausible reasons (e.g. early disease, atypical infection, unrelated viral illness). "
            "In that situation, include the exact sentence \"Clinical correlation recommended.\" on its own or as a clear closing line. "
            "Stay medically accurate and conservative; if evidence is thin, say what is uncertain. Do not invent "
            "studies, vitals, or findings. End with a one- or two-sentence disclaimer: educational only, not a "
            "diagnosis or substitute for in-person care, and mention urgent red flags (e.g. severe dyspnea, chest pain, "
            "altered mental status). After the main prose, add a section with the exact heading "
            "\"Recommended Next Steps\" on its own line, followed by 2–3 short bullet lines in the spirit of: "
            "consider further imaging if symptoms persist (when clinically appropriate); consult a physician if "
            "symptoms worsen; keep language general, non-diagnostic, and avoid naming a specific diagnosis or "
            "prescribing treatment."
        )

    def generate(
        self,
        *,
        prediction: str,
        symptoms: str | Sequence[str] | None,
        retrieved: Sequence[RetrievedPassage],
    ) -> str:
        sym = _format_symptoms(symptoms)
        ctx = "\n\n".join(
            f"Passage ({r.document_id}, score={r.score:.3f}):\n{r.text}" for r in retrieved
        ) or "(no passages retrieved)"
        mismatch = _symptom_imaging_mismatch_pneumonia_vs_normal(prediction, symptoms)
        mismatch_directive = ""
        if mismatch:
            mismatch_directive = (
                "\n\nCLINICAL–IMAGING ALIGNMENT: Symptoms are pneumonia-salient but the model label is normal. "
                "Address this head-on in natural prose (not as a labeled section): include wording equivalent to "
                "\"Although symptoms suggest...\" and that \"the image analysis did not strongly support\" pneumonia on "
                "this read, then mention 1–2 brief differentials for the discordance. You must include the exact phrase "
                "\"Clinical correlation recommended.\" "
            )
        user_block = (
            f"Model's leading label: {prediction}\n"
            f"Symptoms / clinical notes: {sym}\n\n"
            f"Retrieved context:\n{ctx}\n\n"
            "Write a short explanation (a few paragraphs) that reads like thoughtful clinical reasoning—connect symptoms "
            "to the predicted label when they align, or explain thoughtfully when they diverge; weave in retrieved facts "
            "smoothly, and avoid numbered lists or stiff section titles except you must include the final "
            "\"Recommended Next Steps\" section as specified in the system instructions."
            f"{mismatch_directive}"
        )
        prompt = f"{self._system_preamble}\n\n---\n\n{user_block}"
        return self._complete(prompt).strip()


class MedicalRAGService:
    """Wires retriever + generator; returns API-shaped output."""

    def __init__(
        self,
        retriever: MedicalRetriever,
        generator: MedicalExplanationGenerator,
        *,
        default_top_k: int = 3,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._default_top_k = default_top_k

    def explain(
        self,
        *,
        prediction: str,
        symptoms: str | Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, str]:
        k = self._default_top_k if top_k is None else top_k
        retrieved = self._retriever.retrieve(prediction, top_k=k)
        text = self._generator.generate(
            prediction=prediction,
            symptoms=symptoms,
            retrieved=retrieved,
        )
        return {"explanation": text}
