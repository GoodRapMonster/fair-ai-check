import os
import json
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from functools import lru_cache
import hashlib

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# In-memory cache for language responses
_narration_cache: Dict[str, str] = {}


def _cache_key(*args) -> str:
    return hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()


class LLMNarrator:
    """
    Gemini-powered narrative generator for bias reports.
    Produces plain-English explanations, story mode narrations,
    and multi-language summaries.
    """

    MODEL = "gemini-1.5-flash"

    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=self.MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1200,
            ),
        )

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with error handling."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Narration unavailable: {str(e)}]"

    def narrate(
        self,
        analysis_results: Dict[str, Any],
        domain: str,
        language: str = "English",
        organization_name: str = "your organization",
        role: str = "professional",
    ) -> Dict[str, Any]:
        """
        Generate a 3-paragraph bias report narration in the chosen language.
        Results are cached per unique analysis + language combination.
        """
        cache_key = _cache_key(
            analysis_results.get("analysis_id", ""),
            language,
            organization_name,
        )

        if cache_key in _narration_cache:
            return json.loads(_narration_cache[cache_key])

        bias_score = analysis_results.get("bias_score", 0.5)
        risk_level = analysis_results.get("risk_level", "HIGH")
        metrics = analysis_results.get("metrics", [])
        proxy_results = analysis_results.get("proxy_results", [])
        intersectional = analysis_results.get("intersectional_results", [])

        # Build metrics summary
        metrics_summary = "\n".join(
            f"- {m.get('name', '')}: {m.get('score', 0):.3f} (threshold {m.get('threshold', 0.8):.1f}) — {m.get('status', '')}"
            for m in metrics[:7]
        )

        top_proxies = [p for p in proxy_results if p.get("correlation", 0) > 0.5]
        proxy_summary = ", ".join(
            f"{p.get('feature', '')} ({int(p.get('correlation', 0)*100)}% correlated with {p.get('protected_attr', '')})"
            for p in top_proxies[:3]
        ) or "none detected above threshold"

        worst_intersectional = intersectional[0] if intersectional else {}

        prompt = f"""You are a fairness auditor explaining AI bias findings to a {domain} professional.
Your audience is a {role}. Write in {language}.
Use culturally appropriate examples for regions where {language} is spoken.
Always cite specific numbers. Be clear, direct, and action-oriented.

## Bias Report Data
Domain: {domain}
Organization: {organization_name}
Overall Fairness Score: {bias_score:.3f} / 1.0
Risk Level: {risk_level}

Fairness Metrics:
{metrics_summary}

Proxy Variables Detected: {proxy_summary}

Worst Intersectional Subgroup: {worst_intersectional.get('subgroup', 'N/A')} 
  - Positive rate: {worst_intersectional.get('positive_rate', 'N/A')}
  - Overall rate: {worst_intersectional.get('overall_rate', 'N/A')}

## Output Format (respond with JSON only)
{{
  "executive_headline": "<one powerful sentence for a board slide — in {language}>",
  "paragraph1": "<What bias was found and who is affected. Cite exact numbers. In {language}.>",
  "paragraph2": "<Root causes: proxies, data imbalance, historical patterns. In {language}.>",
  "paragraph3": "<Ordered action steps with expected outcomes. In {language}.>",
  "action_steps": ["<step 1>", "<step 2>", "<step 3>", "<step 4>"],
  "risk_badge": "{risk_level}"
}}"""

        raw = self._call_gemini(prompt)

        # Parse JSON response
        try:
            # Strip markdown code fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            result = json.loads(clean.strip())
        except Exception:
            result = {
                "executive_headline": f"AI system shows {risk_level} bias risk with fairness score {bias_score:.2f}.",
                "paragraph1": raw[:400] if raw else "Bias analysis complete.",
                "paragraph2": f"Key proxy variables found: {proxy_summary}.",
                "paragraph3": "Apply reweighing, then disparate impact removal, then threshold adjustment.",
                "action_steps": [
                    "Apply Reweighing to training data",
                    "Remove or transform proxy features",
                    "Adjust decision thresholds per group",
                    "Re-audit after mitigation",
                ],
                "risk_badge": risk_level,
            }

        _narration_cache[cache_key] = json.dumps(result)
        return result

    def generate_story(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
        score_a: float,
        score_b: float,
        protected_attr: str,
        domain: str = "hiring",
    ) -> str:
        """
        Write a powerful human story about two identical candidates with different outcomes.
        """
        name_a = profile_a.get("name", "Alex")
        name_b = profile_b.get("name", "Jordan")
        val_a = profile_a.get("protected_value", "Group A")
        val_b = profile_b.get("protected_value", "Group B")
        decision_a = "APPROVED" if score_a >= 0.5 else "REJECTED"
        decision_b = "APPROVED" if score_b >= 0.5 else "REJECTED"

        # Build shared feature description
        shared = {
            k: v for k, v in profile_a.items()
            if k not in ["name", "protected_value", "protected_attr"]
        }
        features_text = ", ".join(f"{k}: {v}" for k, v in list(shared.items())[:6])

        prompt = f"""Write a 4-sentence human story about two people in a {domain} scenario.

Person A: {name_a} ({protected_attr}={val_a}) — AI Score: {score_a:.2f} → {decision_a}
Person B: {name_b} ({protected_attr}={val_b}) — AI Score: {score_b:.2f} → {decision_b}

They share identical qualifications: {features_text}
The ONLY difference is {protected_attr}: {name_a}={val_a}, {name_b}={val_b}.

Rules:
- Make it human, emotional, and powerful
- Use first names throughout
- Do NOT use technical jargon
- End with: "The only difference between {name_a} and {name_b} was their {protected_attr}."
- Write as flowing prose, not bullet points
- 4 sentences maximum"""

        return self._call_gemini(prompt)

    def prescribe_fixes(
        self, analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate ordered, specific fix recommendations based on analysis results.
        """
        bias_score = analysis_results.get("bias_score", 0.5)
        metrics = analysis_results.get("metrics", [])
        proxy_results = analysis_results.get("proxy_results", [])

        failing_metrics = [m for m in metrics if m.get("status") == "FAIL"]
        high_proxies = [p for p in proxy_results if p.get("correlation", 0) > 0.6]

        prompt = f"""You are an AI fairness engineer. Generate exactly 4 ordered mitigation steps.

Analysis: Fairness score {bias_score:.2f}, {len(failing_metrics)} failing metrics.
High-risk proxy features: {[p.get('feature') for p in high_proxies[:3]]}
Failing metrics: {[m.get('name') for m in failing_metrics[:4]]}

Return a JSON array of exactly 4 strings. Each string = one concrete action step with expected outcome.
Example format: ["Step 1: Apply Reweighing — expected +15% DI improvement", ...]
Respond with only the JSON array, no other text."""

        raw = self._call_gemini(prompt)
        try:
            clean = raw.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except Exception:
            return [
                "Step 1: Apply Reweighing to balance class representation (+12% DI)",
                "Step 2: Remove top proxy features (zip_code, neighborhood) from training set",
                "Step 3: Apply Disparate Impact Remover with repair_level=0.8",
                "Step 4: Adjust decision thresholds per group to equalize TPR",
            ]

    def calculate_human_impact(
        self,
        affected_count: int,
        domain: str,
        bias_gap: float,
    ) -> Dict[str, str]:
        """
        Generate a human-scale narrative for the impact estimate.
        Includes city comparison for context.
        """
        city_comparisons = [
            (500000, "Sacramento, California"),
            (300000, "Pittsburgh, Pennsylvania"),
            (200000, "Baton Rouge, Louisiana"),
            (100000, "South Bend, Indiana"),
            (50000, "Rapid City, South Dakota"),
            (20000, "a small town"),
            (10000, "a large neighborhood"),
        ]

        city_name = "a small community"
        for threshold, city in city_comparisons:
            if affected_count >= threshold:
                city_name = city
                break

        domain_terms = {
            "hiring": "incorrectly rejected from jobs",
            "lending": "wrongly denied loans",
            "medical": "deprioritized in medical triage",
            "criminal_justice": "unfairly assessed by the justice system",
            "other": "negatively impacted by this algorithm",
        }
        impact_term = domain_terms.get(domain, "affected")

        prompt = f"""In 2 sentences, convey the human impact of AI bias.

Facts:
- {affected_count:,} people have been {impact_term}
- This equals roughly the population of {city_name}
- Domain: {domain}
- Bias gap: {bias_gap:.1%}

Be direct, human, and powerful. No technical jargon. 2 sentences only."""

        narrative = self._call_gemini(prompt)

        return {
            "narrative": narrative,
            "city_comparison": city_name,
            "impact_term": impact_term,
        }
