"""Static pages — privacy policy, contact, pricing."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/pages", tags=["pages"])


@router.get("/privacy")
def privacy_policy():
    # MOCK DATA
    """Return the Sahayak AI privacy policy."""
    return {
        "title": "Sahayak AI Privacy Policy",
        "last_updated": "2026-06-01",
        "sections": [
            {
                "heading": "1. Information We Collect",
                "content": (
                    "We collect information you provide directly: account details (username, email, role), "
                    "uploaded documents, queries, and interaction data. We also collect usage analytics "
                    "such as session duration and feature usage patterns."
                ),
            },
            {
                "heading": "2. How We Use Your Information",
                "content": (
                    "Your data is used to provide and improve the Sahayak AI learning platform. "
                    "Documents are processed for RAG (Retrieval-Augmented Generation) and are not shared "
                    "with third parties. We use analytics to improve user experience."
                ),
            },
            {
                "heading": "3. Data Storage and Security",
                "content": (
                    "All data is stored securely using encryption at rest and in transit. "
                    "Passwords are hashed with bcrypt. JWT tokens have configurable expiration. "
                    "Vector embeddings are stored locally or in your configured Qdrant instance."
                ),
            },
            {
                "heading": "4. Data Retention",
                "content": (
                    "Your data is retained as long as your account is active. You may request "
                    "deletion of your account and associated data at any time by contacting support."
                ),
            },
            {
                "heading": "5. GDPR Compliance",
                "content": (
                    "We comply with GDPR regulations. You have the right to access, rectify, "
                    "or delete your personal data. Data portability is supported via the /sync/export endpoint. "
                    "Contact privacy@sahayak.ai for any data-related requests."
                ),
            },
            {
                "heading": "6. Third-Party Services",
                "content": (
                    "When configured, Sahayak AI may use Groq, OpenAI, or HuggingFace APIs for text generation. "
                    "Your queries are sent to these services only when answering questions. "
                    "We do not sell or share your data with advertisers."
                ),
            },
            {
                "heading": "7. Contact",
                "content": "For privacy concerns, contact us at privacy@sahayak.ai or via our social channels.",
            },
        ],
    }


@router.get("/contact")
def contact_info():
    # MOCK DATA
    """Return Sahayak AI contact information and social media handles."""
    return {
        "email": "hello@sahayak.ai",
        "support_email": "support@sahayak.ai",
        "social_media": {
            "twitter": "https://twitter.com/sahayak_ai",
            "linkedin": "https://linkedin.com/company/sahayak-ai",
            "github": "https://github.com/sahayak-ai",
            "instagram": "https://instagram.com/sahayak_ai",
            "discord": "https://discord.gg/sahayak-ai",
            "youtube": "https://youtube.com/@sahayak-ai",
        },
        "office": {
            "address": "Sahayak AI HQ, India",
            "phone": "+91-XXXXX-XXXXX",
        },
        "hours": "Monday - Friday, 9:00 AM - 6:00 PM IST",
    }


@router.get("/pricing")
def pricing_plans():
    # MOCK DATA
    """Return Sahayak AI pricing tiers."""
    return {
        "plans": [
            {
                "name": "Free",
                "price": 0,
                "currency": "INR",
                "period": "forever",
                "description": "Get started with essential RAG features",
                "features": [
                    "Upload up to 50 documents",
                    "Basic RAG search (5 queries/day)",
                    "Text summarization",
                    "Local vector store",
                    "Community support",
                ],
                "limitations": {
                    "documents": 50,
                    "queries_per_day": 5,
                    "voice_assistant": False,
                    "ai_counselor": False,
                    "priority_support": False,
                },
            },
            {
                "name": "Pro",
                "price": 999,
                "currency": "INR",
                "period": "month",
                "description": "Full-featured learning platform with AI tools",
                "features": [
                    "Unlimited document uploads",
                    "Unlimited RAG queries",
                    "Voice assistant & voice interaction",
                    "AI counselor access",
                    "Knowledge graph visualization",
                    "Progress tracking & analytics",
                    "NCERT book library",
                    "Priority email support",
                    "Cloud sync & backup",
                ],
                "limitations": {
                    "documents": -1,
                    "queries_per_day": -1,
                    "voice_assistant": True,
                    "ai_counselor": True,
                    "priority_support": True,
                },
            },
            {
                "name": "Enterprise",
                "price": None,
                "currency": "INR",
                "period": "custom",
                "description": "Custom deployment for institutions and organizations",
                "features": [
                    "Everything in Pro",
                    "Custom model fine-tuning",
                    "Dedicated infrastructure",
                    "Admin dashboard & analytics",
                    "SSO / LDAP integration",
                    "SLA guarantee (99.9%)",
                    "Dedicated account manager",
                    "On-premise deployment option",
                    "Custom API rate limits",
                ],
                "limitations": {
                    "documents": -1,
                    "queries_per_day": -1,
                    "voice_assistant": True,
                    "ai_counselor": True,
                    "priority_support": True,
                    "custom_models": True,
                },
            },
        ],
    }
