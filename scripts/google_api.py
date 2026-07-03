# -*- coding: utf-8 -*-
import os
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
import google.generativeai as genai


def authenticate_gemini(api_key=None, model_name=None):
    """Authenticates the Gemini client using an API key and selects the best available model."""
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print(
            "ERROR: No Gemini API key found. Please set the GEMINI_API_KEY environment variable."
        )
        return None

    try:
        genai.configure(api_key=api_key)

        available_models = []
        try:
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    available_models.append(m.name)
            available_models = [
                m
                for m in available_models
                if not any(
                    dep in m for dep in ["-2.0-", "-1.5-", "gemini-2.0", "gemini-1.5"]
                )
            ]
        except Exception as list_err:
            print(
                f"Warning: Could not list models: {list_err}. Defaulting to fallbacks."
            )

        selected_model = model_name if model_name else "gemini-3.5-flash"

        if available_models:
            if model_name:
                matched = False
                for a_model in available_models:
                    if a_model == model_name or a_model.endswith(f"/{model_name}"):
                        selected_model = a_model
                        matched = True
                        break
                if not matched:
                    print(
                        f"Warning: Requested model '{model_name}' not found in available models. Selecting a priority fallback instead."
                    )
                    model_name = None

            if not model_name:
                priority = [
                    "gemini-3.5-flash",
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-3.1-flash-lite",
                ]
                selected_model = next(
                    (a for p in priority for a in available_models if a == p or a.endswith(f"/{p}")),
                    available_models[0],
                )

        gemini_client = genai.GenerativeModel(selected_model)
        return gemini_client
    except Exception as e:
        print(f"An unexpected error occurred during Gemini authentication: {e}")
        return None
