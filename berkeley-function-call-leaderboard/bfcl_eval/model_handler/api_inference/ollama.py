# Copyright 2024- aicloudsoft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from .openai_completion import OpenAICompletionsHandler


class OllamaHandler(OpenAICompletionsHandler):
    """
    Handler for Ollama models, which are compatible with OpenAI's API.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        # We can reuse the parent's __init__
        if "ollama" in model_name.lower():
            model_name = model_name.replace("ollama/", "")
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    def _build_client_kwargs(self):
        """
        Build client kwargs for Ollama.
        It uses a hardcoded base_url or an environment variable like OLLAMA_HOST.
        """
        # Call the parent method to get any other default settings it might have
        kwargs = super()._build_client_kwargs()

        # Override or set the base_url for Ollama
        kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL")

        # API key is not required for Ollama, but the client might expect something non-empty
        kwargs["api_key"] = os.getenv("OLLAMA_API_KEY")

        return kwargs

