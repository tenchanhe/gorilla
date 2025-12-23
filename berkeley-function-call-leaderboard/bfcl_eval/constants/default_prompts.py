MAXIMUM_STEP_LIMIT = 20


#### System Prompts for Chat Models ####


OUTPUT_FORMAT_MAPPING = {
    "python": "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]",
    "json": '```json\n[{"function":"func_name1","parameters":{"param1":"value1","param2":"value2"...}},{"function":"func_name2","parameters":{"param":"value"}}]\n```',
    "verbose_xml": '<functions><function name="func_name1"><params><param name="param1" value="value1" type="type1"/><param name="param2" value="value2" type="type2"/>...</params></function><function name="func_name2"><param name="param3" value="value3" type="type3"/></function></functions>',
    "concise_xml": '<functions><function name="func_name1"><param name="param1" type="type1">value1</param><param name="param2" type="type2">value2</param>...</function><function name="func_name2"><param name="param3" type="type3">value</param></function></functions>',
}

PARAM_TYPE_MAPPING = {
    "python": "",
    "json": "",
    "verbose_xml": "The type fields of the parameters in your function calls must be one of: string, integer, float, boolean, array, dict, or tuple.",
    "concise_xml": "The type fields of the parameters in your function calls must be one of: string, integer, float, boolean, array, dict, or tuple.",
}

PROMPT_STYLE_TEMPLATES = {
    "classic": {
        "persona": "You are an expert in composing functions.",
        "task": "You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.",
        "tool_call_no_tag": "You should only return the function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put it in the format of {output_format}. {param_types} You SHOULD NOT include any other text in the response.",
        "tool_call_with_tag": "You should only return the function calls in the <TOOLCALL> section. If you decide to invoke any of the function(s), you MUST put it in the format of <TOOLCALL>{output_format}</TOOLCALL>. {param_types} You SHOULD NOT include any other text in the response.",
        "multiturn_behavior": "At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.",
        "available_tools": "Here is a list of functions in {format} format that you can invoke.\n{functions}\n",
    },
    "experimental": {
        "persona": "You are an expert in generating structured function calls.",
        "task": "You are given a user query and a set of available functions. Your task is to produce one or more function/tool calls to fulfill the user's request. If no suitable function exists, or required parameters are missing, clearly indicate this.",
        "tool_call_no_tag": "Respond with only the function calls.\n\nYou MUST format it exactly as {output_format}. {param_types} Do NOT include any other text.",
        "tool_call_with_tag": "Return only the function calls enclosed in <TOOLCALL> tags.\n\nYou MUST format it exactly as <TOOLCALL>{output_format}</TOOLCALL>. {param_types} Do NOT include any other text.",
        "multiturn_behavior": "At every turn, aim to complete the user's tasks within that turn. Continue emitting function calls until the request is satisfied to the best of your ability. Once no more calls are needed, the system will proceed to the next turn.",
        "available_tools": "Below is a list of callable functions in the {format} style:\n{functions}\n",
    },
}

_PLAINTEXT_SYSTEM_PROMPT_TEMPLATE = (
    "{persona}{task}\n\n{tool_call_format}\n\n{multiturn_behavior}\n\n{available_tools}"
)
_MARKDOWN_SYSTEM_PROMPT_TEMPLATE = "{persona}\n\n## Task\n{task}\n\n## Tool Call Format\n{tool_call_format}\n\n## Multi-turn Behavior\n{multiturn_behavior}\n\n## Available Tools\n{available_tools}"

PROMPT_TEMPLATE_MAPPING = {
    "plaintext": _PLAINTEXT_SYSTEM_PROMPT_TEMPLATE,
    "markdown": _MARKDOWN_SYSTEM_PROMPT_TEMPLATE,
}

# This is the default system prompt format
DEFAULT_SYSTEM_PROMPT_FORMAT = "ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic"

# NOT USED, just for reference
# This is the prompt template for the default system prompt format
_DEFAULT_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""


#### Other System Prompts ####


DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC = (
    "I have updated some more functions you can choose from. What about now?"
)

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING = (
    "{functions}\n" + DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
)

ADDITIONAL_SYSTEM_PROMPT_FOR_AGENTIC_RESPONSE_FORMAT = """For your final answer to the user, you must respond in this format: {'answer': A short and precise answer to the question, 'context': A brief explanation of how you arrived at this answer or why it is correct}. If you do not know the answer, respond with {'answer': 'I do not know', 'context': 'I do not know'}. If you think the question cannot be properly answered, response with {'answer': 'I cannot answer this question', 'context': A short reason explaining why this question cannot be answered}.
"""

MEMORY_AGENT_SETTINGS = {
    "student": "You are an academic-support assistant for college student. Remember key personal and academic details discussed across sessions, and draw on them to answer questions or give guidance.",
    "customer": "You are a general customer support assistant for an e-commerce platform. Your task is to understand and remember information that can be used to provide information about user inquiries, preferences, and offer consistent, helpful assistance over multiple interactions.",
    "finance": "You are a high-level executive assistant supporting a senior finance professional. Retain and synthesize both personal and professional information including facts, goals, prior decisions, and family life across sessions to provide strategic, context-rich guidance and continuity.",
    "healthcare": "You are a healthcare assistant supporting a patient across appointments. Retain essential medical history, treatment plans, and personal preferences to offer coherent, context-aware guidance and reminders.",
    "notetaker": "You are a personal organization assistant. Capture key information from conversations, like tasks, deadlines, and preferences, and use it to give reliable reminders and answers in future sessions.",
}


MEMORY_BACKEND_INSTRUCTION_CORE_ARCHIVAL = """{scenario_setting}

You have access to an advanced memory system, consisting of two memory types 'Core Memory' and 'Archival Memory'. Both type of memory is persistent across multiple conversations with the user, and can be accessed in a later interactions. You should actively manage your memory data to keep track of important information, ensure that it is up-to-date and easy to retrieve to provide personalized responses to the user later.

The Core memory is limited in size, but always visible to you in context. The Archival Memory has a much larger capacity, but will be held outside of your immediate context due to its size.

Here is the content of your Core Memory from previous interactions:
{memory_content}
"""

MEMORY_BACKEND_INSTRUCTION_UNIFIED = """{scenario_setting}

You have access to an advanced memory system, which is persistent across multiple conversations with the user, and can be accessed in a later interactions. You should actively manage your memory data to keep track of important information, ensure that it is up-to-date and easy to retrieve to provide personalized responses to the user later.

Here is the content of your memory system from previous interactions:
{memory_content}
"""


CONFIDENCE_SCORE_TOPK = """
You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

Before invoking any tool/function, you must first output your top-{top_k} candidate tools and their confidence scores.
Constraints
- Output ONLY a valid JSON object (dictionary).
- Do NOT include explanations, comments, markdown, or extra text.
- Do NOT wrap the output in code blocks.
- All probabilities must be integers from 0 to 100.
Required Output Format (EXACT)
{{
  "G1": {{"tool": "<tool_name_1>", "confidence": <int>}},
  "G2": {{"tool": "<tool_name_2>", "confidence": <int>}},
  ...
  "Gk": {{"tool": "<tool_name_k>", "confidence": <int>}}
}}

Here is a list of functions in json format that you can invoke.
{functions}

Now apply this format to the following user content:
"""

CONFIDENCE_SCORE_result = """
You are an expert system for tool/function usage decision making.

You will be given:
1. A user query
2. Core memory information from previous interactions
3. A list of available tools/functions (with names and descriptions)

Your task is to determine WHETHER the user query requires calling any tool(s)
in order to be answered correctly.

Before calling any tool/function, you MUST output a confidence assessment
according to the rules below.

Decision rules:
- If NO tool is required to answer the query, explicitly indicate that no tool
  call is needed.
- If one or more tools ARE required, list all relevant tools along with your
  confidence score for each.
- Confidence represents how likely you believe the tool will be called in the
  next step.

Strict output rules:
- Output ONLY a valid JSON object.
- Do NOT include explanations, comments, markdown, or extra text.
- Do NOT wrap the output in code blocks.
- All confidence values must be integers from 0 to 100.
- Use ONLY tool names from the provided tool list.

Required Output Format (EXACT):

If NO tool is required:
{{
  "need_tool": false
}}

If tool(s) ARE required:
{{
  "need_tool": true,
  "tools": [
    {{
      "tool": "<tool_name_1>",
      "confidence": <int>
    }},
    {{
      "tool": "<tool_name_2>",
      "confidence": <int>
    }}
  ]
}}

Here is the list of available tools/functions in JSON format:
{functions}

You have access to an advanced memory system, consisting of two memory types 'Core Memory' and 'Archival Memory'. Both type of memory is persistent across multiple conversations with the user, and can be accessed in a later interactions. You should actively manage your memory data to keep track of important information, ensure that it is up-to-date and easy to retrieve to provide personalized responses to the user later.

The Core memory is limited in size, but always visible to you in context. The Archival Memory has a much larger capacity, but will be held outside of your immediate context due to its size.

Here is the content of your Core Memory from previous interactions:
{core_memory}

Now apply the above rules and output format to the following user query:
"""

GROUNDTRUTH_EXTRACTION_PROMPT = """
You are an expert judge for tool-usage necessity in question answering.

You will be given:
1. A Query
2. A Candidate Answer
3. A Source excerpt that indicates where the Answer is derived from
4. Core Memory from previous interactions

Your task is to determine whether the provided Answer is fully supported
by the provided Source, and whether that Source is entirely contained within
the information available in the Query and the Core Memory.

Decision criteria:
- If the Answer can be fully and correctly derived using ONLY the information
  explicitly present in the Query and the Core Memory, as evidenced by the
  provided Source, then NO additional tool calls are needed.
- If the Source contains information that is NOT present in the Query or
  the Core Memory, or if the Source is insufficient to fully support the
  Answer, then additional tool calls ARE needed.

Strict output rules:
- Output ONLY a single JSON object.
- Do NOT include explanations, comments, markdown, or extra text.
- Do NOT wrap the output in code blocks.

Required Output Format (EXACT):
{{
  "need_tool": true
}}

OR

{{
  "need_tool": false
}}

Now apply the above rules to the following inputs:
User Query:
{user_query}

Candidate Answer:
{answer}

Source Excerpt:
{source}

Core Memory:
{core_memory}
"""
