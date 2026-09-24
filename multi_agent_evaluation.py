""" 
This script implements an evaluation using a multi-agent system. It allows multiple personas to 
evaluate a set of items based on defined criteria, generating structured responses that include 
ratings, justifications, and rankings.
"""
import os
import json
import operator
import re
import asyncio
from typing import TypedDict, Annotated, List

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

class Vote(TypedDict):
    """ 
    Represents a single vote from a persona.
    """
    df: pd.DataFrame
    justification: str
    persona: str

class State(TypedDict):
    """ 
    Represents the state of the evaluation process, containing a list of votes from different 
    personas. 
    """
    votes: Annotated[List[Vote], operator.add]

class Evaluator:
    """
    This class orchestrates the multi-agent evaluation process. It manages the prompt generation,
    LLM calls, response parsing, and concurrency control to ensure that the evaluation is performed 
    efficiently and within the rate limits of the LLM provider.
    """
    def __init__(self, llm, prompt_variables,
                 max_concurrency: int = 1, delay_between_calls: float = 12.0):
        self.llm = llm
        self.prompt_variables = prompt_variables
        # restricts how many LLM calls can happen simultaneously
        self.semaphore = asyncio.Semaphore(max_concurrency)
        # delay (12s * 5 requests = 60s; to remain under 5 RPM)
        self.delay_between_calls = delay_between_calls

    @staticmethod
    def clean_json_string(text: str) -> str:
        """
        Cleans the markdowns around a json response.
        """
        cleaned = re.sub(r"```(?:json)?", "", text)
        return cleaned.replace("```", "").strip()

    def make_prompt(self, persona_description):
        """
        Prompt function which takes a persona description and applies the template and the template 
        variables previously set.
        """
        return f"""You are a {self.prompt_variables["persona_role"]} with the following worldview:

    {persona_description}

    {self.prompt_variables["instruction"]}

    Evaluate each {self.prompt_variables["evaluation_subject"]} based on the following {len(self.prompt_variables["criteria"])} criteria, scoring from 1 (low) to 5 (high):

    {"".join(f"{key}: {value}{chr(10)}" for key, value in self.prompt_variables["criteria"].items())}
    Here are the {self.prompt_variables["evaluation_subject"]}s to evaluate:
    {chr(10).join('- ' + item for item in self.prompt_variables["items"])}

    Please respond ONLY in the following strict JSON format:

    ```json
    {{
    "ratings": [
        {{
        "item": "the corresponding {self.prompt_variables["evaluation_subject"]} name here, following the ordering in the given list"{"".join(f',{chr(10)}      "{criteria}": int' for criteria in self.prompt_variables["criteria"])}
        }},
        // ...More {self.prompt_variables["evaluation_subject"]} evaluations here
    ],
    "justification": "Your paragraph explaining the ratings here.",
    "ranking": ["{self.prompt_variables["evaluation_subject"]}1", "{self.prompt_variables["evaluation_subject"]}2", ..., "{self.prompt_variables["evaluation_subject"]}{len(self.prompt_variables["items"])}"]
    }}
    ```

    - The ratings list must include all {len(self.prompt_variables["items"])} {self.prompt_variables["evaluation_subject"]}s.
    - The ranking list must be in your personal order (1st to {len(self.prompt_variables["items"])}th).
    - Do not include any commentary outside the JSON block.
    """

    def parse_json_response(self, response):
        """
        Parses the JSON response from the LLM, extracting ratings, justification, and ranking.
        """
        response_cleaned = self.clean_json_string(response)
        data = json.loads(response_cleaned)
        ratings = data["ratings"]
        justification = data["justification"]
        ranking = data["ranking"]
        ranking_column = []
        for i, item in enumerate(ranking):
            ranking_column += [{"item": item, "rank": i + 1}]
        df = pd.DataFrame(ratings)
        df = pd.merge(df, pd.DataFrame(ranking_column), on="item", how="left")
        df.columns = \
            [self.prompt_variables["evaluation_subject"][0].upper() + \
                self.prompt_variables["evaluation_subject"][1:]] + \
                    list(self.prompt_variables["criteria"].keys()) + ["Rank"]
        return df, justification

    @retry(retry=retry_if_exception_type((ResourceExhausted, Exception)),
           stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=15, max=60))
    async def get_llm_response(self, prompt):
        """ 
        Gets LLM response.
        """
        async with self.semaphore:
            response = await self.llm.ainvoke(prompt)
            # sleep briefly after the API call to adhere by a 5 RPM rate limit
            await asyncio.sleep(self.delay_between_calls)
            return self.parse_json_response(response.content)

    # Agent node
    def make_agent_node(self, persona_key, pbar):
        """
        Creates an asynchronous node for a given persona. Each node generates a prompt, calls the 
        LLM, and parses the response into structured data, which is then stored in the state.
        """
        async def node(state):
            persona = self.prompt_variables["personas"][persona_key]
            prompt = self.make_prompt(persona)
            df, justification = await self.get_llm_response(prompt)

            state['votes'] = [{
                    "df": df,
                    "justification": justification,
                    "persona": persona_key
            }]

            pbar.update(1)
            return state
        return node

    async def main(self):
        """ 
        Performs the API calls, parses the responses, and stores them. The output displays the 
        scores of items under each criterion by persona (higher is better). The last column named 
        rank is the ordering of items given by the persona (1 is the best).        
        """

        print("Example prompt:")
        print(self.make_prompt(list(self.prompt_variables["personas"].values())[0]))

        print("Starting evaluation...")

        # initialize progress bar
        try:
            pbar.close()
        except NameError:
            pass
        pbar = \
            tqdm(f"Evaluating {self.prompt_variables["evaluation_subject"]}s with personas",
                total=len(self.prompt_variables["personas"]), unit="persona")

        # Here is where the work happens is defined. Each node runs asynchronously and
        # independently. Inside the node:
        # 1. The persona's prompt is generated.
        # 2. LangChain (llm.ainvoke) is called to get the raw model output.
        # 3. The raw JSON is parsed into structured data.
        agent_keys = list(self.prompt_variables["personas"].keys())
        graph = StateGraph(State)
        for agent in agent_keys:
            graph.add_node(agent, self.make_agent_node(agent, pbar))

        # Here LangGraph defines the workflow orchestration. It fans out the execution to
        # one distinct node for every persona defined in your dictionary.
        for agent in agent_keys:
            graph.add_edge(START, agent)
        graph.add_edge([agent for agent in agent_keys], END)

        # Orchestration begins here when 'compiled.ainvoke' is called with an empty list.
        # LangGraph takes the initial state and triggers multiple nodes simultaneously as
        # previously defined.
        compiled = graph.compile()
        results = await compiled.ainvoke({"votes": []})
        votes = results['votes']

        return votes

if __name__ == '__main__':

    # --- environment variables ---
    # load local environment variables (GOOGLE_API_KEY should be set either in .env file)
    load_dotenv()
    # support GEMINI_API_KEY if GOOGLE_API_KEY isn't directly defined
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    # --- LLM config ---
    # here you can change the model, tweak its parameters, or even use a different LLM provider
    example_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)

    # -- prompt setup ---
    # evaluation subject setup
    EVALUATION_SUBJECT = ("acid pump (operating with 15 percent hydrochloric acid "
                        "at 75 degrees Celsius and 800 kilopascals) casing material")
    # items to evaluate setup
    ITEMS = ['Hastelloy B-3',
            'Hastelloy C-276',
            'Titanium Grade 7', 
            'Titanium Grade 12',
            'Zirconium 702']
    # persona definitions setup
    # each persona generates a call to the LLM, therefore must be aware of LLM provider rate limits
    PERSONAS = {
        "corrosion engineer": "You are a corrosion engineer. You are a specialist who analyzes the chemical interactions between chemicals and surfaces to predict uniform corrosion, pitting, and stress cracking rates. You utilize material phase diagrams and laboratory data to ensure the selected alloy will not catastrophically degrade under reducing conditions.",
        "materials selection specialist": "You are a materials selection specialist. You are an expert who evaluates the mechanical properties, manufacturing history, and long-term performance data of alloys. You balance the chemical vulnerabilities of the materials against the structural demands of the equipment to choose the most resilient option.",
        "metallurgical engineer": "You are a metallurgical engineer. You are a professional who examines the internal microstructure, grain boundary stability, and heat-treatment requirements of specialized alloys. You ensure that casting, welding, and machining of equipment will not inadvertently create weak points prone to localized chemical attack.",
        "chemical process engineer": "You are a chemical process engineer. You are an engineer who defines the exact operational boundaries of equipment, including precise chemical concentrations, temperature fluctuations, and flow velocities. You identify whether dissolved iron or oxidizers are building up in the fluid stream, which heavily influences the final material choice.",
        "pump application engineer": "You are a pump application engineer. You are a manufacturing expert who translates the system's hydraulic requirements into physical pump dimensions, ensuring the casing can handle the high-velocity discharge pressures. You collaborate with metallurgists to verify that the chosen alloy can be successfully cast into complex volute geometries without internal defects."
    }
    # criteria for evaluation setup
    CRITERIA = {
        "Chemical Compatibility": "An alloy must fundamentally resist rapid uniform thinning and localized pitting when exposed to hot, reducing chemical.",
        "Oxidation Resistance": "A selected material must tolerate fluid impurities, such as dissolved iron or ferric chlorides, without suffering accelerated galvanic or chemical breakdown.",
        "Erosion-Corrosion Resistance": "Equipment must withstand high-velocity fluid friction and turbulence without losing its protective surface passivation layer.",
        "Mechanical Strength": "A selected material must retain high yield strength and structural integrity at high temperature to handle internal pressure and piping stresses.",
        "Weldability and Fabrication": " An alloy must be easily weldable and castable into complex equipment geometries without developing micro-cracks or hazardous element segregation.",
        "Lifecycle Cost-Effectiveness": "High initial procurement cost of premium alloys must be balanced against expected lifespan and reduced maintenance downtime."
    }
    # instructions and background information for the personas
    PERSONA_ROLE = "material evaluator"
    INSTRUCTION = f"You have been asked to evaluate the suitability of {len(ITEMS)} acid pump casing materials."
    # assign prompt variables to dictionary
    example_prompt_variables = {
        "evaluation_subject": EVALUATION_SUBJECT,
        "items": ITEMS,
        "personas": PERSONAS,        
        "criteria": CRITERIA,
        "persona_role": PERSONA_ROLE,
        "instruction": INSTRUCTION
    }

    # --- obtain evaluation ---
    example_evaluator = Evaluator(example_llm, example_prompt_variables)
    example_results = asyncio.run(example_evaluator.main())

    # --- display results ---
    for persona_results in example_results:
        print(f"\n ========= Evaluation by persona: {persona_results['persona']} =========")
        print(persona_results['df'])
        print(persona_results['justification'])
