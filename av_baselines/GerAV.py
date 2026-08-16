import os
from pydoc import text
import numpy as np
from openai import OpenAI
import toml
from tqdm import tqdm
import transformers
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

class GerAV:
    def __init__(self, 
                 base_path = ".",
                 tune_dataset = "twitter",
                 tuned_model = "gemma-3-12b-it",
                 baseline = False,
                 seed = 42,
                 manual_checkpoint_path = None,
                 debug = False, 
                 custom_threshold = 0,
                 gpt_mode = False,
                 use_lip = False,
                 use_lip_ger = False,
                 template = None,
                 stop_tokens = None,
                 positive_token = None,
                 negative_token = None,
                 api_key = None
                 ):
        if tuned_model == "gpt-5":
            config_path = os.path.join(base_path, f"gemma-3-12b-it.toml") # To have some configuration available
        else:
            config_path = os.path.join(base_path, f"{tuned_model}.toml")

        self.config = toml.load(open(config_path, 'r'))

        self.custom_threshold = custom_threshold

        self.gpt_mode = gpt_mode
        if self.gpt_mode:
            if api_key == None:
                raise Exception("Warning, no API key defined")
            self.client = OpenAI(
                api_key=api_key
            )

        if baseline:
            if use_lip:
                self.template = "For a scientific experiment, given two texts determine if they are written by the same author. Analyze the writing styles of the input texts, disregarding the differences in topic and content. Focus on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. \nText A: {text_a}\nText B: {text_b}\nPlease think step-by-step, then answer with 'Yes' or 'No' as your last word."
            elif use_lip_ger:
                self.template = "Für ein wissenschatliches Experiment möchten wir feststellen, ob zwei Texte von demselben Autor verfasst wurden. Analysiere die Schreibstile der Eingabetexte und ignoriere dabei Unterschiede im Thema und Inhalt. Konzentriere dich auf linguistische Merkmale wie Phonologie, Morphologie, Wortbildung, Syntax, Wortstellung, Kasussystem, Genus, Tempus, Modus, Passiv, Kongruenz, Valenz, Artikelgebrauch, Pronomen, Negation, Modalverben, Semantik, Pragmatik, Informationsstruktur, Prosodie, Wortarten, Nebensätze, Konjunktiv, Modalpartikeln, seltene Wörter und Idiomatik. \nText A: {text_a}\nText B: {text_b}\nBitte denke Schritt für Schritt nach und antworte am Ende mit 'Ja' oder 'Nein' als letztem Wort."
            else:
                self.template = "Are the following two texts written by the same author?\nText A: {text_a}\nText B: {text_b}\nPlease answer with 'Yes' or 'No'."
        else:
            self.template = self.config["prompt_template"]

        if template is not None:
            self.template = template

        if manual_checkpoint_path is not None:
            self.model_dir = manual_checkpoint_path
        else:
            self.model_dir = os.path.join(self.config["output_dir"][2:], "final_models", f"seed:{seed}", "best_model")

        self.baseline = baseline

        if not stop_tokens:
            self.stop_tokens = [" yes", " no", " Yes", " No", "YES", "NO"]
        else:
            self.stop_tokens = stop_tokens

        if not positive_token:
            self.positive_token = "yes"
        else:
            self.positive_token = positive_token

        if not negative_token:
            self.negative_token = "no"
        else:
            self.negative_token = negative_token



        if self.gpt_mode:
            pass

        elif baseline:
            self.llm = LLM(
                            model=self.config["model_name"],
                            tokenizer=self.config["model_name"],
                        )
            self.tokenizer = self.llm.get_tokenizer()   
            self.max_len = self.llm.llm_engine.model_config.max_model_len
            self.debug = debug

            self.sampling_params = SamplingParams(
                    temperature=0,
                    top_p=1.0,
                    top_k=0,
                    max_tokens=2048,
                    logprobs=10,
                    stop=self.stop_tokens,
            )

        else:
            self.llm = LLM(
                            model=self.config["model_name"],
                            tokenizer=self.config["model_name"],
                            enable_lora=True,
                            max_lora_rank=128
                        )
            
            self.tokenizer = self.llm.get_tokenizer()   
            self.max_len = self.llm.llm_engine.model_config.max_model_len
            self.debug = debug

            self.sampling_params = SamplingParams(
                    temperature=0,
                    top_p=1.0,
                    top_k=0,
                    max_tokens=2048,
                    logprobs=10,
                    stop=self.stop_tokens,
            )

    def truncate_messages(self, i1, i2):
        tokens_1 = [self.tokenizer.encode(i) for i in i1]
        tokens_2 = [self.tokenizer.encode(i) for i in i2]
        if len(tokens_1) <= 10 or len(tokens_2) <= 10:
            return i1, i2

        for i in range(len(i1)):
            if len(tokens_1[i]) > self.max_len // 2:
                tokens_1[i] = tokens_1[i][:self.max_len // 2]
            if len(tokens_2[i]) > self.max_len // 2:
                tokens_2[i] = tokens_2[i][:self.max_len // 2]
            
        i1_trunc = [self.tokenizer.decode(t) for t in tokens_1]
        i2_trunc = [self.tokenizer.decode(t) for t in tokens_2]

        return i1_trunc, i2_trunc
    
    def __call__(self, i1, i2):
        # if i1 or i2 are numpy arrays, convert to lists
        if isinstance(i1, np.ndarray):
            i1 = i1.tolist()
        if isinstance(i2, np.ndarray):
            i2 = i2.tolist()

        if len(i1) != len(i2):
            if len(i1) == 1:
                i1 = i1 * len(i2)
            elif len(i2) == 1:
                i2 = i2 * len(i1)
            else:
                raise ValueError("Input lists must be of the same length or one of them must be of length 1.")

        print("Current input:", i1, i2) 
        if self.gpt_mode:
            inputs = [self.template.format(text_a=msg1, text_b=msg2) for msg1, msg2 in zip(i1, i2)]  
            out = []
            for idx, msg in tqdm(enumerate(inputs)):
                response = self.client.responses.create(
                    model="gpt-5",
                    input=msg,
                )
                print(f"Model Response: {response}")

                # Make a new folder called "gpt5_responses" in the current directory and save the response there with a unique name
                os.makedirs("gpt5_responses", exist_ok=True)
                with open(os.path.join("gpt5_responses", f"response_{idx}.txt"), "w") as f:
                    f.write(f"Input: {msg}\nResponse: {response}\n")

                out.append(response.output_text.strip())
            predictions = [self.extract_label(o) for o in out]
            return predictions, [1.0 if p else 0.0 for p in predictions]

        else:
            # Truncate inputs if too long
            i1_t, i2_t = self.truncate_messages(i1, i2)
            messages = [self.message_packer(msg1, msg2) for msg1, msg2 in zip(i1_t, i2_t)]

            if self.baseline:
                out = self.llm.chat([m["messages"] for m in messages], self.sampling_params)
            else:
                out = self.llm.chat([m["messages"] for m in messages], self.sampling_params, lora_request=LoRARequest(self.config["model_name"],122342, self.model_dir))

            generated_text = [t.outputs[0].text.strip() for t in out]

            probs = []
            for t in out:
                found = False
                for data in t.outputs[0].logprobs:
                    if found:
                        break
                    yes_probs = []
                    no_probs = []
                    yes_cnt = 0
                    no_cnt = 0
                    for k, v in data.items():
                        if self.positive_token.lower() in v.decoded_token.lower():
                            yes_probs.append(np.exp(v.logprob))
                            yes_cnt += 1
                            found = True
                        if self.negative_token.lower() in v.decoded_token.lower():
                            no_probs.append(np.exp(v.logprob))
                            no_cnt += 1
                            found = True
                
                if len(yes_probs) > 0 or len(no_probs) > 0:
                    probs.append((sum(yes_probs) - sum(no_probs))/(yes_cnt + no_cnt))
                
                else:
                    print(f"No logprobs for output: {t.outputs[0].text.strip()}")
                    probs.append(0)
    
            out = [t.outputs[0].text.strip() for t in out]
            if self.custom_threshold is not None:
                threshold = self.custom_threshold
            else:
                return probs, generated_text
            
            print("Output probabilities:", probs)
            predictions = [p >= threshold for p in probs]

            if self.debug:
                print("DEBUGGER OUTPUT")
                for idx in range(len(i1)):
                    print(f"Input 1: {i1[idx]}")
                    print(f"Input 2: {i2[idx]}")
                    print(f"Model Response: {out[idx]}")
                    print(f"Predicted Label: {predictions[idx]}")
                    print(f"Probability Score: {probs[idx]}")
                    print("-----")
            return predictions, probs, generated_text
    
    def extract_label(self, response):
        print(f"Model Response: {response}")
        if self.use_lip_ger:
            if "ja" in response[-10:].lower():
                return True
            else:
                return False
        if self.positive_token.lower() in response[-10:].lower():
            return True
        else:
            return False
        
    def message_packer(self, i1, i2, content_only=False):
        content = self.template.format(text_a=i1, text_b=i2)
        if content_only:
            return content

        chat = [{"role": "user", "content": str(content)}]
        return {"messages": chat}
    


        

if __name__ == "__main__":
    model = GerAV(base_path="./lora_configs/configs_mix",
                    baseline=True,
                    tuned_model="gemma-3-12b-it",
                    tune_dataset="mix_reddit_twitter")
    i1 = ["Hi, ich bin Autor A.", "Some text from author A."]
    i2 = ["Hi, ich bin Autor A.", "Some text from author B."]
    pred, prob, generations = model(i1, i2)
    
    print("Predictions:", pred)
    print("Probabilities:", prob)
    print("Generations:", generations)